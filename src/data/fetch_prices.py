"""Download daily OHLCV price data for the project universe from Yahoo Finance
and store it in PostgreSQL.

Raw data is saved exactly as received (no adjustment, no filling) so that
every cleaning decision is made explicitly in Step 2.

Tables written (see sql/schema.sql)
-----------------------------------
prices_raw      one row per ticker and trading day (upsert: re-running updates rows)
download_runs   one row per run
download_log    one row per ticker per run: status, history length,
                missing values and suspicious jumps

Usage (from the project root, after `python -m src.data.setup_db`)
------------------------------------------------------------------
    python -m src.data.fetch_prices
    python -m src.data.fetch_prices --start 2000-01-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import yfinance as yf
from psycopg2.extras import execute_values
from sqlalchemy import Engine, text

from src.data.setup_db import sync_universe
from src.db import get_engine

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
MAX_RETRIES = 3
PAUSE_SECONDS = 1.0       # be polite to Yahoo between requests
JUMP_THRESHOLD = 0.5      # |daily close change| > 50% is flagged as suspicious

UPSERT_PRICES = """
    INSERT INTO prices_raw (ticker, date, open, high, low, close, adj_close, volume)
    VALUES %s
    ON CONFLICT (ticker, date) DO UPDATE SET
        open          = EXCLUDED.open,
        high          = EXCLUDED.high,
        low           = EXCLUDED.low,
        close         = EXCLUDED.close,
        adj_close     = EXCLUDED.adj_close,
        volume        = EXCLUDED.volume,
        downloaded_at = now()
"""

LOG_COLUMNS = ["status", "first_date", "last_date", "years_of_history", "n_rows",
               "missing_ohlc_rows", "zero_volume_rows", "jumps_over_50pct"]

INSERT_LOG = text(f"""
    INSERT INTO download_log (run_id, ticker, {", ".join(LOG_COLUMNS)})
    VALUES (:run_id, :ticker, {", ".join(":" + c for c in LOG_COLUMNS)})
""")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_ticker(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download one ticker and return a tidy DataFrame (empty if no data).

    Retries a few times because Yahoo occasionally drops requests.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,   # keep raw OHLC + Adj Close; adjust later on purpose
                actions=False,
                progress=False,
                threads=False,
            )
            return _tidy(raw, ticker)
        except Exception as exc:  # network errors, Yahoo hiccups
            logger.warning("%s: attempt %d/%d failed (%s)", ticker, attempt, MAX_RETRIES, exc)
            time.sleep(PAUSE_SECONDS * attempt * 2)
    return _tidy(pd.DataFrame(), ticker)


def _tidy(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert yfinance output to long format with snake_case columns."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *PRICE_COLUMNS])

    df = raw.copy()
    # Recent yfinance versions return MultiIndex columns (field, ticker) even for one ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    df = df.reindex(columns=PRICE_COLUMNS)
    df.columns.name = None

    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="all", subset=["open", "high", "low", "close"])

    df = df.reset_index()
    df.insert(1, "ticker", ticker)
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Basic quality figures for one ticker, stored in download_log."""
    if df.empty:
        return {"status": "NO DATA", **{c: None for c in LOG_COLUMNS[1:]}}

    close = df["close"]
    jumps = close.pct_change(fill_method=None).abs() > JUMP_THRESHOLD
    first, last = df["date"].min(), df["date"].max()
    return {
        "status": "OK",
        "first_date": first.date(),
        "last_date": last.date(),
        "years_of_history": round((last - first).days / 365.25, 1),
        "n_rows": len(df),
        "missing_ohlc_rows": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "jumps_over_50pct": int(jumps.sum()),
    }


# ---------------------------------------------------------------------------
# Database writes
# ---------------------------------------------------------------------------
def to_rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> list of tuples with plain Python types (NaN -> NULL)."""
    out = df[["ticker", "date", *PRICE_COLUMNS]].copy()
    out["date"] = out["date"].dt.date
    out["volume"] = out["volume"].round().astype("Int64")
    out = out.astype(object).where(out.notna(), None)
    return list(out.itertuples(index=False, name=None))


def save_prices(engine: Engine, df: pd.DataFrame) -> None:
    """Upsert one ticker's prices into prices_raw (fast batch insert)."""
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, UPSERT_PRICES, to_rows(df), page_size=5000)
        raw.commit()
    finally:
        raw.close()


def start_run(engine: Engine, start: str, end: str | None, n_tickers: int) -> int:
    with engine.begin() as conn:
        return conn.execute(
            text("""INSERT INTO download_runs (start_date, end_date, n_tickers)
                    VALUES (:s, :e, :n) RETURNING run_id"""),
            {"s": start, "e": end, "n": n_tickers},
        ).scalar_one()


def log_ticker(engine: Engine, run_id: int, ticker: str, summary: dict) -> None:
    with engine.begin() as conn:
        conn.execute(INSERT_LOG, {"run_id": run_id, "ticker": ticker, **summary})


def finish_run(engine: Engine, run_id: int, n_ok: int, n_rows: int) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""UPDATE download_runs
                    SET finished_at = now(), n_ok = :ok, n_rows = :rows
                    WHERE run_id = :id"""),
            {"ok": n_ok, "rows": n_rows, "id": run_id},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(start: str = "2000-01-01", end: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    engine = get_engine()

    universe = sync_universe(engine)          # keeps the DB in line with data/universe.csv
    run_id = start_run(engine, start, end, len(universe))
    logger.info("Run %d: downloading %d tickers from %s to %s",
                run_id, len(universe), start, end or "today")

    n_ok = n_rows = 0
    failed = []
    for i, ticker in enumerate(universe["ticker"], start=1):
        df = download_ticker(ticker, start, end)
        if not df.empty:
            save_prices(engine, df)
            n_ok += 1
            n_rows += len(df)
        else:
            failed.append(ticker)
        log_ticker(engine, run_id, ticker, summarise(df))
        logger.info("[%2d/%d] %-12s %s", i, len(universe), ticker,
                    f"{len(df)} rows" if not df.empty else "NO DATA")
        time.sleep(PAUSE_SECONDS)

    finish_run(engine, run_id, n_ok, n_rows)
    logger.info("Run %d finished: %d rows for %d/%d tickers saved to prices_raw",
                run_id, n_rows, n_ok, len(universe))
    if failed:
        logger.warning("No data for: %s", ", ".join(failed))

    report = pd.read_sql("SELECT * FROM latest_download_report", engine)
    pd.set_option("display.width", 200)
    print("\nDownload report (view: latest_download_report):")
    print(report.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-01-01", help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="last date, exclusive (default: today)")
    args = parser.parse_args()
    main(args.start, args.end)
