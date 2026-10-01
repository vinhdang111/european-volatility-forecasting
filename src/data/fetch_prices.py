"""Download daily OHLCV price data for the project universe from Yahoo Finance.

Raw data is saved exactly as received (no adjustment, no filling) so that
every cleaning decision is made explicitly in Step 2.

Outputs
-------
data/raw/prices.parquet         long format: date, ticker, open, high, low,
                                close, adj_close, volume
data/raw/download_report.csv    one row per ticker: status, history length,
                                missing values and suspicious jumps

Usage (from the project root)
-----------------------------
    python -m src.data.fetch_prices
    python -m src.data.fetch_prices --start 2000-01-01 --end 2026-09-30
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATH = PROJECT_ROOT / "data" / "universe.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
MAX_RETRIES = 3
PAUSE_SECONDS = 1.0       # be polite to Yahoo between requests
JUMP_THRESHOLD = 0.5      # |daily close change| > 50% is flagged as suspicious

logger = logging.getLogger(__name__)


def load_universe(path: Path = UNIVERSE_PATH) -> pd.DataFrame:
    """Read the list of tickers (stocks, indices and external series)."""
    universe = pd.read_csv(path)
    duplicated = universe["ticker"][universe["ticker"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicated tickers in universe: {list(duplicated)}")
    return universe


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


def summarise(df: pd.DataFrame, ticker: str) -> dict:
    """Basic quality figures for one ticker, used in the download report."""
    if df.empty:
        return {"ticker": ticker, "status": "NO DATA"}

    close = df["close"]
    jumps = close.pct_change(fill_method=None).abs() > JUMP_THRESHOLD
    first, last = df["date"].min(), df["date"].max()
    return {
        "ticker": ticker,
        "status": "OK",
        "first_date": first.date(),
        "last_date": last.date(),
        "years_of_history": round((last - first).days / 365.25, 1),
        "n_rows": len(df),
        "missing_ohlc_rows": int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
        "zero_volume_rows": int((df["volume"] == 0).sum()),
        "jumps_over_50pct": int(jumps.sum()),
    }


def main(start: str = "2000-01-01", end: str | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    universe = load_universe()
    logger.info("Downloading %d tickers from %s to %s", len(universe), start, end or "today")

    frames, report = [], []
    for i, row in enumerate(universe.itertuples(index=False), start=1):
        df = download_ticker(row.ticker, start, end)
        summary = summarise(df, row.ticker)
        summary.update(name=row.name, asset_type=row.asset_type, country=row.country)
        report.append(summary)
        if not df.empty:
            frames.append(df)
        logger.info("[%2d/%d] %-12s %s", i, len(universe), row.ticker,
                    f"{summary.get('n_rows', 0)} rows" if summary["status"] == "OK" else "NO DATA")
        time.sleep(PAUSE_SECONDS)

    if not frames:
        raise RuntimeError("No data downloaded. Check your internet connection or yfinance version.")

    prices = pd.concat(frames, ignore_index=True)
    prices_path = RAW_DIR / "prices.parquet"
    prices.to_parquet(prices_path, index=False)

    report_df = pd.DataFrame(report)
    front = ["ticker", "name", "asset_type", "country", "status"]
    report_df = report_df[front + [c for c in report_df.columns if c not in front]]
    report_path = RAW_DIR / "download_report.csv"
    report_df.to_csv(report_path, index=False)

    ok = report_df["status"].eq("OK")
    logger.info("Saved %d rows for %d/%d tickers to %s", len(prices), ok.sum(), len(report_df), prices_path)
    if (~ok).any():
        logger.warning("No data for: %s", ", ".join(report_df.loc[~ok, "ticker"]))

    pd.set_option("display.width", 200)
    print("\nDownload report (sorted by history length):")
    print(report_df.sort_values("years_of_history", na_position="first").to_string(index=False))
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-01-01", help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="last date, exclusive (default: today)")
    args = parser.parse_args()
    main(args.start, args.end)
