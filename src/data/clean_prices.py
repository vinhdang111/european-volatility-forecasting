"""Clean the raw prices and store them in PostgreSQL.

Reads `prices_raw`, applies a fixed sequence of documented rules and writes:

prices_clean   cleaned daily prices (rebuilt from scratch on every run)
cleaning_log   one row per change: ticker, date, rule, action, detail

Every rule was designed from issues actually found in the Yahoo Finance data
(see notebooks/01_data_quality.ipynb):

1. before_valid_from   drop history before the ticker's `valid_from` date
                       (set in data/universe.csv when the early history is
                       unreliable, e.g. Shell before 2005).
2. incomplete_session  drop bars dated on (or after) the download day: the
                       session may still have been trading.
3. stale_bar           drop non-trading or missing days that Yahoo fills with
                       an old price: zero volume and either (a) the same close
                       as the previous row, or (b) a flat open = high = low
                       with a close outside that range (mis-adjusted rows).
4. price_spike         drop isolated bad prints: a close more than 40% away
                       from BOTH the median of the 5 previous and the 5 next
                       closes (a spike that immediately reverts). Real crashes
                       persist and are kept. Not applied to the VIX.
5. range_repair        widen high/low when the open or close lies outside the
                       reported [low, high] range.
6. close_only_bar      set open/high/low to NULL when open = high = low = close
                       (the source only provided a closing price).
7. invalid_adj_close   set adj_close to NULL when it is zero or negative.

Volume is set to NULL (not logged) for indices and the VIX, where Yahoo does
not report a meaningful volume.

Usage (from the project root)
-----------------------------
    python -m src.data.clean_prices
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values
from sqlalchemy import Engine

from src.db import get_engine

SPIKE_THRESHOLD = 1.4     # 40% away from both neighbouring medians
SPIKE_WINDOW = 5          # trading days on each side
SPIKE_MAX_PASSES = 5      # repeat until no new spike is found

PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]
LOG_COLUMNS = ["ticker", "date", "rule", "action", "detail"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(values: pd.Series) -> pd.Series:
    """Format numbers for the log (object dtype, so string concatenation works even when empty)."""
    return pd.Series(["NULL" if pd.isna(x) else f"{x:.4g}" for x in values], index=values.index, dtype=object)


def _log(rows: pd.DataFrame, rule: str, action: str, detail: pd.Series | str) -> pd.DataFrame:
    """Build cleaning-log entries for the given rows."""
    out = rows[["ticker", "date"]].copy()
    out["rule"] = rule
    out["action"] = action
    out["detail"] = detail
    return out


def _prev_close(df: pd.DataFrame) -> pd.Series:
    return df.groupby("ticker")["close"].shift(1)


# ---------------------------------------------------------------------------
# Rules: each takes the current frame and returns (frame, log entries)
# ---------------------------------------------------------------------------
def drop_before_valid_from(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["valid_from"].notna() & (df["date"] < df["valid_from"])
    detail = "before valid_from " + df.loc[mask, "valid_from"].dt.strftime("%Y-%m-%d").astype(object)
    return df[~mask], _log(df[mask], "before_valid_from", "dropped", detail)


def drop_incomplete_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = df["date"] >= df["download_day"]
    detail = "downloaded on " + df.loc[mask, "download_day"].dt.strftime("%Y-%m-%d").astype(object) + ", session may be unfinished"
    return df[~mask], _log(df[mask], "incomplete_session", "dropped", detail)


def drop_stale_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = df["asset_type"] != "external"          # VIX never reports volume
    zero_volume = df["volume"].fillna(0) == 0
    same_close = df["close"] == _prev_close(df)
    flat_ohl = (df["open"] == df["high"]) & (df["high"] == df["low"])
    close_outside = (df["close"] - df["high"]).abs() > 1e-9 * df["close"]
    stale = eligible & zero_volume & same_close
    misaligned = eligible & zero_volume & flat_ohl & close_outside & ~stale
    mask = stale | misaligned

    close_txt = "volume 0, close " + df.loc[mask, "close"].pipe(_fmt)
    misaligned_txt = close_txt + " inconsistent with flat open/high/low " + df.loc[mask, "high"].pipe(_fmt)
    detail = (close_txt + " repeated from previous day").where(stale[mask], misaligned_txt)
    return df[~mask], _log(df[mask], "stale_bar", "dropped", detail)


def _spike_mask(close: pd.Series) -> pd.Series:
    before = close.shift(1).rolling(SPIKE_WINDOW, min_periods=1).median()
    after = close[::-1].shift(1).rolling(SPIKE_WINDOW, min_periods=1).median()[::-1]
    up = (close / before > SPIKE_THRESHOLD) & (close / after > SPIKE_THRESHOLD)
    down = (close / before < 1 / SPIKE_THRESHOLD) & (close / after < 1 / SPIKE_THRESHOLD)
    return up | down


def drop_price_spikes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logs = []
    for _ in range(SPIKE_MAX_PASSES):
        mask = df.groupby("ticker")["close"].transform(_spike_mask).astype(bool)
        mask &= df["asset_type"] != "external"
        if not mask.any():
            break
        prev, nxt = _prev_close(df), df.groupby("ticker")["close"].shift(-1)
        detail = ("close " + df.loc[mask, "close"].pipe(_fmt) + " vs neighbours "
                  + prev[mask].pipe(_fmt) + " / " + nxt[mask].pipe(_fmt)
                  + ", volume " + df.loc[mask, "volume"].pipe(_fmt))
        logs.append(_log(df[mask], "price_spike", "dropped", detail))
        df = df[~mask]
    log = pd.concat(logs) if logs else pd.DataFrame(columns=LOG_COLUMNS)
    return df, log


def repair_ranges(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    has_range = df[["open", "high", "low"]].notna().all(axis=1)
    new_high = df[["open", "high", "close"]].max(axis=1)
    new_low = df[["open", "low", "close"]].min(axis=1)
    mask = has_range & ((new_high > df["high"]) | (new_low < df["low"]))

    detail = ("range [" + df.loc[mask, "low"].pipe(_fmt) + ", " + df.loc[mask, "high"].pipe(_fmt)
              + "] -> [" + new_low[mask].pipe(_fmt) + ", " + new_high[mask].pipe(_fmt) + "]")
    log = _log(df[mask], "range_repair", "repaired", detail)
    df.loc[mask, "high"] = new_high[mask]
    df.loc[mask, "low"] = new_low[mask]
    return df, log


def null_close_only_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    mask = (df["open"] == df["high"]) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    log = _log(df[mask], "close_only_bar", "set_null", "open = high = low = close: no intraday range")
    df.loc[mask, ["open", "high", "low"]] = np.nan
    return df, log


def null_invalid_adj_close(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    mask = df["adj_close"] <= 0
    detail = "adj_close " + df.loc[mask, "adj_close"].pipe(_fmt) + " <= 0"
    log = _log(df[mask], "invalid_adj_close", "set_null", detail)
    df.loc[mask, "adj_close"] = np.nan
    return df, log


RULES = [
    drop_before_valid_from,
    drop_incomplete_sessions,
    drop_stale_bars,
    drop_price_spikes,
    repair_ranges,
    null_close_only_bars,
    null_invalid_adj_close,
]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all rules in order. Pure function: no database access.

    `raw` needs the prices_raw columns plus `asset_type`, `valid_from` and
    `download_day` (all three come from load_raw).
    Returns (clean prices, cleaning log).
    """
    df = raw.sort_values(["ticker", "date"]).reset_index(drop=True)
    logs = []
    for rule in RULES:
        df, log = rule(df)
        logs.append(log)
        logger.info("%-26s %6d rows", rule.__name__, len(log))

    df = df.copy()
    df.loc[df["asset_type"] != "stock", "volume"] = np.nan
    log = pd.concat(logs, ignore_index=True)[LOG_COLUMNS].sort_values(["ticker", "date"])
    return df[["ticker", "date", *PRICE_COLUMNS]].reset_index(drop=True), log.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------
LOAD_RAW = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.adj_close, p.volume,
           u.asset_type,
           u.valid_from,
           (p.downloaded_at AT TIME ZONE 'Europe/Paris')::date AS download_day
    FROM prices_raw p
    JOIN universe u USING (ticker)
"""


def load_raw(engine: Engine) -> pd.DataFrame:
    df = pd.read_sql(LOAD_RAW, engine, parse_dates=["date", "valid_from", "download_day"])
    logger.info("Loaded %d raw rows for %d tickers", len(df), df["ticker"].nunique())
    return df


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    out = df[columns].copy()
    out["date"] = out["date"].dt.date
    if "volume" in out:
        out["volume"] = out["volume"].round().astype("Int64")
    out = out.astype(object).where(out.notna(), None)
    return list(out.itertuples(index=False, name=None))


def save(engine: Engine, prices: pd.DataFrame, log: pd.DataFrame) -> None:
    """Replace the content of prices_clean and cleaning_log in one transaction."""
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("TRUNCATE prices_clean, cleaning_log RESTART IDENTITY")
            execute_values(
                cur,
                f"INSERT INTO prices_clean (ticker, date, {', '.join(PRICE_COLUMNS)}) VALUES %s",
                _rows(prices, ["ticker", "date", *PRICE_COLUMNS]),
                page_size=5000,
            )
            execute_values(
                cur,
                f"INSERT INTO cleaning_log ({', '.join(LOG_COLUMNS)}) VALUES %s",
                _rows(log, LOG_COLUMNS),
                page_size=5000,
            )
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    engine = get_engine()
    raw = load_raw(engine)
    prices, log = clean(raw)
    save(engine, prices, log)
    logger.info("prices_clean: %d rows (%d dropped) | cleaning_log: %d entries",
                len(prices), len(raw) - len(prices), len(log))

    pd.set_option("display.width", 200)
    print("\nCleaning summary (view: cleaning_summary):")
    print(pd.read_sql("SELECT * FROM cleaning_summary", engine).to_string(index=False))


if __name__ == "__main__":
    main()
