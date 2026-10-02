"""Build the modelling dataset: daily variance, features and forecast targets.

Reads `prices_clean` and writes the table `features` (rebuilt on every run).

Timing convention (the single most important rule of the project)
-----------------------------------------------------------------
A row dated *t* describes what is known **at the close of day t**:

* every feature uses data up to and including day t (prices of the ticker and of
  the other European stocks, which close at the same time), and the VIX of the
  last US session **strictly before** t (the US market closes after Europe);
* every target uses only days **after** t: `target_<h>d` is the log of the average
  daily variance over the next h trading days (t+1 ... t+h).

tests/test_features.py verifies this by rebuilding the features on truncated data.

Usage (from the project root)
-----------------------------
    python -m src.features.build_features
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from src.db import get_engine
from src.features.returns import load_clean_prices
from src.features.volatility import MAX_GAP_DAYS, add_daily_variance

HORIZONS = (1, 5, 22)            # forecast horizons in trading days: day, week, month
WEEK, MONTH, QUARTER = 5, 22, 66
LONG_RUN_MIN_OBS = 250           # one year of history before the long-run level is defined
MIN_MARKET_STOCKS = 20           # stocks needed on a date to compute the market-wide volatility

FEATURE_COLUMNS = [
    # own volatility at three horizons (the HAR structure) plus quarter and long-run level
    "log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q", "log_rv_lt",
    # returns: direction and leverage effect
    "ret_d", "ret_w", "ret_m", "neg_ret_d", "down_share_m",
    # instability of volatility, trading activity
    "vol_of_vol_m", "log_volume_ratio",
    # market-wide volatility (median European stock)
    "mkt_log_rv_d", "mkt_log_rv_w", "mkt_log_rv_m",
    # implied volatility from the US options market, lagged
    "log_vix", "vix_chg_5d",
    # calendar
    "dow",
]
TARGET_COLUMNS = [f"target_{h}d" for h in HORIZONS]
TABLE_COLUMNS = ["ticker", "date", "rv", "rv_source", *FEATURE_COLUMNS, *TARGET_COLUMNS]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _past_mean(series: pd.Series, window: int) -> pd.Series:
    """Mean of the last `window` observations, including the current one."""
    return series.rolling(window, min_periods=window).mean()


def _future_mean(series: pd.Series, horizon: int) -> pd.Series:
    """Mean of the next `horizon` observations, excluding the current one."""
    reversed_mean = series[::-1].rolling(horizon, min_periods=horizon).mean()[::-1]
    return reversed_mean.shift(-1)


def add_own_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from each ticker's own history (rows sorted by ticker, date)."""
    df = df.copy()
    g = df.groupby("ticker", sort=False)

    # Volatility at several horizons
    df["log_rv_d"] = np.log(df["rv"])
    df["log_rv_w"] = np.log(g["rv"].transform(_past_mean, WEEK))
    df["log_rv_m"] = np.log(g["rv"].transform(_past_mean, MONTH))
    df["log_rv_q"] = np.log(g["rv"].transform(_past_mean, QUARTER))
    df["log_rv_lt"] = g["log_rv_d"].transform(lambda s: s.expanding(min_periods=LONG_RUN_MIN_OBS).mean())

    # Returns (close to close; NaN across a hole in the source data)
    gap_days = (df["date"] - g["date"].shift(1)).dt.days
    df["ret_d"] = (np.log(df["close"]) - np.log(g["close"].shift(1))).where(gap_days <= MAX_GAP_DAYS)
    g = df.groupby("ticker", sort=False)
    df["ret_w"] = g["ret_d"].transform(lambda s: s.rolling(WEEK, min_periods=WEEK).sum())
    df["ret_m"] = g["ret_d"].transform(lambda s: s.rolling(MONTH, min_periods=MONTH).sum())
    df["neg_ret_d"] = df["ret_d"].clip(upper=0)

    # Share of the last month's squared returns that came from down days (0.5 = symmetric)
    squared = df["ret_d"] ** 2
    down = squared.where(df["ret_d"] < 0, 0.0).where(df["ret_d"].notna())
    sum_down = down.groupby(df["ticker"], sort=False).transform(lambda s: s.rolling(MONTH, min_periods=MONTH).sum())
    sum_all = squared.groupby(df["ticker"], sort=False).transform(lambda s: s.rolling(MONTH, min_periods=MONTH).sum())
    df["down_share_m"] = (sum_down / sum_all).where(sum_all > 0)

    # How unstable volatility itself has been
    df["vol_of_vol_m"] = g["log_rv_d"].transform(lambda s: s.rolling(MONTH, min_periods=MONTH).std())

    # Trading activity relative to the last month (stocks only: indices have no volume)
    avg_volume = g["volume"].transform(_past_mean, MONTH)
    ratio = df["volume"].where(df["volume"] > 0) / avg_volume.where(avg_volume > 0)
    df["log_volume_ratio"] = np.log(ratio).clip(-3, 3)          # cap data glitches (e.g. a volume of 1 share)

    df["dow"] = df["date"].dt.dayofweek
    return df


def add_market_features(df: pd.DataFrame, min_stocks: int = MIN_MARKET_STOCKS) -> pd.DataFrame:
    """Market-wide volatility: median across stocks of the same-day volatility measures."""
    stocks = df[df["asset_type"] == "stock"]
    counts = stocks.groupby("date")["log_rv_d"].count()
    valid_dates = counts[counts >= min_stocks].index
    market = (stocks[stocks["date"].isin(valid_dates)]
              .groupby("date")[["log_rv_d", "log_rv_w", "log_rv_m"]].median()
              .add_prefix("mkt_"))
    return df.merge(market, left_on="date", right_index=True, how="left")


def add_vix_features(df: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """VIX of the last US session strictly before each row's date, and its 5-session change."""
    v = vix[["date", "close"]].dropna().sort_values("date").rename(columns={"date": "vix_date"})
    v["log_vix"] = np.log(v["close"])
    v["vix_chg_5d"] = v["log_vix"] - v["log_vix"].shift(5)
    out = pd.merge_asof(
        df.sort_values("date"), v[["vix_date", "log_vix", "vix_chg_5d"]],
        left_on="date", right_on="vix_date",
        allow_exact_matches=False,            # same-day VIX is NOT known at the European close
    )
    return out.drop(columns="vix_date").sort_values(["ticker", "date"], ignore_index=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Log of the average daily variance over the next h trading days, for each horizon."""
    df = df.copy()
    g = df.groupby("ticker", sort=False)
    for h in HORIZONS:
        df[f"target_{h}d"] = np.log(g["rv"].transform(_future_mean, h))
    return df


def build_features(prices: pd.DataFrame, min_market_stocks: int = MIN_MARKET_STOCKS) -> pd.DataFrame:
    """Full pipeline. Pure function: no database access.

    `prices` holds the clean prices of all series (columns: ticker, date, open,
    high, low, close, volume, asset_type). The VIX rows (asset_type 'external')
    are used as a feature source only; the output has one row per stock / index
    and trading day.
    """
    vix = prices[prices["asset_type"] == "external"]
    series = prices[prices["asset_type"] != "external"]

    df = add_daily_variance(series)
    df = add_own_features(df)
    df = add_market_features(df, min_market_stocks)
    df = add_vix_features(df, vix)
    df = add_targets(df)
    return df


# ---------------------------------------------------------------------------
# Database I/O
# ---------------------------------------------------------------------------
def to_copy_buffer(df: pd.DataFrame, columns: list[str]) -> io.StringIO:
    """CSV text in the format expected by PostgreSQL COPY (empty field = NULL)."""
    out = df[columns].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    if "dow" in out:
        out["dow"] = out["dow"].astype("Int64")
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, header=False, na_rep="")
    buffer.seek(0)
    return buffer


def save(engine: Engine, features: pd.DataFrame) -> None:
    """Replace the content of the features table (fast bulk load with COPY)."""
    buffer = to_copy_buffer(features, TABLE_COLUMNS)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("TRUNCATE features")
            cur.copy_expert(
                f"COPY features ({', '.join(TABLE_COLUMNS)}) FROM STDIN WITH (FORMAT csv, NULL '')",
                buffer,
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
    prices = load_clean_prices(engine)
    logger.info("Loaded %d clean rows", len(prices))

    features = build_features(prices)
    save(engine, features)

    complete = features.dropna(subset=[c for c in FEATURE_COLUMNS if c != "log_volume_ratio"] + TARGET_COLUMNS)
    logger.info("features: %d rows for %d series | %d rows with all features and targets",
                len(features), features["ticker"].nunique(), len(complete))
    share = features["rv_source"].value_counts(normalize=True)
    logger.info("Daily variance from the intraday range: %.1f%% of rows (close-to-close fallback: %.1f%%)",
                100 * share.get("range", 0), 100 * share.get("close", 0))


if __name__ == "__main__":
    main()
