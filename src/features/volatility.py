"""Daily variance estimators.

With daily data, volatility on a single day cannot be observed directly. The
squared close-to-close return is an unbiased but very noisy estimate. The
high-low range carries much more information about how far the price travelled
during the day, so range-based estimators are several times more precise.

All functions return a **daily variance** (decimal units, e.g. 0.0001 = 1% daily
volatility). Use `to_annual_vol` to express it as an annualised volatility in %.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MAX_GAP_DAYS = 7                 # a "previous close" older than this is not yesterday's close
VARIANCE_FLOOR = 1e-6            # (0.1% daily move)^2, about 1.6% annualised volatility
LN2 = np.log(2.0)


def to_annual_vol(variance):
    """Daily variance -> annualised volatility in percent."""
    return np.sqrt(variance * TRADING_DAYS) * 100


def squared_return(close: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Close-to-close squared log return: unbiased, but extremely noisy."""
    return np.log(close / prev_close) ** 2


def parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    """Parkinson (1980): uses only the high-low range. Intraday variance only."""
    return np.log(high / low) ** 2 / (4 * LN2)


def garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Garman-Klass (1980): range plus open-to-close move. Intraday variance only."""
    return 0.5 * np.log(high / low) ** 2 - (2 * LN2 - 1) * np.log(close / open_) ** 2


def rogers_satchell(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Rogers-Satchell (1991): unbiased when the price drifts. Intraday variance only."""
    return (np.log(high / close) * np.log(high / open_)
            + np.log(low / close) * np.log(low / open_))


def overnight_variance(open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    """Squared overnight return (previous close to today's open)."""
    return np.log(open_ / prev_close) ** 2


def add_daily_variance(prices: pd.DataFrame, floor: float = VARIANCE_FLOOR) -> pd.DataFrame:
    """Add the daily variance used throughout the project, `rv`, for each ticker.

    rv = overnight variance + Garman-Klass intraday variance
         (the Garman-Klass estimator extended with the overnight jump, as in
         Yang and Zhang, 2000), which measures close-to-close risk using the full
         open/high/low/close information.

    When the range is missing (close-only bars) the squared close-to-close return
    is used instead. `rv_source` records which one was used: 'range' or 'close'.
    A small floor avoids log(0) on days with no price change.

    Also adds the individual estimators (`rv_close`, `rv_parkinson`, `rv_gk`,
    `rv_overnight`) for comparison. Rows must contain ticker, date, open, high,
    low, close.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")
    prev_close = grouped["close"].shift(1)
    gap_days = (df["date"] - grouped["date"].shift(1)).dt.days
    prev_close = prev_close.where(gap_days <= MAX_GAP_DAYS)       # no "yesterday" across a data hole

    df["rv_close"] = squared_return(df["close"], prev_close)
    df["rv_parkinson"] = parkinson(df["high"], df["low"])
    df["rv_gk"] = garman_klass(df["open"], df["high"], df["low"], df["close"])
    df["rv_overnight"] = overnight_variance(df["open"], prev_close)

    range_based = df["rv_overnight"] + df["rv_gk"]
    df["rv"] = range_based.fillna(df["rv_close"])
    df["rv_source"] = np.where(range_based.notna(), "range", "close")
    df.loc[df["rv"].isna(), "rv_source"] = None
    df["rv"] = df["rv"].clip(lower=floor)
    return df
