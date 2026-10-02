"""Returns and simple realised-volatility measures built from clean prices.

Used by the exploratory analysis (Step 3) and by feature engineering (Step 4).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import Engine

TRADING_DAYS = 252
MAX_GAP_DAYS = 7   # a return spanning more than a week is not a daily return

LOAD_CLEAN = """
    SELECT p.ticker, p.date, p.open, p.high, p.low, p.close, p.volume,
           u.name, u.asset_type, u.country, u.sector
    FROM prices_clean p
    JOIN universe u USING (ticker)
    ORDER BY p.ticker, p.date
"""


def load_clean_prices(engine: Engine) -> pd.DataFrame:
    """Clean prices joined with the universe metadata, sorted by ticker and date."""
    return pd.read_sql(LOAD_CLEAN, engine, parse_dates=["date"])


def add_log_returns(prices: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> pd.DataFrame:
    """Add the daily log return of the close, `ret`, for each ticker.

    Returns are computed from `close` (split-adjusted), not `adj_close`
    (see notebooks/01_data_quality.ipynb). When two consecutive rows are more
    than `max_gap_days` calendar days apart (a hole in the source data), the
    return is set to NaN: it would be a multi-day return, not a daily one.
    """
    df = prices.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")
    df["ret"] = np.log(df["close"]) - np.log(grouped["close"].shift(1))
    gap = (df["date"] - grouped["date"].shift(1)).dt.days
    df.loc[gap > max_gap_days, "ret"] = np.nan
    return df


def rolling_volatility(returns: pd.Series, window: int = 21, min_periods: int | None = None) -> pd.Series:
    """Annualised rolling volatility (in %) of one ticker's daily log returns."""
    min_periods = window if min_periods is None else min_periods
    return returns.rolling(window, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS) * 100


def add_rolling_volatility(df: pd.DataFrame, window: int = 21, column: str | None = None) -> pd.DataFrame:
    """Add the rolling volatility of `ret` per ticker (column `vol_<window>d` by default)."""
    column = column or f"vol_{window}d"
    out = df.copy()
    out[column] = out.groupby("ticker")["ret"].transform(lambda r: rolling_volatility(r, window))
    return out
