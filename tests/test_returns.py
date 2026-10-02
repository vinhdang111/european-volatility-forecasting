"""Unit tests for returns and rolling volatility (no database needed)."""
import numpy as np
import pandas as pd

from src.features.returns import add_log_returns, add_rolling_volatility, rolling_volatility


def make(closes, ticker="A", dates=None):
    dates = pd.bdate_range("2024-01-01", periods=len(closes)) if dates is None else pd.to_datetime(dates)
    return pd.DataFrame({"ticker": ticker, "date": dates, "close": np.asarray(closes, dtype=float)})


def test_log_return_values():
    df = add_log_returns(make([100, 110, 99]))
    assert np.isnan(df["ret"].iloc[0])
    assert np.isclose(df["ret"].iloc[1], np.log(1.10))
    assert np.isclose(df["ret"].iloc[2], np.log(0.90))


def test_returns_do_not_cross_tickers():
    df = add_log_returns(pd.concat([make([100, 101], "A"), make([50, 51], "B")]))
    first_rows = df.groupby("ticker").head(1)
    assert first_rows["ret"].isna().all()


def test_return_across_long_gap_is_nan():
    df = add_log_returns(make([100, 101, 120], dates=["2024-01-02", "2024-01-03", "2024-02-15"]))
    assert not np.isnan(df["ret"].iloc[1])
    assert np.isnan(df["ret"].iloc[2])


def test_long_weekend_is_not_a_gap():
    # Thursday before Easter -> Tuesday after: 5 calendar days
    df = add_log_returns(make([100, 102], dates=["2024-03-28", "2024-04-02"]))
    assert np.isclose(df["ret"].iloc[1], np.log(1.02))


def test_rolling_volatility_is_annualised_percent():
    rng = np.random.default_rng(1)
    daily_sigma = 0.01
    ret = pd.Series(rng.normal(0, daily_sigma, 5000))
    vol = rolling_volatility(ret, window=5000).iloc[-1]
    assert abs(vol - daily_sigma * np.sqrt(252) * 100) < 0.5      # about 15.9%


def test_rolling_volatility_uses_only_past_data():
    rng = np.random.default_rng(2)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    full = add_rolling_volatility(add_log_returns(make(closes)), window=21)
    truncated = add_rolling_volatility(add_log_returns(make(closes[:40])), window=21)
    # the value on day 40 must not change when later days are added
    assert np.isclose(full["vol_21d"].iloc[39], truncated["vol_21d"].iloc[39])
