"""Unit tests for the daily variance estimators (no database needed)."""
import numpy as np
import pandas as pd

from src.features.volatility import (VARIANCE_FLOOR, add_daily_variance, garman_klass, parkinson,
                                     rogers_satchell, to_annual_vol)


def simulate_bars(n_days=4000, sigma_daily=0.02, steps=390, seed=0):
    """Driftless Brownian motion observed `steps` times a day -> daily open/high/low/close."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0, sigma_daily / np.sqrt(steps), (n_days, steps))
    log_path = np.cumsum(increments.ravel()).reshape(n_days, steps)
    log_open = np.concatenate([[0.0], log_path[:-1, -1]])          # no overnight gap
    intraday = np.column_stack([log_open, log_path])
    return pd.DataFrame({
        "ticker": "SIM",
        "date": pd.bdate_range("2000-01-03", periods=n_days),
        "open": np.exp(log_open),
        "high": np.exp(intraday.max(axis=1)),
        "low": np.exp(intraday.min(axis=1)),
        "close": np.exp(log_path[:, -1]),
    })


def test_estimators_recover_the_true_variance():
    sigma = 0.02
    bars = simulate_bars(sigma_daily=sigma)
    true_var = sigma**2
    # Discrete sampling misses the exact extremes, so range estimators sit slightly below the truth
    assert abs(parkinson(bars["high"], bars["low"]).mean() / true_var - 1) < 0.10
    assert abs(garman_klass(bars["open"], bars["high"], bars["low"], bars["close"]).mean() / true_var - 1) < 0.10
    assert abs(rogers_satchell(bars["open"], bars["high"], bars["low"], bars["close"]).mean() / true_var - 1) < 0.10


def test_range_estimators_are_less_noisy_than_squared_returns():
    bars = add_daily_variance(simulate_bars())
    noise_close = bars["rv_close"].std()
    noise_range = bars["rv"].std()
    assert noise_range < noise_close / 2          # theory: about 7x more efficient


def test_daily_variance_includes_the_overnight_gap():
    bars = simulate_bars(n_days=300)
    gapped = bars.copy()
    gapped.loc[100:, ["open", "high", "low", "close"]] *= 1.10      # +10% overnight jump on day 100
    rv = add_daily_variance(gapped)
    base = add_daily_variance(bars)
    assert np.isclose(rv.loc[100, "rv_overnight"], np.log(gapped.loc[100, "open"] / gapped.loc[99, "close"]) ** 2)
    assert rv.loc[100, "rv"] > base.loc[100, "rv"] + 0.008          # (log 1.1)^2 = 0.0091


def test_close_only_bars_fall_back_to_squared_returns():
    bars = simulate_bars(n_days=50)
    bars.loc[10, ["open", "high", "low"]] = np.nan
    rv = add_daily_variance(bars)
    assert rv.loc[10, "rv_source"] == "close"
    assert np.isclose(rv.loc[10, "rv"], max(rv.loc[10, "rv_close"], VARIANCE_FLOOR))
    assert rv.loc[11, "rv_source"] == "range"


def test_variance_is_floored_and_never_zero():
    bars = simulate_bars(n_days=30)
    bars.loc[5, ["open", "high", "low", "close"]] = bars.loc[4, "close"]   # no move at all
    rv = add_daily_variance(bars)
    assert rv.loc[5, "rv"] == VARIANCE_FLOOR
    assert (rv["rv"].dropna() > 0).all()


def test_no_previous_close_across_a_data_hole():
    bars = simulate_bars(n_days=40)
    bars.loc[20:, "date"] += pd.Timedelta(days=30)                  # one-month hole in the data
    rv = add_daily_variance(bars)
    assert np.isnan(rv.loc[20, "rv_close"]) and np.isnan(rv.loc[20, "rv_overnight"])
    assert np.isnan(rv.loc[20, "rv"])


def test_annualisation():
    assert np.isclose(to_annual_vol(0.01**2), 0.01 * np.sqrt(252) * 100)
