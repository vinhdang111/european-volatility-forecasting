"""Unit tests for the cleaning rules (no database needed).

Run from the project root:  python -m pytest
"""
import numpy as np
import pandas as pd

from src.data.clean_prices import clean


def make_prices(closes, ticker="TEST.PA", asset_type="stock", volume=1_000, start="2024-01-01"):
    """Well-behaved daily bars around the given closes."""
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "ticker": ticker,
        "date": dates,
        "open": closes,
        "high": closes * 1.01,
        "low": closes * 0.99,
        "close": closes,
        "adj_close": closes * 0.9,
        "volume": float(volume),
        "asset_type": asset_type,
        "valid_from": pd.NaT,
        "range_valid_from": pd.NaT,
        "download_day": dates[-1] + pd.Timedelta(days=30),
    })


def rules_applied(log, rule):
    return log[log["rule"] == rule]


def test_clean_data_is_left_untouched():
    raw = make_prices(np.linspace(100, 110, 30))
    prices, log = clean(raw)
    assert len(prices) == 30
    assert log.empty


def test_incomplete_session_is_dropped():
    raw = make_prices(np.linspace(100, 110, 10))
    raw["download_day"] = raw["date"].iloc[-1]          # downloaded during the last session
    prices, log = clean(raw)
    assert len(prices) == 9
    assert len(rules_applied(log, "incomplete_session")) == 1


def test_rows_before_valid_from_are_dropped():
    raw = make_prices(np.linspace(100, 110, 10))
    raw["valid_from"] = raw["date"].iloc[4]
    prices, log = clean(raw)
    assert prices["date"].min() == raw["date"].iloc[4]
    assert len(rules_applied(log, "before_valid_from")) == 4


def test_stale_holiday_bar_is_dropped():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[5, ["open", "high", "low", "close"]] = raw.loc[4, "close"]
    raw.loc[5, "volume"] = 0
    prices, log = clean(raw)
    assert raw.loc[5, "date"] not in set(prices["date"])
    assert len(rules_applied(log, "stale_bar")) == 1


def test_unchanged_close_with_volume_is_kept():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[5, "close"] = raw.loc[4, "close"]            # real trading day, price unchanged
    prices, log = clean(raw)
    assert len(prices) == 10
    assert rules_applied(log, "stale_bar").empty


def test_isolated_spike_is_dropped_but_real_crash_is_kept():
    closes = [100.0] * 15
    closes[7] = 200.0                                    # bad print that reverts the next day
    spike = make_prices(closes)
    crash = make_prices([100.0] * 7 + [40.0] * 8, ticker="CRASH.PA")
    prices, log = clean(pd.concat([spike, crash], ignore_index=True))
    spikes = rules_applied(log, "price_spike")
    assert list(spikes["ticker"]) == ["TEST.PA"]
    assert (prices.loc[prices["ticker"] == "CRASH.PA", "close"] == 40.0).sum() == 8


def test_spike_rule_is_not_applied_to_vix():
    closes = [15.0] * 15
    closes[7] = 40.0                                     # VIX can genuinely spike for a day
    raw = make_prices(closes, ticker="^VIX", asset_type="external", volume=0)
    prices, log = clean(raw)
    assert len(prices) == 15
    assert rules_applied(log, "price_spike").empty


def test_range_is_repaired_when_close_is_outside():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[3, "close"] = raw.loc[3, "high"] * 1.02
    prices, log = clean(raw)
    row = prices.loc[3]
    assert row["high"] == row["close"]
    assert len(rules_applied(log, "range_repair")) == 1


def test_unit_error_in_low_drops_the_range_but_keeps_the_close():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[4, "low"] = raw.loc[4, "low"] / 100            # low quoted in pounds instead of pence
    prices, log = clean(raw)
    assert prices.loc[4, ["open", "high", "low"]].isna().all()
    assert prices.loc[4, "close"] == raw.loc[4, "close"]
    assert len(rules_applied(log, "implausible_range")) == 1


def test_bad_high_print_drops_the_range():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[4, "high"] = raw.loc[4, "close"] * 1.30         # 30% above both open and close
    prices, log = clean(raw)
    assert prices.loc[4, ["open", "high", "low"]].isna().all()
    assert len(rules_applied(log, "implausible_range")) == 1


def test_genuine_intraday_crash_keeps_its_range():
    raw = make_prices([100.0] * 4 + [72.0] * 6)              # the price stays down after the crash
    raw.loc[4, ["open", "high"]] = [100.0, 101.0]            # crash day: opens at 100 ...
    raw.loc[4, ["low", "close"]] = [72.0, 72.0]              # ... and closes 28% lower, at the low
    prices, log = clean(raw)
    assert prices.loc[4, "low"] == 72.0
    assert prices.loc[4, "high"] == 101.0
    assert rules_applied(log, "implausible_range").empty


def test_implausible_range_rule_is_not_applied_to_vix():
    raw = make_prices([15.0] * 10, ticker="^VIX", asset_type="external", volume=0)
    raw.loc[4, "high"] = 22.0                                # intraday spike, back down by the close
    prices, log = clean(raw)
    assert prices.loc[4, "high"] == 22.0
    assert rules_applied(log, "implausible_range").empty


def test_range_before_range_valid_from_is_discarded():
    raw = make_prices(np.linspace(100, 110, 10))
    raw["range_valid_from"] = raw["date"].iloc[6]
    prices, log = clean(raw)
    assert prices.loc[:5, "high"].isna().all()
    assert prices.loc[6:, "high"].notna().all()
    assert len(prices) == 10                                 # closes are all kept
    assert len(rules_applied(log, "unreliable_range")) == 6


def test_close_only_bar_gets_null_range():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[3, ["open", "high", "low"]] = raw.loc[3, "close"]
    prices, log = clean(raw)
    assert prices.loc[3, ["open", "high", "low"]].isna().all()
    assert prices.loc[3, "close"] == raw.loc[3, "close"]


def test_negative_adj_close_is_set_to_null():
    raw = make_prices(np.linspace(100, 110, 10))
    raw.loc[2, "adj_close"] = -3.5
    prices, log = clean(raw)
    assert pd.isna(prices.loc[2, "adj_close"])
    assert len(rules_applied(log, "invalid_adj_close")) == 1


def test_volume_is_null_for_indices():
    raw = make_prices(np.linspace(100, 110, 10), ticker="^IDX", asset_type="index", volume=0)
    prices, _ = clean(raw)
    assert prices["volume"].isna().all()


def test_output_satisfies_database_constraints():
    rng = np.random.default_rng(0)
    raw = make_prices(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    raw.loc[rng.choice(300, 20, replace=False), "close"] *= 1.03   # inject inconsistent bars
    prices, _ = clean(raw)
    has_range = prices["high"].notna()
    p = prices[has_range]
    assert (prices["close"] > 0).all()
    assert (p["high"] >= p["low"]).all()
    assert p["open"].between(p["low"], p["high"]).all()
    assert p["close"].between(p["low"], p["high"]).all()
