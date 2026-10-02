"""Unit tests for the feature pipeline, centred on look-ahead (leakage) checks."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import FEATURE_COLUMNS, HORIZONS, TARGET_COLUMNS, build_features

N_DAYS = 420


def make_panel(n_days=N_DAYS, n_stocks=3, seed=0):
    """Synthetic clean prices: a few stocks, one index and the VIX."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    frames = []
    for i in range(n_stocks + 1):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n_days)))
        open_ = close * np.exp(rng.normal(0, 0.005, n_days))
        frames.append(pd.DataFrame({
            "ticker": f"S{i}.PA" if i < n_stocks else "^IDX",
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, n_days))),
            "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, n_days))),
            "close": close,
            "volume": rng.integers(1_000, 5_000, n_days).astype(float) if i < n_stocks else np.nan,
            "asset_type": "stock" if i < n_stocks else "index",
        }))
    vix = 20 * np.exp(np.cumsum(rng.normal(0, 0.05, n_days)))
    frames.append(pd.DataFrame({"ticker": "^VIX", "date": dates, "open": vix, "high": vix, "low": vix,
                                "close": vix, "volume": np.nan, "asset_type": "external"}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


@pytest.fixture(scope="module")
def features(panel):
    return build_features(panel, min_market_stocks=2)


def test_one_row_per_series_and_day_without_vix(panel, features):
    assert "^VIX" not in set(features["ticker"])
    assert len(features) == (panel["asset_type"] != "external").sum()
    assert not features.duplicated(["ticker", "date"]).any()


def test_features_do_not_change_when_future_data_is_added(panel, features):
    """The core leakage test: features at dates <= T are identical with or without data after T."""
    cutoff = panel["date"].sort_values().unique()[300]
    truncated = build_features(panel[panel["date"] <= cutoff], min_market_stocks=2)
    full = features[features["date"] <= cutoff].reset_index(drop=True)
    truncated = truncated.reset_index(drop=True)
    assert len(full) == len(truncated)
    pd.testing.assert_frame_equal(full[["ticker", "date", "rv", *FEATURE_COLUMNS]],
                                  truncated[["ticker", "date", "rv", *FEATURE_COLUMNS]])


def test_targets_use_only_future_days(features):
    one = features[features["ticker"] == "S0.PA"].reset_index(drop=True)
    t = 200
    for h in HORIZONS:
        expected = np.log(one.loc[t + 1:t + h, "rv"].mean())
        assert np.isclose(one.loc[t, f"target_{h}d"], expected)


def test_targets_do_not_depend_on_today(panel):
    """Changing today's prices must not change today's targets."""
    base = build_features(panel, min_market_stocks=2)
    day = panel["date"].sort_values().unique()[200]
    shocked = panel.copy()
    mask = (shocked["ticker"] == "S0.PA") & (shocked["date"] == day)
    shocked.loc[mask, "high"] *= 1.05
    after = build_features(shocked, min_market_stocks=2)
    pick = lambda f: f[(f["ticker"] == "S0.PA") & (f["date"] == day)].iloc[0]
    assert pick(after)["log_rv_d"] > pick(base)["log_rv_d"]             # the feature moves ...
    assert np.allclose(pick(after)[TARGET_COLUMNS].astype(float),       # ... the targets do not
                       pick(base)[TARGET_COLUMNS].astype(float))


def test_last_rows_have_no_target(features):
    one = features[features["ticker"] == "S0.PA"].reset_index(drop=True)
    for h in HORIZONS:
        assert one[f"target_{h}d"].iloc[-h:].isna().all()
        assert not np.isnan(one[f"target_{h}d"].iloc[-h - 1])


def test_vix_feature_uses_the_previous_session(panel, features):
    vix = panel[panel["ticker"] == "^VIX"].set_index("date")["close"]
    one = features[features["ticker"] == "S0.PA"].set_index("date")
    day, previous = vix.index[150], vix.index[149]
    assert np.isclose(one.loc[day, "log_vix"], np.log(vix.loc[previous]))
    assert not np.isclose(one.loc[day, "log_vix"], np.log(vix.loc[day]))


def test_market_feature_is_the_median_stock_and_excludes_indices(features):
    day = features["date"].unique()[250]
    rows = features[features["date"] == day]
    stocks = rows[rows["ticker"].str.endswith(".PA")]
    assert np.isclose(rows["mkt_log_rv_d"].iloc[0], stocks["log_rv_d"].median())
    assert rows["mkt_log_rv_d"].nunique() == 1                          # same value for every series


def test_market_feature_needs_enough_stocks(panel):
    features = build_features(panel, min_market_stocks=10)               # only 3 stocks available
    assert features["mkt_log_rv_d"].isna().all()


def test_har_features_are_past_averages(features):
    one = features[features["ticker"] == "S1.PA"].reset_index(drop=True)
    t = 100
    assert np.isclose(one.loc[t, "log_rv_w"], np.log(one.loc[t - 4:t, "rv"].mean()))
    assert np.isclose(one.loc[t, "log_rv_m"], np.log(one.loc[t - 21:t, "rv"].mean()))


def test_indices_have_no_volume_feature(features):
    assert features.loc[features["ticker"] == "^IDX", "log_volume_ratio"].isna().all()
    assert features.loc[features["ticker"] == "S0.PA", "log_volume_ratio"].notna().sum() > 300
