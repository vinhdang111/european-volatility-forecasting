"""Unit tests for the baseline forecasters."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.models.baselines import BASELINES, EWMA_LAMBDA, Ewma, HistoricalMean, Naive, RiskMetrics
from tests.test_features import make_panel


@pytest.fixture(scope="module")
def panel():
    return make_panel(n_days=600)


@pytest.fixture(scope="module")
def dataset(panel):
    return build_features(panel, min_market_stocks=2)


def test_naive_repeats_the_last_observed_volatility(dataset):
    rows = dataset.dropna(subset=["log_rv_m"])
    model = Naive()
    assert np.allclose(model.predict(rows, 1), rows["rv"])
    assert np.allclose(model.predict(rows, 5), np.exp(rows["log_rv_w"]))
    assert np.allclose(model.predict(rows, 22), np.exp(rows["log_rv_m"]))


def test_ewma_follows_the_riskmetrics_recursion(dataset):
    prepared = Ewma().prepare(dataset)
    one = prepared[prepared["ticker"] == "S0.PA"].reset_index(drop=True)
    t = 300
    expected = EWMA_LAMBDA * one.loc[t - 1, "bl_ewma"] + (1 - EWMA_LAMBDA) * one.loc[t, "rv"]
    assert np.isclose(one.loc[t, "bl_ewma"], expected)


def test_riskmetrics_uses_squared_returns(dataset):
    prepared = RiskMetrics().prepare(dataset)
    one = prepared[prepared["ticker"] == "S0.PA"].reset_index(drop=True)
    t = 300
    expected = EWMA_LAMBDA * one.loc[t - 1, "bl_riskmetrics"] + (1 - EWMA_LAMBDA) * one.loc[t, "ret_d"] ** 2
    assert np.isclose(one.loc[t, "bl_riskmetrics"], expected)


def test_historical_mean_needs_a_year_of_history(dataset):
    prepared = HistoricalMean().prepare(dataset)
    one = prepared[prepared["ticker"] == "S0.PA"].reset_index(drop=True)
    assert one["bl_hist_mean"].iloc[:248].isna().all()
    t = 400
    assert np.isclose(one.loc[t, "bl_hist_mean"], one.loc[:t, "rv"].mean())


@pytest.mark.parametrize("model_class", BASELINES)
def test_baselines_are_causal(model_class, panel, dataset):
    """Forecasts made on day T are identical whether or not data after T exists."""
    cutoff = panel["date"].sort_values().unique()[450]
    truncated = build_features(panel[panel["date"] <= cutoff], min_market_stocks=2)
    model = model_class()
    full = model.prepare(dataset)
    part = model.prepare(truncated)
    full = full[full["date"] <= cutoff].reset_index(drop=True)
    part = part.reset_index(drop=True)
    for horizon in (1, 5, 22):
        pd.testing.assert_series_equal(pd.Series(model.predict(full, horizon)).reset_index(drop=True),
                                       pd.Series(model.predict(part, horizon)).reset_index(drop=True),
                                       check_names=False)


@pytest.mark.parametrize("model_class", BASELINES)
def test_baseline_forecasts_are_positive(model_class, dataset):
    model = model_class()
    rows = model.prepare(dataset)
    for horizon in (1, 5, 22):
        pred = pd.Series(model.predict(rows, horizon)).dropna()
        assert len(pred) > 0 and (pred > 0).all()
