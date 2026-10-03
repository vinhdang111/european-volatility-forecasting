"""Unit tests for the GARCH and GJR-GARCH models."""
import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.models.garch import (MAX_PERSISTENCE, MIN_OBS_FIT, SCALE, Garch, GarchParams, GjrGarch,
                              average_variance_forecast, conditional_variance, fit_garch, next_day_variance)
from tests.test_features import make_panel


def simulate(n, omega, alpha, gamma, beta, seed=0):
    """Returns drawn from a GJR-GARCH(1,1) process (gamma = 0 gives a plain GARCH)."""
    rng = np.random.default_rng(seed)
    returns = np.empty(n)
    variance = omega / (1 - alpha - 0.5 * gamma - beta)
    for t in range(n):
        returns[t] = np.sqrt(variance) * rng.standard_normal()
        variance = omega + (alpha + gamma * (returns[t] < 0)) * returns[t] ** 2 + beta * variance
    return returns


@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)


# ---------------------------------------------------------------------------
# Recursion and multi-day forecast
# ---------------------------------------------------------------------------
def test_conditional_variance_matches_the_textbook_recursion():
    returns = np.random.default_rng(0).normal(0, 1.2, 300)
    params = GarchParams(omega=0.05, alpha=0.07, gamma=0.10, beta=0.85, initial_variance=1.5)
    expected = np.empty(300)
    expected[0] = 1.5
    for t in range(1, 300):
        shock = returns[t - 1] ** 2
        expected[t] = 0.05 + 0.07 * shock + 0.10 * shock * (returns[t - 1] < 0) + 0.85 * expected[t - 1]
    assert np.allclose(conditional_variance(returns, params), expected)


def test_garch_with_riskmetrics_parameters_is_the_riskmetrics_baseline():
    returns = np.random.default_rng(1).normal(0, 1.0, 500)
    params = GarchParams(omega=0.0, alpha=0.06, gamma=0.0, beta=0.94, initial_variance=returns[0] ** 2)
    ewma = pd.Series(returns ** 2).ewm(alpha=0.06, adjust=False).mean().to_numpy()
    assert np.allclose(next_day_variance(returns, params), ewma)


def test_one_day_forecast_is_tomorrows_variance():
    params = GarchParams(0.05, 0.08, 0.0, 0.90, 1.0)
    next_variance = np.array([0.5, 2.5, 6.0])
    assert np.allclose(average_variance_forecast(next_variance, params, 1), next_variance)


def test_multi_day_forecast_equals_the_average_of_the_iterated_forecasts():
    params = GarchParams(omega=0.05, alpha=0.05, gamma=0.06, beta=0.90, initial_variance=1.0)
    start = 6.0                                           # tomorrow's variance, well above the long-run level
    path, value = [], start
    for _ in range(22):
        path.append(value)
        value = params.omega + params.persistence * value
    forecast = average_variance_forecast(np.array([start]), params, 22)[0]
    assert np.isclose(forecast, np.mean(path))
    assert params.long_run_variance < forecast < start     # reverts towards the long-run level


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------
def test_fit_recovers_the_parameters_of_a_simulated_garch():
    returns = simulate(8000, omega=0.04, alpha=0.08, gamma=0.0, beta=0.89, seed=3)
    params = fit_garch(returns)
    assert abs(params.alpha - 0.08) < 0.025
    assert abs(params.beta - 0.89) < 0.035
    assert params.gamma == 0.0


def test_fit_recovers_the_leverage_effect_of_a_simulated_gjr_garch():
    returns = simulate(8000, omega=0.03, alpha=0.03, gamma=0.10, beta=0.89, seed=4)
    params = fit_garch(returns, asymmetric=True)
    assert abs(params.gamma - 0.10) < 0.035
    assert abs(params.beta - 0.89) < 0.035
    assert abs(params.persistence - 0.97) < 0.02


def test_fit_finds_no_leverage_effect_when_there_is_none():
    returns = simulate(8000, omega=0.04, alpha=0.08, gamma=0.0, beta=0.89, seed=5)
    assert fit_garch(returns, asymmetric=True).gamma < 0.03


@pytest.mark.parametrize("asymmetric", [False, True])
def test_fitted_parameters_respect_the_constraints(asymmetric):
    rng = np.random.default_rng(6)
    difficult = [
        rng.normal(0, 1, 600),                                    # no volatility clustering at all
        np.cumsum(rng.normal(0, 0.05, 600)) * rng.normal(0, 1, 600),   # variance drifting upwards
        rng.standard_t(2.5, 600),                                 # very fat tails
    ]
    for returns in difficult:
        p = fit_garch(returns, asymmetric)
        assert p.omega > 0 and p.alpha >= 0 and p.gamma >= 0 and p.beta >= 0
        assert p.persistence <= MAX_PERSISTENCE + 1e-6
        assert np.isfinite(p.long_run_variance) and p.long_run_variance > 0


def test_fit_agrees_with_the_arch_package():
    arch = pytest.importorskip("arch")
    returns = simulate(4000, omega=0.03, alpha=0.03, gamma=0.10, beta=0.89, seed=7)
    for asymmetric in (False, True):
        reference = arch.arch_model(returns, mean="Zero", vol="GARCH", p=1, o=int(asymmetric), q=1,
                                    dist="normal", rescale=False).fit(disp="off").params
        ours = fit_garch(returns, asymmetric)
        assert abs(ours.alpha - reference["alpha[1]"]) < 0.02
        assert abs(ours.beta - reference["beta[1]"]) < 0.03
        if asymmetric:
            assert abs(ours.gamma - reference["gamma[1]"]) < 0.03


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_class", [Garch, GjrGarch])
def test_forecasts_are_positive_and_one_per_test_row(model_class, dataset):
    model = model_class()
    data = model.prepare(dataset)
    cutoff = data["date"].sort_values().unique()[700]
    train, test = data[data["date"] <= cutoff], data[data["date"] > cutoff]
    for horizon in (1, 5, 22):
        forecast = model.fit(train, horizon).predict(test, horizon)
        assert forecast.index.equals(test.index)
        assert forecast.notna().all() and (forecast > 0).all()
    # daily variance in decimal units: an annualised volatility between 5% and 100%
    assert 0.05 < np.sqrt(forecast.median() * 252) < 1.0


@pytest.mark.parametrize("model_class", [Garch, GjrGarch])
def test_forecasts_do_not_use_future_returns(model_class, dataset):
    """A forecast made on day T is the same whether or not returns after T exist."""
    dates = dataset["date"].sort_values().unique()
    train_end, last_known = dates[700], dates[800]
    train = dataset[dataset["date"] <= train_end]
    test = dataset[(dataset["date"] > train_end) & (dataset["date"] <= last_known)]

    full, truncated = model_class(), model_class()
    full.prepare(dataset)
    truncated.prepare(dataset[dataset["date"] <= last_known])
    for horizon in (1, 22):
        pd.testing.assert_series_equal(full.fit(train, horizon).predict(test, horizon),
                                       truncated.fit(train, horizon).predict(test, horizon))


def test_parameters_are_estimated_on_training_returns_only(dataset):
    dates = dataset["date"].sort_values().unique()
    train = dataset[dataset["date"] <= dates[700]]
    model = Garch()
    model.prepare(dataset)
    model.fit(train, 1)
    one = dataset[(dataset["ticker"] == "S0.PA") & (dataset["date"] <= dates[700])].dropna(subset=["ret_d"])
    expected = fit_garch(one["ret_d"].to_numpy() * SCALE)
    assert model.params_["S0.PA"] == expected


def test_young_series_borrows_the_dynamics_but_keeps_its_own_level(dataset):
    dates = dataset["date"].sort_values().unique()
    young_start = dates[700 - 100]                                  # only 100 returns before the test period
    data = dataset[(dataset["ticker"] != "S2.PA") | (dataset["date"] >= young_start)]
    train, test = data[data["date"] <= dates[700]], data[data["date"] > dates[700]]
    assert (train["ticker"] == "S2.PA").sum() < MIN_OBS_FIT

    model = GjrGarch()
    model.prepare(data)
    forecast = model.fit(train, 5).predict(test, 5)
    assert "S2.PA" not in model.params_                              # no estimate of its own ...
    assert forecast[test["ticker"] == "S2.PA"].notna().all()         # ... but it still gets forecasts

    young = data[(data["ticker"] == "S2.PA") & (data["date"] <= dates[700])].dropna(subset=["ret_d"])
    borrowed = model._borrowed_params(young["ret_d"].to_numpy() * SCALE)
    assert (borrowed.alpha, borrowed.gamma, borrowed.beta) == model.typical_
    assert np.isclose(borrowed.long_run_variance, np.mean((young["ret_d"] * SCALE) ** 2))
