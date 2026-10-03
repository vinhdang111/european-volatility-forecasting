"""Unit tests for the HAR and HAR-X regressions."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.store import PARAMETER_COLUMNS, parameters_long
from src.features.build_features import build_features
from src.models.har import EXTRA_FEATURES, HAR_FEATURES, Har, HarX
from tests.test_features import make_panel


@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)


def synthetic(n=5000, noise=0.3, seed=0):
    """Rows whose target is an exact linear function of the HAR features, plus noise."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(-9, 1, (n, 3)), columns=HAR_FEATURES)
    frame["target_5d"] = -1.0 + 0.2 * frame["log_rv_d"] + 0.3 * frame["log_rv_w"] + 0.4 * frame["log_rv_m"] + rng.normal(0, noise, n)
    frame["date"] = pd.bdate_range("2010-01-01", periods=n)
    frame["ticker"] = "S0.PA"
    return frame


def test_har_recovers_the_coefficients():
    model = Har().fit(synthetic(), 5)
    assert np.allclose(model.coef_[HAR_FEATURES], [0.2, 0.3, 0.4], atol=0.02)
    assert abs(model.coef_["intercept"] + 1.0) < 0.3


def test_forecast_is_unbiased_for_the_variance_not_for_its_logarithm():
    """exp(predicted log) under-predicts the variance; the smearing factor removes that bias."""
    data = synthetic(noise=0.6)
    model = Har().fit(data, 5)
    actual = np.exp(data["target_5d"])
    forecast = model.predict(data, 5)
    assert np.isclose((actual / forecast).mean(), 1.0)                  # corrected forecast
    uncorrected = forecast / np.exp(model.correction_)
    assert (actual / uncorrected).mean() > 1.15                         # exp(0.6^2 / 2) = 1.20
    assert np.isclose(np.exp(model.correction_), np.exp(0.6 ** 2 / 2), rtol=0.05)


def test_har_uses_only_rows_with_a_target(dataset):
    train = dataset[dataset["date"] <= dataset["date"].sort_values().unique()[700]]
    with_gaps = train.copy()
    with_gaps.loc[with_gaps.index[::7], "target_5d"] = np.nan
    model = Har().fit(with_gaps, 5)
    expected = Har().fit(with_gaps.dropna(subset=["target_5d"]), 5)
    assert np.allclose(model.coef_, expected.coef_)


@pytest.mark.parametrize("model_class", [Har, HarX])
def test_forecasts_are_positive(model_class, dataset):
    dates = dataset["date"].sort_values().unique()
    train = dataset[dataset["date"] <= dates[700]]
    test = dataset[dataset["date"] > dates[700]].dropna(subset=HAR_FEATURES)
    model = model_class()
    for horizon in (1, 5, 22):
        forecast = model.fit(train, horizon).predict(test, horizon)
        assert forecast.index.equals(test.index)
        assert forecast.notna().all() and (forecast > 0).all()


def test_harx_fills_missing_optional_features_but_requires_the_har_ones(dataset):
    dates = dataset["date"].sort_values().unique()
    train = dataset[dataset["date"] <= dates[700]]
    test = dataset[dataset["date"] > dates[700]].dropna(subset=HAR_FEATURES).copy()
    model = HarX().fit(train, 5)

    test.loc[test.index[0], "log_vix"] = np.nan              # optional feature missing: still a forecast
    test.loc[test.index[1], "log_rv_w"] = np.nan             # core feature missing: no forecast
    forecast = model.predict(test, 5)
    assert np.isfinite(forecast.iloc[0])
    assert np.isnan(forecast.iloc[1])
    assert forecast.iloc[2:].notna().all()


def test_harx_uses_all_the_extra_features(dataset):
    train = dataset[dataset["date"] <= dataset["date"].sort_values().unique()[700]]
    model = HarX().fit(train, 5)
    assert list(model.coef_.index) == ["intercept", *HAR_FEATURES, *EXTRA_FEATURES]
    assert np.isfinite(model.coef_).all()


def test_parameters_are_stored_one_row_per_parameter(dataset):
    train = dataset[dataset["date"] <= dataset["date"].sort_values().unique()[700]]
    model = Har()
    model.fit(train, 1)
    model.fit(train, 5)
    long = parameters_long(model.history_)
    assert list(long.columns) == PARAMETER_COLUMNS
    assert set(long["series"]) == {"ALL"} and set(long["horizon"]) == {1, 5}
    assert {"intercept", *HAR_FEATURES, "r2", "smearing_factor", "n_train"} == set(long["parameter"])
    assert not long.duplicated(["model", "horizon", "train_end", "series", "parameter"]).any()
