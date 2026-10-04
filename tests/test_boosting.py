"""Unit tests for the LightGBM models (skipped if lightgbm or optuna is not installed)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

from src.evaluation.metrics import qlike
from src.evaluation.store import PARAMETER_COLUMNS, parameters_long
from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMNS, build_features
from src.models.boosting import DEFAULT_PARAMS, FEATURES, MAX_TREES, Lgbm, LgbmHybrid, add_is_index
from src.models.har import HAR_FEATURES, HarX
from tests.test_features import make_panel

MODELS = [Lgbm, LgbmHybrid]


@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)


@pytest.fixture(scope="module")
def split(dataset):
    dates = dataset["date"].sort_values().unique()
    train = dataset[dataset["date"] <= dates[700]]
    test = dataset[dataset["date"] > dates[700]].dropna(subset=HAR_FEATURES)
    return train, test


def quick(model_class, **kwargs):
    """A model that skips the hyperparameter search (default hyperparameters)."""
    return model_class(n_trials=0, **kwargs)


def test_features_are_known_at_forecast_time():
    assert set(FEATURES) - {"is_index"} == set(FEATURE_COLUMNS)
    assert not set(FEATURES) & set(TARGET_COLUMNS)


def test_is_index_comes_from_the_asset_type(dataset):
    flagged = add_is_index(dataset)
    assert set(flagged.loc[flagged["is_index"] == 1, "ticker"]) == {"^IDX"}
    with pytest.raises(KeyError):
        add_is_index(dataset.drop(columns="asset_type"))


@pytest.mark.parametrize("model_class", MODELS)
def test_forecasts_are_positive_and_one_per_test_row(model_class, split):
    train, test = split
    model = quick(model_class)
    for horizon in (1, 5, 22):
        forecast = model.fit(train, horizon).predict(test, horizon)
        assert forecast.index.equals(test.index)
        assert forecast.notna().all() and (forecast > 0).all()
    assert 0.05 < np.sqrt(forecast.median() * 252) < 1.0          # a plausible annualised volatility


@pytest.mark.parametrize("model_class", MODELS)
def test_rows_without_the_core_features_get_no_forecast(model_class, split):
    train, test = split
    test = test.copy()
    test.loc[test.index[0], "log_rv_w"] = np.nan                  # core feature: no forecast
    test.loc[test.index[1], "log_vix"] = np.nan                   # other feature: LightGBM handles the gap
    forecast = quick(model_class).fit(train, 5).predict(test, 5)
    assert np.isnan(forecast.iloc[0])
    assert forecast.iloc[1:].notna().all()


def test_hybrid_is_harx_times_the_lightgbm_correction(split):
    train, test = split
    model = quick(LgbmHybrid).fit(train, 5)
    harx = HarX().fit(train.dropna(subset=[*HAR_FEATURES, "target_5d"]), 5).predict(test, 5)
    correction = model.model_.predict(add_is_index(test)[FEATURES])
    assert np.allclose(model.predict(test, 5), harx * correction)
    assert 0.5 < np.median(correction) < 2.0                      # a correction, not a forecast of its own


@pytest.mark.parametrize("model_class", MODELS)
def test_shap_values_add_up_to_the_forecast(model_class, split):
    """On the log scale, the baseline plus the feature contributions equals the model output."""
    train, test = split
    model = quick(model_class).fit(train, 5)
    rows = add_is_index(test)
    contributions = model.model_.predict(rows[FEATURES], pred_contrib=True)
    assert contributions.shape == (len(rows), len(FEATURES) + 1)
    assert np.allclose(contributions.sum(axis=1), np.log(model.model_.predict(rows[FEATURES])), atol=1e-6)
    assert list(model.explain(test).columns) == FEATURES


def test_tuning_returns_hyperparameters_inside_the_search_space(split):
    train, _ = split
    model = Lgbm(n_trials=3, validation_days=120)
    rows = add_is_index(train).dropna(subset=[*HAR_FEATURES, "target_5d"])
    params = model.tune(rows, 5)
    assert 0.02 <= params["learning_rate"] <= 0.1
    assert 4 <= params["num_leaves"] <= 32
    assert 1 <= params["n_estimators"] <= MAX_TREES
    assert params["validation_qlike"] > 0


def test_tuning_falls_back_to_defaults_when_history_is_too_short(split):
    train, _ = split
    rows = add_is_index(train).dropna(subset=[*HAR_FEATURES, "target_5d"])
    assert Lgbm(n_trials=3).tune(rows, 5) == DEFAULT_PARAMS       # two years of validation do not fit in 700 days


def test_hyperparameters_are_retuned_every_few_folds(dataset):
    dates = dataset["date"].sort_values().unique()
    model = Lgbm(n_trials=2, retune_every=2, validation_days=120)
    for end in (600, 650, 700):
        model.fit(dataset[dataset["date"] <= dates[end]], 5)
    assert [h["tuned_in_this_fold"] for h in model.history_] == [1.0, 0.0, 1.0]
    first, second = model.history_[0], model.history_[1]
    assert first["num_leaves"] == second["num_leaves"]            # the fold in between reuses the hyperparameters


@pytest.mark.parametrize("model_class", MODELS)
def test_history_can_be_stored_as_parameters(model_class, split):
    train, test = split
    model = quick(model_class)
    model.fit(train, 5).predict(test, 5)
    record = model.history_[-1]
    assert all(f"shap_{feature}" in record for feature in FEATURES)
    long = parameters_long(model.history_)
    assert list(long.columns) == PARAMETER_COLUMNS and long["value"].notna().all()
    assert {"num_leaves", "n_estimators", "shap_log_rv_m"} <= set(long["parameter"])


def test_lightgbm_learns_a_non_linear_effect_that_harx_cannot(dataset):
    """Synthetic target: the variance triples when the VIX is above its median."""
    data = add_is_index(dataset).dropna(subset=[*HAR_FEATURES, "log_vix"]).copy()
    regime = data["log_vix"] > data["log_vix"].median()
    data["target_5d"] = data["log_rv_m"] + np.log(3.0) * regime
    dates = data["date"].sort_values().unique()
    train, test = data[data["date"] <= dates[600]], data[data["date"] > dates[600]]

    actual = np.exp(test["target_5d"])
    model = LgbmHybrid(fixed_params={**DEFAULT_PARAMS, "min_child_samples": 50})
    loss_lgbm = qlike(actual, model.fit(train, 5).predict(test, 5)).mean()
    loss_harx = qlike(actual, HarX().fit(train, 5).predict(test, 5)).mean()
    assert loss_lgbm < 0.5 * loss_harx
