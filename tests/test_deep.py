"""Unit tests for the neural networks (skipped if PyTorch is not installed)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.evaluation.metrics import qlike
from src.evaluation.store import PARAMETER_COLUMNS, parameters_long
from src.features.build_features import HORIZONS, build_features
from src.models.deep import Lstm, Transformer, VolNet, qlike_loss, qlike_terms
from src.models.har import HAR_FEATURES
from src.models.sequences import SEQUENCE_CHANNELS, TABULAR_FEATURES, WINDOW
from tests.test_features import make_panel

MODELS = [Lstm, Transformer]
SMALL = {"max_epochs": 2, "hidden": 8, "batch_size": 256, "validation_days": 120, "device": "cpu"}


@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)


def split(data, train_end=700, test_end=None):
    dates = data["date"].sort_values().unique()
    train = data[data["date"] <= dates[train_end]]
    test = data[data["date"] > dates[train_end]]
    if test_end is not None:
        test = test[test["date"] <= dates[test_end]]
    return train, test


@pytest.mark.parametrize("encoder", ["lstm", "transformer"])
def test_network_output_has_one_column_per_horizon(encoder):
    net = VolNet(len(TABULAR_FEATURES), len(SEQUENCE_CHANNELS), len(HORIZONS), encoder, hidden=8)
    out = net(torch.randn(7, len(TABULAR_FEATURES)), torch.randn(7, WINDOW, len(SEQUENCE_CHANNELS)))
    assert out.shape == (7, len(HORIZONS))


@pytest.mark.parametrize("encoder", ["lstm", "transformer"])
def test_network_starts_as_the_linear_model(encoder):
    """Before training, the deep path contributes nothing: the output is the linear regression."""
    torch.manual_seed(0)
    net = VolNet(4, 3, 2, encoder, hidden=8, window=5).eval()
    weights, bias = np.arange(8.0).reshape(4, 2) / 10, np.array([1.0, -1.0])
    net.start_from_linear(weights, bias)
    tabular = torch.randn(6, 4)
    out = net(tabular, torch.randn(6, 5, 3))
    assert np.allclose(out.detach().numpy(), tabular.numpy() @ weights + bias, atol=1e-5)


def test_qlike_loss_matches_the_evaluation_metric():
    rng = np.random.default_rng(0)
    actual, forecast = rng.uniform(0.5, 2.0, (50, 3)), rng.uniform(0.5, 2.0, (50, 3))
    terms = qlike_terms(torch.tensor(np.log(forecast)), torch.tensor(np.log(actual)))
    assert np.allclose(terms.numpy(), qlike(actual, forecast))
    perfect = torch.tensor(np.log(actual))
    assert float(qlike_loss(perfect, perfect, torch.ones(50, 3))) == 0.0


def test_qlike_loss_ignores_masked_targets():
    prediction, target = torch.zeros(4, 3), torch.zeros(4, 3)
    target[0, 0] = 5.0                                           # a large error ...
    mask = torch.ones(4, 3)
    assert float(qlike_loss(prediction, target, mask)) > 1.0
    mask[0, 0] = 0.0                                             # ... that is masked out
    assert float(qlike_loss(prediction, target, mask)) == 0.0


@pytest.mark.parametrize("model_class", MODELS)
def test_forecasts_are_positive_and_one_per_test_row(model_class, dataset):
    model = model_class(**SMALL)
    data = model.prepare(dataset)
    train, test = split(data)
    for horizon in HORIZONS:
        forecast = model.fit(train[train[f"target_{horizon}d"].notna()], horizon).predict(test, horizon)
        assert forecast.index.equals(test.index)
        complete = test[HAR_FEATURES].notna().all(axis=1)
        assert forecast[complete].notna().all() and (forecast[complete] > 0).all()
        assert forecast[~complete].isna().all()
    assert 0.05 < np.sqrt(forecast.median() * 252) < 1.0          # a plausible annualised volatility


def test_one_network_serves_the_three_horizons_of_a_fold(dataset):
    model = Lstm(**SMALL)
    data = model.prepare(dataset)
    dates = data["date"].sort_values().unique()
    for horizon in HORIZONS:                                      # the backtest ends training `horizon` days earlier
        model.fit(data[data["date"] <= dates[700 - horizon]], horizon)
    assert len(model._networks) == 1
    assert sorted(record["horizon"] for record in model.history_) == list(HORIZONS)


def test_network_is_retrained_every_few_folds(dataset):
    model = Lstm(refit_every=2, **SMALL)
    data = model.prepare(dataset)
    dates = data["date"].sort_values().unique()
    for end in (640, 670, 700):
        model.fit(data[data["date"] <= dates[end]], 1)
    networks = list(model._networks.values())
    assert networks[0] is networks[1]                             # second fold reuses the first network
    assert networks[2] is not networks[0]                         # third fold trains a new one
    assert len(model.history_) == 2 * len(HORIZONS)


def test_forecasts_do_not_use_data_after_the_forecast_date(dataset):
    """Same forecasts whether or not later data exists (training and inputs only look back)."""
    dates = dataset["date"].sort_values().unique()
    truncated = dataset[dataset["date"] <= dates[800]]
    forecasts = []
    for data in (dataset, truncated):
        model = Lstm(**SMALL)
        prepared = model.prepare(data)
        train, test = split(prepared, 700, 800)
        forecasts.append(model.fit(train, 5).predict(test, 5).dropna().to_numpy())
    assert len(forecasts[0]) == len(forecasts[1]) > 0
    assert np.allclose(forecasts[0], forecasts[1], rtol=1e-3)


def test_training_learns_a_non_linear_pattern(dataset):
    """A V-shaped effect of the VIX is invisible to a linear model; the trained network must capture it."""
    data = dataset.copy()
    z = (data["log_vix"] - data["log_vix"].mean()) / data["log_vix"].std()
    for horizon in HORIZONS:
        data[f"target_{horizon}d"] = data["log_rv_m"] + 1.5 * z.abs()
    settings = {"hidden": 16, "batch_size": 256, "learning_rate": 0.01, "device": "cpu"}

    trained = Lstm(max_epochs=30, patience=30, validation_days=120, **settings)
    train, _ = split(trained.prepare(data))
    rows = train.dropna(subset=[*HAR_FEATURES, "target_5d"])
    actual = np.exp(rows["target_5d"])
    loss_trained = qlike(actual, trained.fit(train, 5).predict(rows, 5)).mean()

    linear_start = Lstm(max_epochs=0, validation_days=10_000, **settings)      # no training: the least-squares start
    linear_start.prepare(data)
    loss_start = qlike(actual, linear_start.fit(train, 5).predict(rows, 5)).mean()
    assert loss_trained < 0.7 * loss_start


@pytest.mark.parametrize("model_class", MODELS)
def test_training_records_can_be_stored_as_parameters(model_class, dataset):
    model = model_class(**SMALL)
    data = model.prepare(dataset)
    train, _ = split(data)
    model.fit(train, 1)
    long = parameters_long(model.history_)
    assert list(long.columns) == PARAMETER_COLUMNS and long["value"].notna().all()
    assert {"epochs", "n_train", "training_seconds", "validation_qlike"} <= set(long["parameter"])
    assert set(long["horizon"]) == set(HORIZONS)
