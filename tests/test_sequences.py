"""Unit tests for the data preparation of the neural networks (no deep-learning library needed)."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.splits import walk_forward_folds
from src.features.build_features import HORIZONS, build_features
from src.models.base import add_is_index
from src.models.sequences import (SEQUENCE_CHANNELS, TABULAR_FEATURES, TARGET_SHIFT, WINDOW, SequenceData,
                                  Standardiser, ols_start)
from tests.test_features import make_panel


@pytest.fixture(scope="module")
def dataset():
    data = add_is_index(build_features(make_panel(n_days=900), min_market_stocks=2))
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def seq(dataset):
    return SequenceData(dataset)


def test_sequence_of_a_row_is_the_last_22_days_of_the_same_ticker(dataset, seq):
    row = int(np.flatnonzero((dataset["ticker"] == "S1.PA").to_numpy())[300])
    window = dataset.iloc[seq.positions[row]]
    assert seq.valid[row].all() and len(window) == WINDOW
    assert (window["ticker"] == "S1.PA").all()
    assert window["date"].is_monotonic_increasing
    assert window["date"].iloc[-1] == dataset.loc[row, "date"]          # ends on the day itself: nothing later
    assert np.array_equal(seq.channels[seq.positions[row]], window[SEQUENCE_CHANNELS].to_numpy(dtype=float), equal_nan=True)


def test_sequence_never_reaches_into_another_ticker(dataset, seq):
    first = int(np.flatnonzero((dataset["ticker"] == "S1.PA").to_numpy())[0])
    for offset in (0, 5, WINDOW - 2):
        row = first + offset
        assert seq.valid[row].sum() == offset + 1                        # only the days that exist
        assert (dataset.iloc[seq.positions[row][seq.valid[row]]]["ticker"] == "S1.PA").all()
    assert seq.valid[first + WINDOW - 1].all()


def test_rows_of_finds_each_ticker_and_date(dataset, seq):
    sample = dataset.sample(50, random_state=0)
    assert np.array_equal(seq.rows_of(sample), sample.index.to_numpy())
    unknown = pd.DataFrame({"ticker": ["NOPE.PA"], "date": [dataset["date"].iloc[0]]})
    assert seq.rows_of(unknown).tolist() == [-1]


def test_fold_anchor_is_the_same_for_the_three_horizons(dataset, seq):
    """The backtest passes a different training end per horizon (embargo); the fold must be recognised."""
    anchors = set()
    for horizon in HORIZONS:
        usable = dataset[dataset[f"target_{horizon}d"].notna()]
        fold = walk_forward_folds(usable["date"], embargo=horizon, first_test_year=2022)[0]
        anchors.add(seq.fold_anchor(usable.loc[fold.train_mask(usable["date"]), "date"].max(), horizon))
        test_start = fold.test_start
    assert len(anchors) == 1
    assert seq.calendar[anchors.pop() + 1] == np.datetime64(test_start)   # the anchor is the day before the test period


def test_training_targets_end_before_the_test_period(dataset, seq):
    anchor = int(np.searchsorted(seq.calendar, np.datetime64("2022-06-30")))
    mask = seq.training_mask(seq.cutoffs(anchor))
    position = np.searchsorted(seq.calendar, seq.dates)                   # calendar position of every row
    for j, horizon in enumerate(HORIZONS):
        assert mask[:, j].any()
        assert (position[mask[:, j]] + horizon <= anchor).all()           # target window ends on the anchor at the latest
        assert (position[mask[:, j]] + horizon).max() == anchor           # ... and nothing usable is wasted
    assert not mask[~seq.complete].any()


def test_validation_period_is_separated_from_the_inner_training_set(seq):
    anchor = int(np.searchsorted(seq.calendar, np.datetime64("2022-06-30")))
    cutoffs = seq.cutoffs(anchor)
    mask = seq.training_mask(cutoffs)
    inner, validation = seq.validation_split(mask, cutoffs, validation_days=120)
    position = np.searchsorted(seq.calendar, seq.dates)
    assert not (inner & validation).any()
    assert ((inner | validation) <= mask).all()
    for j, horizon in enumerate(HORIZONS):
        first_validation_day = position[validation[:, j]].min()
        assert (position[inner[:, j]] + horizon < first_validation_day).all()   # inner targets end before validation starts
    assert seq.validation_split(mask, cutoffs, validation_days=600) is None      # not enough history left


def test_standardiser_uses_training_statistics_and_fills_gaps_with_the_mean():
    train = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
    scaler = Standardiser.fit(train)
    assert np.allclose(scaler.mean, [3.0, 20.0])
    out = scaler.transform(np.array([[3.0, np.nan], [5.0, 30.0]]))
    assert np.allclose(out[0], [0.0, 0.0])                                 # the mean, and a gap, both map to 0
    assert np.allclose(out[1], [(5 - 3) / train[:, 0].std(), 1.0])
    assert out.dtype == np.float32


def test_ols_start_recovers_a_linear_relation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 3))
    y = np.column_stack([1.0 + x @ [0.5, -0.2, 0.1], -2.0 + x @ [0.0, 0.3, 0.3]]) + rng.normal(0, 0.01, (2000, 2))
    mask = np.ones((2000, 2), dtype=bool)
    mask[:500, 1] = False                                                  # the second target is missing for some rows
    y[:500, 1] = np.nan
    weights, bias = ols_start(x, y, mask)
    assert np.allclose(weights.T, [[0.5, -0.2, 0.1], [0.0, 0.3, 0.3]], atol=0.01)
    assert np.allclose(bias, [1.0, -2.0], atol=0.01)


def test_targets_are_log_variances_in_percent_squared(dataset, seq):
    assert np.allclose(seq.targets[:, 1], dataset["target_5d"].to_numpy() + TARGET_SHIFT, equal_nan=True)
    assert seq.tabular.shape[1] == len(TABULAR_FEATURES) and "is_index" in TABULAR_FEATURES
