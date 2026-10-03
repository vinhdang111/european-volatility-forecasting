"""Unit tests for losses, walk-forward splits, the backtest loop and forecast storage helpers."""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import common_sample, run_backtest
from src.evaluation.metrics import abs_error_vol, add_losses, log_squared_error, qlike, score
from src.evaluation.splits import walk_forward_folds
from src.evaluation.store import check_model_name, long_view_sql, to_wide, upsert_sql
from src.features.build_features import build_features
from src.models.base import Forecaster
from tests.test_features import make_panel


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def test_losses_are_zero_for_a_perfect_forecast():
    v = np.array([1e-4, 4e-4, 9e-4])
    assert np.allclose(qlike(v, v), 0)
    assert np.allclose(log_squared_error(v, v), 0)
    assert np.allclose(abs_error_vol(v, v), 0)


def test_qlike_is_positive_and_scale_free():
    assert qlike(2e-4, 1e-4) > 0 and qlike(1e-4, 2e-4) > 0
    assert np.isclose(qlike(2e-4, 1e-4), qlike(2e-2, 1e-2))       # depends on the ratio only


def test_qlike_penalises_underprediction_more():
    actual = 4e-4
    too_low, too_high = actual / 2, actual * 2                     # both wrong by a factor of 2
    assert qlike(actual, too_low) > qlike(actual, too_high)
    assert np.isclose(log_squared_error(actual, too_low), log_squared_error(actual, too_high))


def test_mae_is_in_annualised_volatility_points():
    one_pct_daily, two_pct_daily = 0.01**2, 0.02**2
    assert np.isclose(abs_error_vol(two_pct_daily, one_pct_daily), 0.01 * np.sqrt(252) * 100)


def test_score_averages_by_group():
    preds = pd.DataFrame({"model": ["a", "a", "b", "b"], "actual_var": [1e-4] * 4,
                          "pred_var": [1e-4, 1e-4, 2e-4, 2e-4]})
    out = score(preds, ["model"])
    assert out.loc["a", "qlike"] == 0 and out.loc["b", "qlike"] > 0
    assert list(out["n"]) == [2, 2]


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------
DATES = pd.Series(pd.bdate_range("2003-01-01", "2008-12-31"))


def test_one_fold_per_year_and_training_strictly_before_test():
    folds = walk_forward_folds(DATES, embargo=5, first_test_year=2006)
    assert [f.test_year for f in folds] == [2006, 2007, 2008]
    for fold in folds:
        assert fold.train_end < fold.test_start <= fold.test_end
        assert fold.test_start.year == fold.test_end.year == fold.test_year


@pytest.mark.parametrize("embargo", [1, 5, 22])
def test_embargo_removes_the_last_h_days_before_the_test_year(embargo):
    fold = walk_forward_folds(DATES, embargo=embargo, first_test_year=2006)[0]
    skipped = DATES[(DATES > fold.train_end) & (DATES < fold.test_start)]
    assert len(skipped) == embargo
    # the target of the last training row ends before the test period starts
    last_target_day = DATES[DATES > fold.train_end].iloc[embargo - 1]
    assert last_target_day < fold.test_start


def test_training_window_expands_and_test_years_do_not_overlap():
    folds = walk_forward_folds(DATES, embargo=5, first_test_year=2006)
    assert folds[0].train_end < folds[1].train_end < folds[2].train_end
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end < later.test_start


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)     # 2020-01 to mid-2023


class Oracle(Forecaster):
    """Cheats: returns the realised value. Used to check the plumbing."""
    name = "oracle"

    def predict(self, test, horizon):
        return np.exp(test[f"target_{horizon}d"])


class Spy(Forecaster):
    """Records what the backtest hands to fit() and predict()."""
    name = "spy"

    def __init__(self):
        self.calls = []

    def fit(self, train, horizon):
        self.last_train_date = train["date"].max()
        return self

    def predict(self, test, horizon):
        self.calls.append((horizon, self.last_train_date, test["date"].min(), test["date"].max()))
        return np.exp(test["log_rv_m"])


def test_oracle_has_zero_loss_and_only_test_years_are_forecast(dataset):
    preds = run_backtest(dataset, [Oracle()], horizons=(1, 5), first_test_year=2022)
    assert set(preds["test_year"]) == {2022, 2023}
    assert preds["date"].min() >= pd.Timestamp("2022-01-01")
    assert np.allclose(add_losses(preds)["qlike"], 0)


def test_model_is_never_trained_on_the_period_it_forecasts(dataset):
    spy = Spy()
    run_backtest(dataset, [spy], horizons=(22,), first_test_year=2022)
    calendar = np.sort(dataset["date"].unique())
    for horizon, train_end, test_start, _ in spy.calls:
        assert train_end < test_start
        days_between = ((calendar > np.datetime64(train_end)) & (calendar < np.datetime64(test_start))).sum()
        assert days_between >= horizon


def test_common_sample_keeps_only_cells_forecast_by_every_model():
    preds = pd.DataFrame({
        "model": ["a", "a", "b"], "horizon": 1, "ticker": "X",
        "date": pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-03"]),
        "pred_var": 1e-4, "actual_var": 1e-4,
    })
    out = common_sample(preds)
    assert len(out) == 2 and set(out["date"]) == {pd.Timestamp("2022-01-03")}


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------
def test_model_names_must_be_safe_column_names():
    assert check_model_name("har_rv") == "har_rv"
    for bad in ["HAR", "har-rv", "1model", "date", "x; DROP TABLE forecasts"]:
        with pytest.raises(ValueError):
            check_model_name(bad)


def test_wide_format_and_generated_sql():
    preds = pd.DataFrame({
        "model": ["a", "b", "a"], "horizon": [1, 1, 5], "ticker": "X",
        "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-03"]), "pred_var": [1e-4, 2e-4, 3e-4],
    })
    wide = to_wide(preds)
    assert list(wide.columns) == ["horizon", "ticker", "date", "a", "b"]
    assert len(wide) == 2 and wide["b"].isna().sum() == 1
    assert "a = EXCLUDED.a" in upsert_sql(["a", "b"])
    assert "('a', f.a), ('b', f.b)" in long_view_sql(["a", "b"])
