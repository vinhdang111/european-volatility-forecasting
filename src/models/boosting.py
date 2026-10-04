"""LightGBM: gradient-boosted decision trees on the features of Step 4.

A decision tree splits the data with yes/no questions ("is the VIX above 25?",
"did the stock fall today?") and gives one forecast per leaf. Gradient boosting
adds hundreds of small trees one after the other, each one correcting the
errors left by the previous ones. Unlike the linear HAR-X, the model can learn
**non-linear effects and interactions** by itself (e.g. "a fall matters more
when volatility is already high").

Two models
----------
lgbm         LightGBM alone: forecasts the variance from the features.
lgbm_hybrid  LightGBM on top of HAR-X: the linear model gives a first forecast,
             the trees learn a multiplicative correction to it. Trees cannot
             forecast beyond the range seen in training, which hurts in a crisis;
             here the linear part carries the level and the trees only add what
             a straight line cannot capture.

Training objective
------------------
Both are trained with LightGBM's `gamma` objective, whose loss function,
actual/forecast + log(forecast), is QLIKE up to a constant. The model therefore
minimises the project's primary loss directly and forecasts the *mean* of the
variance: no correction factor is needed (unlike the log regressions of Step 6).

Hyperparameters
---------------
Tuned with Optuna on the last two years of the training window (never on test
data), and re-tuned every few folds. The number of trees is set by early
stopping on the same validation period. The search is restricted to small,
regularised trees: daily volatility is noisy, and large trees fit the noise.

Interpretation
--------------
SHAP values (computed by LightGBM's TreeSHAP) split each forecast into the
contribution of every feature; the average absolute contribution per feature is
stored with the model parameters.
"""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.evaluation.metrics import qlike
from src.models.base import Forecaster, add_is_index
from src.models.har import EXTRA_FEATURES, HAR_FEATURES, HarX

FEATURES = HAR_FEATURES + EXTRA_FEATURES + ["dow", "is_index"]
REQUIRED = HAR_FEATURES              # a row needs these to get a forecast; LightGBM handles other gaps itself
VARIANCE_SCALE = 1e4                 # daily variance in percent squared: targets of order 1 instead of 0.0001

FIXED_PARAMS = {"objective": "gamma", "subsample": 0.8, "subsample_freq": 1,
                "random_state": 0, "n_jobs": -1, "verbose": -1}
DEFAULT_PARAMS = {"learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 500,
                  "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_estimators": 150}
MAX_TREES = 600                      # upper limit; early stopping chooses the actual number
PATIENCE = 30                        # stop when the validation loss has not improved for this many trees
VALIDATION_DAYS = 504                # two years of trading days
RETUNE_EVERY = 7                     # folds between two hyperparameter searches (test years 2006, 2013, 2020)
SHAP_SAMPLE = 2000                   # test rows per fold used to measure feature contributions

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Lgbm(Forecaster):
    """LightGBM on all features, trained on the QLIKE (gamma) objective."""

    name = "lgbm"

    def __init__(self, n_trials: int = 15, retune_every: int = RETUNE_EVERY, validation_days: int = VALIDATION_DAYS,
                 fixed_params: dict | None = None) -> None:
        """`fixed_params` skips the search and uses the given hyperparameters (e.g. to refit a stored model)."""
        self.n_trials = n_trials
        self.fixed_params = fixed_params
        self.retune_every = retune_every
        self.validation_days = validation_days
        self.model_: lgb.LGBMRegressor | None = None
        self.base_ = None                              # first-stage model (hybrid only)
        self.params_: dict[int, dict] = {}             # horizon -> tuned hyperparameters
        self._n_fits: dict[int, int] = {}              # horizon -> number of folds fitted so far
        self.history_: list[dict] = []                 # hyperparameters and SHAP importance of every fit

    # -- first stage: a constant for the plain model, HAR-X for the hybrid ----------------
    def _fit_base(self, rows: pd.DataFrame, horizon: int):
        return None

    def _base_variance(self, base, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        """Variance forecast of the first stage; LightGBM models actual / base."""
        return np.full(len(frame), 1.0 / VARIANCE_SCALE)

    # -- data ------------------------------------------------------------------------------
    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return add_is_index(data)

    @staticmethod
    def _complete(frame: pd.DataFrame) -> pd.Series:
        return frame[REQUIRED].notna().all(axis=1)

    # -- hyperparameter search ----------------------------------------------------------------
    def tune(self, rows: pd.DataFrame, horizon: int) -> dict:
        """Optuna search on a time-ordered split: fit on the past, validate on the last two years.

        The `horizon` days before the validation period are left out, because
        their targets overlap it. Returns the best hyperparameters, including the
        number of trees found by early stopping.
        """
        target = f"target_{horizon}d"
        dates = np.sort(rows["date"].unique())
        if self.n_trials <= 0 or len(dates) < self.validation_days + horizon + 250:
            return dict(DEFAULT_PARAMS)
        validation_start = dates[-self.validation_days]
        inner_end = dates[-self.validation_days - horizon - 1]
        inner, validation = rows[rows["date"] <= inner_end], rows[rows["date"] >= validation_start]

        base = self._fit_base(inner, horizon)
        base_inner = self._base_variance(base, inner, horizon)
        base_validation = self._base_variance(base, validation, horizon)
        actual_validation = np.exp(validation[target].to_numpy())
        x_inner, y_inner = inner[FEATURES], np.exp(inner[target].to_numpy()) / base_inner
        x_validation, y_validation = validation[FEATURES], actual_validation / base_validation

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 4, 32, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 200, 5000, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
            }
            model = lgb.LGBMRegressor(**FIXED_PARAMS, n_estimators=MAX_TREES, **params)
            model.fit(x_inner, y_inner, eval_set=[(x_validation, y_validation)],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
            trial.set_user_attr("n_estimators", int(model.best_iteration_ or MAX_TREES))
            return float(qlike(actual_validation, base_validation * model.predict(x_validation)).mean())

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=self.n_trials)
        best = study.best_trial
        return {**best.params, "n_estimators": best.user_attrs["n_estimators"], "validation_qlike": best.value}

    # -- walk-forward interface -------------------------------------------------------------------
    def fit(self, train: pd.DataFrame, horizon: int) -> "Lgbm":
        target = f"target_{horizon}d"
        rows = add_is_index(train)
        rows = rows[self._complete(rows) & rows[target].notna()]

        fold = self._n_fits.get(horizon, 0)
        self._n_fits[horizon] = fold + 1
        tuned_now = self.fixed_params is None and fold % self.retune_every == 0
        if self.fixed_params is not None:
            self.params_[horizon] = dict(self.fixed_params)
        elif tuned_now:
            self.params_[horizon] = self.tune(rows, horizon)
        params = {k: v for k, v in self.params_[horizon].items() if k != "validation_qlike"}

        self.base_ = self._fit_base(rows, horizon)
        label = np.exp(rows[target].to_numpy()) / self._base_variance(self.base_, rows, horizon)
        self.model_ = lgb.LGBMRegressor(**FIXED_PARAMS, **params).fit(rows[FEATURES], label)

        train_end = train["date"].max()
        self.history_.append({"model": self.name, "horizon": horizon, "train_end": train_end, "n_train": len(rows),
                              "tuned_in_this_fold": float(tuned_now), **self.params_[horizon]})
        logger.info("%-12s h=%-2d train to %s: %d rows, %d trees, %d leaves%s", self.name, horizon,
                    pd.Timestamp(train_end).date(), len(rows), params["n_estimators"], params["num_leaves"],
                    " (hyperparameters re-tuned)" if tuned_now else "")
        return self

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        frame = add_is_index(test)
        complete = self._complete(frame)
        forecast = pd.Series(np.nan, index=test.index)
        if complete.any():
            rows = frame[complete]
            forecast[complete] = self._base_variance(self.base_, rows, horizon) * self.model_.predict(rows[FEATURES])
            if self.history_ and len(rows):
                sample = rows.sample(min(SHAP_SAMPLE, len(rows)), random_state=0)
                importance = self.explain(sample).abs().mean()
                self.history_[-1].update({f"shap_{feature}": float(value) for feature, value in importance.items()})
        return forecast

    def explain(self, frame: pd.DataFrame) -> pd.DataFrame:
        """SHAP value of every feature for every row, on the log scale of the forecast.

        A value of 0.10 means that the feature multiplies the forecast variance
        by exp(0.10), i.e. raises it by about 10%.
        """
        frame = add_is_index(frame)
        contributions = self.model_.predict(frame[FEATURES], pred_contrib=True)
        return pd.DataFrame(contributions[:, :-1], index=frame.index, columns=FEATURES)   # last column: baseline


class LgbmHybrid(Lgbm):
    """HAR-X corrected by LightGBM: the trees learn what the linear model misses."""

    name = "lgbm_hybrid"

    def _fit_base(self, rows: pd.DataFrame, horizon: int):
        return HarX().fit(rows, horizon)

    def _base_variance(self, base, frame: pd.DataFrame, horizon: int) -> np.ndarray:
        return base.predict(frame, horizon).to_numpy()
