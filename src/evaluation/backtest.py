"""Walk-forward backtest: train on the past, forecast the next year, move forward."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluation.splits import FIRST_TEST_YEAR, walk_forward_folds
from src.models.base import Forecaster

PREDICTION_COLUMNS = ["model", "horizon", "ticker", "date", "test_year", "pred_var", "actual_var"]

logger = logging.getLogger(__name__)


def run_backtest(data: pd.DataFrame, models: list[Forecaster], horizons=(1, 5, 22),
                 first_test_year: int = FIRST_TEST_YEAR) -> pd.DataFrame:
    """Out-of-sample forecasts of every model, horizon and test year.

    `data` is the features table (one row per ticker and date) with the columns
    `target_<h>d`. Returns one row per model, horizon, ticker and test date with
    the forecast (`pred_var`) and the realised value (`actual_var`), both as
    average daily variances.

    Rows without a realised target or without a forecast (warm-up periods) are
    dropped, and reported in the log.
    """
    for model in models:
        data = model.prepare(data)
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)

    results = []
    for horizon in horizons:
        target = f"target_{horizon}d"
        usable = data[data[target].notna()]
        folds = walk_forward_folds(usable["date"], embargo=horizon, first_test_year=first_test_year)
        for model in models:
            parts = []
            for fold in folds:
                train = usable[fold.train_mask(usable["date"])]
                test = usable[fold.test_mask(usable["date"])]
                pred = model.fit(train, horizon).predict(test, horizon)
                parts.append(pd.DataFrame({
                    "model": model.name, "horizon": horizon,
                    "ticker": test["ticker"].to_numpy(), "date": test["date"].to_numpy(),
                    "test_year": fold.test_year,
                    "pred_var": np.asarray(pred, dtype=float),
                    "actual_var": np.exp(test[target].to_numpy()),
                }))
            out = pd.concat(parts, ignore_index=True)
            valid = out["pred_var"].notna() & (out["pred_var"] > 0)
            logger.info("%-16s h=%-2d  %7d forecasts (%d rows without a forecast dropped)",
                        model.name, horizon, valid.sum(), (~valid).sum())
            results.append(out[valid])
    return pd.concat(results, ignore_index=True)[PREDICTION_COLUMNS]


def common_sample(predictions: pd.DataFrame) -> pd.DataFrame:
    """Keep only the (horizon, ticker, date) cells for which every model has a forecast.

    Models are only comparable on identical observations: a model that skips
    difficult days would otherwise look better than it is.
    """
    n_models = predictions["model"].nunique()
    counts = predictions.groupby(["horizon", "ticker", "date"])["model"].transform("nunique")
    return predictions[counts == n_models]
