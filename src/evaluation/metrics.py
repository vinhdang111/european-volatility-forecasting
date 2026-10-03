"""Loss functions for variance forecasts.

Everything is computed on **daily variance** (decimal units): `actual` is the
average daily variance realised over the forecast horizon, `forecast` is the
model's prediction of it.

Two losses are reported throughout the project:

* QLIKE     actual/forecast - log(actual/forecast) - 1
            The standard loss of the volatility literature (Patton, 2011). It is
            robust to noise in the realised-variance proxy, depends only on the
            ratio actual/forecast (so calm and turbulent periods are on the same
            scale) and penalises under-prediction of risk more than over-prediction.
* log MSE   (log actual - log forecast)^2
            Squared error on the log scale, the scale on which models are trained.

For communication, the mean absolute error is also given in annualised
volatility points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.volatility import to_annual_vol

LOSSES = ("qlike", "log_mse", "mae_vol")


def qlike(actual, forecast):
    ratio = np.asarray(actual, dtype=float) / np.asarray(forecast, dtype=float)
    return ratio - np.log(ratio) - 1


def log_squared_error(actual, forecast):
    return (np.log(np.asarray(actual, dtype=float)) - np.log(np.asarray(forecast, dtype=float))) ** 2


def abs_error_vol(actual, forecast):
    """Absolute error in annualised volatility points (%)."""
    return np.abs(to_annual_vol(np.asarray(actual, dtype=float)) - to_annual_vol(np.asarray(forecast, dtype=float)))


def add_losses(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add one column per loss to a frame with `actual_var` and `pred_var`."""
    out = predictions.copy()
    out["qlike"] = qlike(out["actual_var"], out["pred_var"])
    out["log_mse"] = log_squared_error(out["actual_var"], out["pred_var"])
    out["mae_vol"] = abs_error_vol(out["actual_var"], out["pred_var"])
    return out


def score(predictions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Average losses per group, e.g. by=['model', 'horizon'] (adds losses if missing)."""
    if "qlike" not in predictions:
        predictions = add_losses(predictions)
    grouped = predictions.groupby(by, observed=True)
    out = grouped[list(LOSSES)].mean()
    out.insert(0, "n", grouped.size())
    return out
