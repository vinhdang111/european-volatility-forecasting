"""Common interface for all forecasting models.

A model forecasts, at the close of day t, the **average daily variance over the
next h trading days** for every series. Forecasts are returned as variances
(not logs), so that all models are compared on the same scale.
"""
from __future__ import annotations

import pandas as pd


class Forecaster:
    """Base class. Subclasses set `name` and implement `predict`; `fit` is optional."""

    name: str = "model"

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add the columns the model needs to the full dataset, once, before the backtest.

        Anything added here must be *causal*: the value on day t may only depend
        on data up to day t (checked by the unit tests).
        """
        return data

    def fit(self, train: pd.DataFrame, horizon: int) -> "Forecaster":
        """Estimate parameters on the training rows. Models without parameters skip this."""
        return self

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        """Forecast variance for each test row (same index as `test`)."""
        raise NotImplementedError
