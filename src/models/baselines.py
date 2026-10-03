"""Baseline forecasters: no estimated parameters, only simple rules.

A sophisticated model is only worth its complexity if it beats these.

naive            the volatility of the last h days continues for the next h days
monthly_average  the average of the last 22 days, whatever the horizon
historical_mean  the series' long-run average (all history up to day t)
ewma             exponentially weighted average of the daily variance (lambda = 0.94)
riskmetrics      the same on squared close-to-close returns: the classic RiskMetrics
                 model, which shows what the range-based variance measure adds
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster

EWMA_LAMBDA = 0.94          # RiskMetrics (1996) value for daily data
MIN_HISTORY = 250           # observations before a long-run average is used

_HORIZON_FEATURE = {1: "log_rv_d", 5: "log_rv_w", 22: "log_rv_m"}


class Naive(Forecaster):
    """Random walk: the last observed h-day variance."""

    name = "naive"

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        return np.exp(test[_HORIZON_FEATURE[horizon]])


class MonthlyAverage(Forecaster):
    """Average daily variance of the last 22 trading days."""

    name = "monthly_average"

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        return np.exp(test["log_rv_m"])


class HistoricalMean(Forecaster):
    """Long-run average variance of the series, using all history up to day t."""

    name = "historical_mean"

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(["ticker", "date"]).copy()
        data["bl_hist_mean"] = (data.groupby("ticker", sort=False)["rv"]
                                .transform(lambda s: s.expanding(min_periods=MIN_HISTORY).mean()))
        return data

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        return test["bl_hist_mean"]


class Ewma(Forecaster):
    """Exponentially weighted moving average of the daily (range-based) variance."""

    name = "ewma"
    source = "rv"
    column = "bl_ewma"

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.sort_values(["ticker", "date"]).copy()
        values = self._input(data)
        data[self.column] = (values.groupby(data["ticker"], sort=False)
                             .transform(lambda s: s.ewm(alpha=1 - EWMA_LAMBDA, adjust=False, min_periods=22).mean()))
        return data

    def _input(self, data: pd.DataFrame) -> pd.Series:
        return data["rv"]

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        return test[self.column]


class RiskMetrics(Ewma):
    """Classic RiskMetrics: EWMA of squared close-to-close returns."""

    name = "riskmetrics"
    column = "bl_riskmetrics"

    def _input(self, data: pd.DataFrame) -> pd.Series:
        return data["ret_d"] ** 2


BASELINES = [Naive, MonthlyAverage, HistoricalMean, Ewma, RiskMetrics]
