"""HAR: the Heterogeneous Autoregressive model of realised volatility (Corsi, 2009).

The forecast is a linear combination of the volatility of the last day, the
last week and the last month:

    log variance(next h days) = a + b_d * log_rv_d + b_w * log_rv_w + b_m * log_rv_m

Three horizons stand for three kinds of market participants (day traders,
weekly rebalancers, long-term investors); together they reproduce the long
memory of volatility with only four parameters. The model is estimated by
ordinary least squares, **pooled**: one set of coefficients for all series.

HAR-X adds the other features built in Step 4 (leverage, market-wide volatility,
VIX, ...) to the same linear regression. It is the linear benchmark for the
machine-learning models: same information, no non-linearity.

From log variance to variance
-----------------------------
The regression predicts the *log* of the variance. Taking the exponential gives
the median of the variance, not its mean, and therefore under-predicts risk on
average. The forecast is multiplied by the average of exp(residual) over the
training sample (Duan's smearing estimator), which is also the scaling that
minimises QLIKE in the training sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster

HAR_FEATURES = ["log_rv_d", "log_rv_w", "log_rv_m"]
EXTRA_FEATURES = [
    "log_rv_q", "log_rv_lt",                                    # slower volatility components
    "ret_d", "ret_w", "ret_m", "neg_ret_d", "down_share_m",     # returns and leverage effect
    "vol_of_vol_m", "log_volume_ratio",                         # instability, trading activity
    "mkt_log_rv_d", "mkt_log_rv_w", "mkt_log_rv_m",             # market-wide volatility
    "log_vix", "vix_chg_5d",                                    # implied volatility (lagged)
]


class Har(Forecaster):
    """Pooled HAR regression on the log variance of the last day, week and month."""

    name = "har"
    features = HAR_FEATURES
    required = HAR_FEATURES          # a row needs these to get a forecast; other gaps are filled

    def __init__(self) -> None:
        self.coef_: pd.Series | None = None      # intercept and slopes of the last fit
        self.correction_: float = 0.0            # log of the smearing factor
        self._fill: pd.Series | None = None      # training means used for missing optional features
        self.history_: list[dict] = []           # coefficients of every fit (for the notebook)

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.features].fillna(self._fill).to_numpy(dtype=float)
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, train: pd.DataFrame, horizon: int) -> "Har":
        target = f"target_{horizon}d"
        rows = train.dropna(subset=[*self.required, target])
        self._fill = rows[self.features].mean()
        design, y = self._design(rows), rows[target].to_numpy(dtype=float)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coef

        self.coef_ = pd.Series(coef, index=["intercept", *self.features])
        self.correction_ = float(np.log(np.mean(np.exp(residuals))))
        self.history_.append({"model": self.name, "horizon": horizon, "train_end": train["date"].max(),
                              "n_train": len(rows), "r2": 1.0 - residuals.var() / y.var(),
                              "smearing_factor": float(np.exp(self.correction_)), **self.coef_.to_dict()})
        return self

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        log_forecast = self._design(test) @ self.coef_.to_numpy() + self.correction_
        forecast = pd.Series(np.exp(log_forecast), index=test.index)
        return forecast.where(test[self.required].notna().all(axis=1))


class HarX(Har):
    """HAR plus the other features of Step 4, still a pooled linear regression."""

    name = "har_x"
    features = HAR_FEATURES + EXTRA_FEATURES
