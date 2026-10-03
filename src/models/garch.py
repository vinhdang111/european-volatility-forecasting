"""GARCH(1,1) and GJR-GARCH(1,1): the classic econometric volatility models.

Both describe tomorrow's variance as a weighted sum of three things:

    variance(t+1) = omega + alpha * return(t)^2 + beta * variance(t)        GARCH
                          + gamma * return(t)^2   if return(t) < 0          GJR-GARCH

* omega  pulls the variance back towards its long-run level,
* alpha  is the reaction to yesterday's shock,
* beta   is the memory of yesterday's variance,
* gamma  is the extra reaction to a *negative* shock (the leverage effect).

With omega = 0, alpha = 0.06 and beta = 0.94 this is exactly the RiskMetrics
baseline: GARCH estimates these numbers for every series instead of fixing them,
and adds mean reversion.

The parameters are estimated by maximum likelihood on daily close-to-close
returns, one model per series, and re-estimated for every test year. The
implementation is self-contained (NumPy / SciPy) so that every step is visible;
tests/test_garch.py checks it against simulated data and against the `arch`
package.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from src.models.base import Forecaster

SCALE = 100.0                # returns are handled in percent: better conditioned for the optimiser
MIN_OBS_FIT = 500            # about two years of returns to estimate a series' own parameters
MIN_OBS_FALLBACK = 22        # returns needed before the test period to measure a variance level
MAX_PERSISTENCE = 0.9995     # keeps the long-run variance finite
_STARTS = [(0.05, 0.90), (0.10, 0.85), (0.03, 0.95), (0.15, 0.70)]   # (alpha, beta) starting points

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    gamma: float             # 0 for the symmetric GARCH
    beta: float
    initial_variance: float  # variance used to start the recursion (sample variance of the training returns)

    @property
    def persistence(self) -> float:
        """Share of today's variance shock that survives to tomorrow (gamma applies half of the time)."""
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def long_run_variance(self) -> float:
        return self.omega / (1.0 - self.persistence)


# ---------------------------------------------------------------------------
# Variance recursion, likelihood, estimation
# ---------------------------------------------------------------------------
def conditional_variance(returns: np.ndarray, params: GarchParams) -> np.ndarray:
    """Variance of each day given the returns of the days before it.

    Element t is the variance of return t, computed from returns 0 ... t-1 only.
    """
    r2 = returns ** 2
    shock = params.alpha * r2 + params.gamma * r2 * (returns < 0)
    inputs = np.empty(len(returns))
    inputs[0] = params.initial_variance
    inputs[1:] = params.omega + shock[:-1]
    # variance[t] = inputs[t] + beta * variance[t-1]: a linear recursion, computed without a Python loop
    return lfilter([1.0], [1.0, -params.beta], inputs)


def next_day_variance(returns: np.ndarray, params: GarchParams) -> np.ndarray:
    """Element t is the forecast of the variance of day t+1, made at the close of day t."""
    variance = conditional_variance(returns, params)
    r2 = returns ** 2
    return params.omega + params.alpha * r2 + params.gamma * r2 * (returns < 0) + params.beta * variance


def average_variance_forecast(next_variance: np.ndarray, params: GarchParams, horizon: int) -> np.ndarray:
    """Average daily variance expected over the next `horizon` days.

    Beyond tomorrow, the forecast decays geometrically towards the long-run
    variance at the speed given by the persistence.
    """
    if horizon == 1:
        return next_variance
    p = params.persistence
    weight = (1.0 - p ** horizon) / (horizon * (1.0 - p))       # average of 1, p, p^2, ..., p^(h-1)
    return params.long_run_variance + (next_variance - params.long_run_variance) * weight


def negative_log_likelihood(theta: np.ndarray, returns: np.ndarray, initial_variance: float, asymmetric: bool) -> float:
    """Gaussian negative log-likelihood (up to a constant) of the returns."""
    params = _to_params(theta, initial_variance, asymmetric)
    variance = conditional_variance(returns, params)
    if not np.all(np.isfinite(variance)) or variance.min() <= 0:
        return 1e12
    return 0.5 * float(np.sum(np.log(variance) + returns ** 2 / variance))


def _to_params(theta: np.ndarray, initial_variance: float, asymmetric: bool) -> GarchParams:
    if asymmetric:
        omega, alpha, gamma, beta = theta
    else:
        (omega, alpha, beta), gamma = theta, 0.0
    return GarchParams(float(omega), float(alpha), float(gamma), float(beta), float(initial_variance))


def fit_garch(returns: np.ndarray, asymmetric: bool = False) -> GarchParams:
    """Maximum-likelihood estimate of a GARCH(1,1) (or GJR-GARCH if `asymmetric`).

    `returns` are in the scale used by the caller (percent in this project).
    Constraints: all parameters non-negative and persistence below one.
    """
    returns = np.asarray(returns, dtype=float)
    sample_variance = float(np.mean(returns ** 2))
    args = (returns, sample_variance, asymmetric)

    def start(alpha: float, beta: float) -> np.ndarray:
        gamma = 0.05 if asymmetric else 0.0
        alpha = alpha - 0.5 * gamma
        omega = sample_variance * (1.0 - alpha - 0.5 * gamma - beta)
        return np.array([omega, alpha, gamma, beta] if asymmetric else [omega, alpha, beta])

    starts = sorted((start(a, b) for a, b in _STARTS), key=lambda t: negative_log_likelihood(t, *args))
    bounds = [(1e-8 * sample_variance, 10 * sample_variance), (0.0, 1.0)] + ([(0.0, 1.0)] if asymmetric else []) + [(0.0, 0.9999)]
    weights = np.array([0.0, 1.0, 0.5, 1.0] if asymmetric else [0.0, 1.0, 1.0])
    stationary = {"type": "ineq", "fun": lambda t: MAX_PERSISTENCE - float(weights @ t)}

    best = None
    for theta0 in starts[:2]:
        result = minimize(negative_log_likelihood, theta0, args=args, method="SLSQP",
                          bounds=bounds, constraints=[stationary], options={"maxiter": 200, "ftol": 1e-9})
        if np.all(np.isfinite(result.x)) and (best is None or result.fun < best.fun):
            best = result
    theta = starts[0] if best is None or best.fun > negative_log_likelihood(starts[0], *args) else best.x
    return _to_params(theta, sample_variance, asymmetric)


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class Garch(Forecaster):
    """GARCH(1,1) on daily returns, one model per series, re-estimated every test year."""

    name = "garch"
    asymmetric = False

    def __init__(self) -> None:
        self._returns: dict[str, pd.Series] = {}        # ticker -> percent returns indexed by date
        self.params_: dict[str, GarchParams] = {}       # parameters of the last fit
        self.typical_: tuple[float, float, float] | None = None   # median (alpha, gamma, beta) of the last fit
        self.history_: list[dict] = []                  # parameters of every fit (for the notebook)

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        valid = data.dropna(subset=["ret_d"]).sort_values(["ticker", "date"])
        self._returns = {ticker: pd.Series(group["ret_d"].to_numpy() * SCALE, index=pd.DatetimeIndex(group["date"]))
                         for ticker, group in valid.groupby("ticker", sort=False)}
        return data

    def fit(self, train: pd.DataFrame, horizon: int) -> "Garch":
        """Estimate the parameters of every series on its returns up to the end of the training window."""
        train_end = train["date"].max()
        self.params_ = {}
        for ticker, series in self._returns.items():
            returns = series[:train_end].to_numpy()
            if len(returns) >= MIN_OBS_FIT:
                self.params_[ticker] = fit_garch(returns, self.asymmetric)

        fitted = list(self.params_.values())
        self.typical_ = tuple(float(np.median([getattr(p, name) for p in fitted]))
                              for name in ("alpha", "gamma", "beta")) if fitted else None
        for ticker, p in self.params_.items():
            self.history_.append({"model": self.name, "horizon": horizon, "train_end": train_end, "ticker": ticker,
                                  "omega": p.omega, "alpha": p.alpha, "gamma": p.gamma, "beta": p.beta,
                                  "persistence": p.persistence})
        return self

    def _borrowed_params(self, returns_before_test: np.ndarray) -> GarchParams | None:
        """Parameters for a series too young for its own estimate.

        It borrows the typical dynamics (median alpha, gamma, beta of the other
        series) and keeps its own variance level, measured before the test period.
        """
        if self.typical_ is None or len(returns_before_test) < MIN_OBS_FALLBACK:
            return None
        alpha, gamma, beta = self.typical_
        variance = float(np.mean(returns_before_test ** 2))
        return GarchParams(variance * (1.0 - alpha - 0.5 * gamma - beta), alpha, gamma, beta, variance)

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        out = pd.Series(np.nan, index=test.index)
        test_start = test["date"].min()
        for ticker, rows in test.groupby("ticker", sort=False):
            series = self._returns.get(ticker)
            if series is None:
                continue
            params = self.params_.get(ticker) or self._borrowed_params(series[series.index < test_start].to_numpy())
            if params is None:
                continue
            forecast = average_variance_forecast(next_day_variance(series.to_numpy(), params), params, horizon)
            # a date without a return (hole in the data) keeps the last available forecast
            by_date = pd.Series(forecast / SCALE ** 2, index=series.index).reindex(pd.DatetimeIndex(rows["date"]), method="ffill")
            out.loc[rows.index] = by_date.to_numpy()
        return out


class GjrGarch(Garch):
    """GJR-GARCH(1,1): GARCH with a stronger reaction to negative returns (leverage effect)."""

    name = "gjr_garch"
    asymmetric = True
