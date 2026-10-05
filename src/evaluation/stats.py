"""Statistical comparison of forecasting models.

A lower average loss does not prove that a model is better: the difference may
be due to chance. Two standard tools answer "is the difference real?":

* **Diebold-Mariano test** (1995): compares two models. It tests whether the
  average difference between their losses is zero.
* **Model Confidence Set** (Hansen, Lunde and Nason, 2011): compares many
  models at once and returns the set of models that cannot be distinguished
  from the best one at a given confidence level.

Both work on a **time series of losses**. In this project a model's loss on a
given day is its average QLIKE across all series on that day: stocks move
together, so the 59 losses of one day are far from independent, and treating
them as 59 separate observations would make every difference look significant.
Averaging per day leaves one observation per date.

Losses of consecutive days are also correlated (volatility clusters, and
multi-day targets overlap). Both tools account for it: the Diebold-Mariano test
with a Newey-West variance, the Model Confidence Set with a block bootstrap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


def default_lag(n_obs: int, horizon: int) -> int:
    """Lags used for the long-run variance: at least horizon - 1 (overlapping targets)."""
    return max(horizon - 1, int(np.floor(n_obs ** (1 / 3))))


def long_run_variance(series: np.ndarray, lag: int) -> float:
    """Newey-West estimate of the variance of a mean, robust to autocorrelation.

    Returns the long-run variance of the series itself (divide by the number of
    observations to get the variance of its mean). With lag = 0 this is the
    ordinary variance.
    """
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    n = len(x)
    variance = float(x @ x) / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)                       # Bartlett kernel
        variance += 2.0 * weight * float(x[k:] @ x[:-k]) / n
    return variance


@dataclass(frozen=True)
class DmResult:
    mean_difference: float       # average of loss_a - loss_b (negative: model a is better)
    statistic: float
    p_value: float               # two-sided
    n_obs: int


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1, lag: int | None = None) -> DmResult:
    """Diebold-Mariano test of equal predictive accuracy between two models.

    `loss_a` and `loss_b` are the losses of the two models on the same dates. A
    negative statistic means that model a has the lower loss; the p-value is the
    probability of a difference at least this large if the two models were
    equally accurate.
    """
    difference = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(difference)
    lag = default_lag(n, horizon) if lag is None else lag
    variance = long_run_variance(difference, lag) / n
    mean = float(difference.mean())
    if variance <= 0:
        return DmResult(mean, float("nan"), float("nan"), n)
    statistic = mean / np.sqrt(variance)
    return DmResult(mean, float(statistic), float(2 * norm.sf(abs(statistic))), n)


def block_bootstrap_means(losses: np.ndarray, block: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """Means of the columns of `losses` over `n_boot` circular block-bootstrap samples.

    Resampling whole blocks of consecutive days keeps the autocorrelation of
    the losses. Returns an array (n_boot, n_models).
    """
    rng = np.random.default_rng(seed)
    n = len(losses)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    means = np.empty((n_boot, losses.shape[1]))
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        index = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        means[b] = losses[index].mean(axis=0)
    return means


def model_confidence_set(losses: pd.DataFrame, horizon: int = 1, alpha: float = 0.10,
                         n_boot: int = 1000, block: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Model Confidence Set with the T-max statistic.

    `losses` has one row per date and one column per model. Models are removed
    one by one, the worst first, for as long as the hypothesis "all remaining
    models are equally accurate" is rejected. Returns one row per model with:

    * `mcs_p_value`  the higher, the more plausible it is that the model is among the best;
    * `in_set`       True if the model is in the confidence set at level `alpha`;
    * `eliminated`   order of elimination (1 = removed first; missing = never tested out).
    """
    names = list(losses.columns)
    values = losses.to_numpy(dtype=float)
    block = max(2 * horizon, 10) if block is None else block
    boot = block_bootstrap_means(values, block, n_boot, seed)
    mean = values.mean(axis=0)

    remaining = list(range(len(names)))
    p_values = np.ones(len(names))
    order = np.full(len(names), np.nan)
    running_p, step = 0.0, 0
    while len(remaining) > 1:
        # loss of each remaining model relative to the average of the remaining models
        relative = mean[remaining] - mean[remaining].mean()
        boot_relative = boot[:, remaining] - boot[:, remaining].mean(axis=1, keepdims=True)
        spread = np.sqrt(((boot_relative - relative) ** 2).mean(axis=0))
        spread[spread == 0] = np.inf
        t_stat = relative / spread
        t_boot = ((boot_relative - relative) / spread).max(axis=1)
        p = float((t_boot >= t_stat.max()).mean())

        running_p = max(running_p, p)                        # MCS p-values never decrease
        worst = remaining[int(np.argmax(t_stat))]
        step += 1
        p_values[worst], order[worst] = running_p, step
        remaining.remove(worst)
    p_values[remaining[0]] = 1.0

    out = pd.DataFrame({"model": names, "mean_loss": mean, "mcs_p_value": p_values, "eliminated": order})
    out["in_set"] = out["mcs_p_value"] >= alpha
    return out.sort_values("mean_loss", ignore_index=True)
