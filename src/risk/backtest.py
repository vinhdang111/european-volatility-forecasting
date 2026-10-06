"""Backtesting a Value-at-Risk: was the promise kept?

A 99% VaR promises that the loss exceeds it on 1% of the days. A day on which
it does is a **violation**. Four questions are asked of every model:

* **Is the number of violations right?** Kupiec's proportion-of-failures test
  (1995), also called the test of unconditional coverage.
* **Do the violations come one at a time or in clusters?** Christoffersen's
  independence test (1998). A VaR that is violated several days in a row has
  failed to react, even if the total count is right. The two tests together
  form the test of conditional coverage.
* **What would the regulator say?** The Basel traffic light: the number of
  violations of the 99% VaR over one year puts a model in the green, yellow or
  red zone.
* **How good is the VaR overall?** The quantile loss, the scoring rule that is
  minimised by the true quantile. It penalises violations and, more lightly,
  a VaR that is needlessly large (capital left idle), so it can rank two
  models that both have the right number of violations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

from src.risk.var import tail_probability

GREEN_BELOW, RED_FROM = 0.95, 0.9999      # Basel zones, as cumulative probabilities under a correct model


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float               # probability of a result at least this extreme if the VaR is correct

    __test__ = False             # not a pytest test class


def kupiec(n_obs: int, violations: int, p: float) -> TestResult:
    """Kupiec test: is the share of violations equal to `p`?

    Likelihood-ratio test of a binomial proportion; the statistic follows a
    chi-squared distribution with one degree of freedom when the VaR is correct.
    """
    observed = violations / n_obs
    log_lik_expected = xlogy(n_obs - violations, 1 - p) + xlogy(violations, p)
    log_lik_observed = xlogy(n_obs - violations, 1 - observed) + xlogy(violations, observed)
    statistic = max(float(-2.0 * (log_lik_expected - log_lik_observed)), 0.0)
    return TestResult(statistic, float(chi2.sf(statistic, 1)))


def christoffersen(hits: np.ndarray) -> TestResult:
    """Christoffersen test of independence: is a violation more likely the day after a violation?

    `hits` is the sequence of violations (True / False) of one series in date
    order. Undefined (NaN) when there is no violation at all.
    """
    hits = np.asarray(hits, dtype=bool)
    previous, current = hits[:-1], hits[1:]
    n00, n01 = int(np.sum(~previous & ~current)), int(np.sum(~previous & current))
    n10, n11 = int(np.sum(previous & ~current)), int(np.sum(previous & current))
    if n01 + n11 == 0 or n10 + n11 == 0:
        return TestResult(float("nan"), float("nan"))
    after_calm, after_violation = n01 / (n00 + n01), n11 / (n10 + n11)
    overall = (n01 + n11) / (n00 + n01 + n10 + n11)
    log_lik_independent = xlogy(n00 + n10, 1 - overall) + xlogy(n01 + n11, overall)
    log_lik_dependent = (xlogy(n00, 1 - after_calm) + xlogy(n01, after_calm)
                         + xlogy(n10, 1 - after_violation) + xlogy(n11, after_violation))
    statistic = max(float(-2.0 * (log_lik_independent - log_lik_dependent)), 0.0)
    return TestResult(statistic, float(chi2.sf(statistic, 1)))


def conditional_coverage(coverage: TestResult, independence: TestResult) -> TestResult:
    """Joint test of the right number of violations and of their independence (two degrees of freedom)."""
    statistic = coverage.statistic + independence.statistic
    return TestResult(statistic, float(chi2.sf(statistic, 2)) if np.isfinite(statistic) else float("nan"))


def quantile_loss(returns: np.ndarray, var: np.ndarray, p: float) -> np.ndarray:
    """Quantile ("pinball") loss of a VaR forecast, one value per observation.

    On a normal day it costs p times the distance between the return and the
    VaR; on a violation it costs (1 - p) times the amount by which the loss
    exceeds the VaR. Its average is minimised by the true quantile.
    """
    returns, var = np.asarray(returns, dtype=float), np.asarray(var, dtype=float)
    distance = returns + var                                   # negative on a violation
    return np.where(distance < 0, (p - 1.0) * distance, p * distance)


def traffic_light(n_obs: int, violations: int, p: float = 0.01) -> str:
    """Basel zone for a number of violations over `n_obs` days.

    The zone depends on the probability that a correct model has at most this
    many violations: green below 95%, red from 99.99%, yellow in between. With
    250 days and a 99% VaR: green up to 4 violations, yellow from 5 to 9, red
    from 10.
    """
    cumulative = binom.cdf(violations, n_obs, p)
    return "green" if cumulative < GREEN_BELOW else "yellow" if cumulative < RED_FROM else "red"


def backtest(data: pd.DataFrame, var: pd.Series, confidence: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest one VaR series.

    `data` has the columns ticker, year and next_ret and is sorted by series and
    date; `var` is aligned with it. Returns two tables:

    * one row per series: counts, average VaR, quantile loss and the three tests;
    * one row per series and year: counts, average VaR, quantile loss, Basel zone.
    """
    p = tail_probability(confidence)
    frame = pd.DataFrame({"ticker": data["ticker"].to_numpy(), "year": data["year"].to_numpy(),
                          "var": np.asarray(var, dtype=float), "ret": data["next_ret"].to_numpy(dtype=float)})
    frame["hit"] = frame["ret"] < -frame["var"]
    frame["loss"] = quantile_loss(frame["ret"], frame["var"], p)

    series_rows = []
    for ticker, rows in frame.groupby("ticker", sort=True):
        n_obs, violations = len(rows), int(rows["hit"].sum())
        coverage, independence = kupiec(n_obs, violations, p), christoffersen(rows["hit"].to_numpy())
        series_rows.append({
            "ticker": ticker, "n": n_obs, "violations": violations,
            "mean_var": float(rows["var"].mean()), "quantile_loss": float(rows["loss"].mean()),
            "kupiec_stat": coverage.statistic, "kupiec_p": coverage.p_value,
            "independence_p": independence.p_value,
            "cond_coverage_p": conditional_coverage(coverage, independence).p_value,
        })

    yearly = (frame.groupby(["ticker", "year"], sort=True)
              .agg(n=("hit", "size"), violations=("hit", "sum"), mean_var=("var", "mean"), quantile_loss=("loss", "mean"))
              .reset_index())
    yearly["violations"] = yearly["violations"].astype(int)
    yearly["zone"] = [traffic_light(n, x, p) for n, x in zip(yearly["n"], yearly["violations"])]
    return pd.DataFrame(series_rows), yearly


def daily_quantile_loss(data: pd.DataFrame, var: pd.Series, confidence: int) -> pd.Series:
    """Average quantile loss across series for each return date (one observation per day, for the tests)."""
    loss = quantile_loss(data["next_ret"], var, tail_probability(confidence))
    return pd.Series(loss, index=data.index).groupby(data["return_date"]).mean()
