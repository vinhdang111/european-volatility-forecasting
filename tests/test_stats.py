"""Unit tests for the statistical comparison of models."""
import numpy as np
import pandas as pd

from src.evaluation.stats import (block_bootstrap_means, default_lag, diebold_mariano, long_run_variance,
                                  model_confidence_set)


def ar1(n, rho, seed):
    """Autocorrelated noise, like daily losses."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = rho * x[t - 1] + rng.normal()
    return x


def test_long_run_variance_without_lags_is_the_ordinary_variance():
    x = np.random.default_rng(0).normal(size=500)
    assert np.isclose(long_run_variance(x, 0), x.var())


def test_long_run_variance_is_larger_for_positively_autocorrelated_series():
    x = ar1(5000, 0.8, seed=1)
    assert long_run_variance(x, 30) > 4 * x.var()            # theory: (1 + rho) / (1 - rho) = 9 times


def test_lag_covers_the_overlap_of_multi_day_targets():
    assert default_lag(5000, 22) == 21
    assert default_lag(5000, 1) == 17                         # cube root of the sample size


def test_dm_detects_a_better_model():
    rng = np.random.default_rng(2)
    common = ar1(3000, 0.5, seed=3)                           # shocks hitting both models
    loss_a = 1.00 + common + rng.normal(0, 0.5, 3000)
    loss_b = 1.10 + common + rng.normal(0, 0.5, 3000)
    result = diebold_mariano(loss_a, loss_b)
    assert result.mean_difference < 0 and result.statistic < -3 and result.p_value < 0.01
    reverse = diebold_mariano(loss_b, loss_a)
    assert np.isclose(reverse.statistic, -result.statistic) and np.isclose(reverse.p_value, result.p_value)


def test_dm_does_not_reject_too_often_when_models_are_equal():
    """With autocorrelated losses of equal mean, about 5% of the tests should reject at the 5% level."""
    rejections = 0
    for seed in range(200):
        difference = ar1(1500, 0.6, seed=seed)
        rejections += diebold_mariano(difference, np.zeros(1500), horizon=5).p_value < 0.05
    assert rejections / 200 < 0.12


def test_ignoring_autocorrelation_would_reject_far_too_often():
    """Why the Newey-West correction is needed: without it, the same test is badly oversized."""
    rejections = 0
    for seed in range(200):
        difference = ar1(1500, 0.6, seed=seed)
        rejections += diebold_mariano(difference, np.zeros(1500), lag=0).p_value < 0.05
    assert rejections / 200 > 0.25


def test_block_bootstrap_keeps_the_sample_mean_on_average():
    losses = np.column_stack([ar1(2000, 0.5, seed=4) + 3.0, ar1(2000, 0.5, seed=5) - 1.0])
    means = block_bootstrap_means(losses, block=20, n_boot=500, seed=0)
    assert means.shape == (500, 2)
    assert np.allclose(means.mean(axis=0), losses.mean(axis=0), atol=0.1)
    assert means.std(axis=0).min() > 0.02                     # block resampling: the means vary


def test_mcs_removes_clearly_worse_models_and_keeps_the_best():
    rng = np.random.default_rng(6)
    common = ar1(2500, 0.5, seed=7)
    losses = pd.DataFrame({
        "best": 1.00 + common + rng.normal(0, 0.3, 2500),
        "as_good": 1.00 + common + rng.normal(0, 0.3, 2500),
        "slightly_worse": 1.02 + common + rng.normal(0, 0.3, 2500),
        "bad": 1.30 + common + rng.normal(0, 0.3, 2500),
        "terrible": 2.00 + common + rng.normal(0, 0.3, 2500),
    })
    result = model_confidence_set(losses, n_boot=500).set_index("model")
    assert result.loc["best", "in_set"] and result.loc["as_good", "in_set"]
    assert not result.loc["bad", "in_set"] and not result.loc["terrible", "in_set"]
    assert result.loc["terrible", "eliminated"] == 1          # the worst model goes first
    assert result.loc["terrible", "mcs_p_value"] <= result.loc["bad", "mcs_p_value"] <= result.loc["best", "mcs_p_value"]


def test_mcs_keeps_everything_when_models_are_indistinguishable():
    rng = np.random.default_rng(8)
    common = ar1(1500, 0.5, seed=9)
    losses = pd.DataFrame({f"m{i}": 1.0 + common + rng.normal(0, 0.5, 1500) for i in range(6)})
    result = model_confidence_set(losses, n_boot=500)
    assert result["in_set"].sum() >= 5
