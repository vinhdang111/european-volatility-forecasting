"""Unit tests for the Value-at-Risk construction and its backtests."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom

from src.risk.backtest import (backtest, christoffersen, conditional_coverage, daily_quantile_loss, kupiec,
                               quantile_loss, traffic_light)
from src.risk.run_var_backtest import run
from src.risk.var import (HS_MODEL, apply_quantiles, empirical_quantiles, historical_simulation, standardised_returns,
                          tail_probability, value_at_risk, var_data_sql)


def simulated(n_days=1500, tickers=("A", "B", "C"), seed=0, degrees=4):
    """Fat-tailed returns with a known, time-varying volatility; the model `good` forecasts the true variance."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    frames = []
    for ticker in tickers:
        vol = 0.01 * np.exp(0.5 * np.sin(np.arange(n_days) / 60 + rng.uniform(0, 6)))
        shock = rng.standard_t(degrees, n_days) / np.sqrt(degrees / (degrees - 2))      # variance one
        frames.append(pd.DataFrame({"ticker": ticker, "date": dates, "return_date": dates + pd.offsets.BDay(1),
                                    "next_ret": vol * shock, "good": vol ** 2, "flat": np.full(n_days, 0.01 ** 2)}))
    data = pd.concat(frames, ignore_index=True)
    data["year"] = data["return_date"].dt.year
    return data


# ---------------------------------------------------------------------------
# Building the VaR
# ---------------------------------------------------------------------------
def test_normal_var_is_the_normal_quantile_times_the_forecast_volatility():
    data = simulated(50)
    assert tail_probability(99) == pytest.approx(0.01)
    assert np.allclose(value_at_risk(data, "good", "normal", 95), 1.6449 * np.sqrt(data["good"]), rtol=1e-4)
    assert np.allclose(value_at_risk(data, "good", "normal", 99), 2.3263 * np.sqrt(data["good"]), rtol=1e-4)


def test_empirical_quantiles_use_earlier_years_only():
    data = simulated()
    before = empirical_quantiles(data, "good", 99, min_obs=200)
    changed = data.copy()
    last = changed["year"] == changed["year"].max()
    changed.loc[last, "next_ret"] *= 10                                   # a crash in the last year
    after = empirical_quantiles(changed, "good", 99, min_obs=200)
    pd.testing.assert_frame_equal(before, after)                           # no quantile has moved

    first_year = data["year"].min()
    assert first_year not in set(before["year"])                           # the first year has no history
    assert value_at_risk(data, "good", "empirical", 99, min_obs=200)[data["year"] == first_year].isna().all()


def test_empirical_quantile_is_the_quantile_of_the_series_own_past():
    data = simulated()
    table = empirical_quantiles(data, "good", 95, min_obs=200).set_index(["ticker", "year"])
    year = data["year"].max()
    past = standardised_returns(data, "good")[(data["ticker"] == "B") & (data["year"] < year)]
    assert table.loc[("B", year), "quantile"] == pytest.approx(past.quantile(0.05))
    assert table.loc[("B", year), "n_obs"] == len(past) and not table.loc[("B", year), "pooled"]


def test_series_with_little_history_use_the_quantile_of_all_series():
    data = simulated()
    table = empirical_quantiles(data, "good", 99, min_obs=10_000)          # no series has that much history
    year = data["year"].max()
    pooled = standardised_returns(data, "good")[data["year"] < year].quantile(0.01)
    rows = table[table["year"] == year]
    assert rows["pooled"].all() and np.allclose(rows["quantile"], pooled)
    assert (rows["n_obs"] == (data["year"] < year).sum()).all()


def test_var_is_positive_and_larger_at_higher_confidence():
    data = simulated()
    low, high = value_at_risk(data, "good", "empirical", 95, 200), value_at_risk(data, "good", "empirical", 99, 200)
    known = low.notna()
    assert (low[known] > 0).all() and (high[known] > low[known]).all()
    table = empirical_quantiles(data, "good", 95, 200)
    assert np.allclose(apply_quantiles(data, "good", table)[known], low[known])


def test_empirical_quantile_corrects_the_fat_tails_that_the_normal_quantile_misses():
    data = simulated(n_days=5000, tickers=("A", "B", "C", "D"), degrees=3)
    known = data["year"] > data["year"].min() + 2
    normal = (data["next_ret"] < -value_at_risk(data, "good", "normal", 99))[known].mean()
    empirical = (data["next_ret"] < -value_at_risk(data, "good", "empirical", 99))[known].mean()
    assert normal > 0.013                                                  # too many violations
    assert abs(empirical - 0.01) < 0.002                                   # the right number


def test_historical_simulation_is_the_quantile_of_the_last_returns_of_the_series():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=400)
    returns = pd.concat([pd.DataFrame({"ticker": t, "date": dates, "ret_d": rng.normal(0, s, 400)})
                         for t, s in [("A", 0.01), ("B", 0.03)]], ignore_index=True)
    out = historical_simulation(returns, 95, window=250, min_obs=250).set_index(["ticker", "date"])["var"]
    assert out.loc["A"].iloc[:249].isna().all() and out.loc["A"].iloc[249:].notna().all()
    window = returns[returns["ticker"] == "B"]["ret_d"].iloc[400 - 250:]
    assert out.loc[("B", dates[-1])] == pytest.approx(-window.quantile(0.05))
    assert out.loc["B"].mean() > 2 * out.loc["A"].mean()                   # each series has its own VaR

    later = returns.copy()
    later.loc[later["date"] > dates[300], "ret_d"] = -0.5                  # the future does not change the past
    again = historical_simulation(later, 95, window=250, min_obs=250).set_index(["ticker", "date"])["var"]
    assert np.allclose(again.loc["A"].iloc[249:301], out.loc["A"].iloc[249:301])


def test_var_data_sql_pairs_each_forecast_with_the_return_of_the_next_day():
    sql = var_data_sql(["har_x", "naive"])
    assert "lead(ret_d) OVER (PARTITION BY ticker ORDER BY date)" in sql
    assert "f.horizon = 1" in sql and "f.har_x IS NOT NULL AND f.naive IS NOT NULL" in sql
    with pytest.raises(ValueError):
        var_data_sql(["har_x; DROP TABLE forecasts"])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_kupiec_statistic_matches_the_binomial_likelihood_ratio():
    n, x, p = 250, 7, 0.01
    expected = -2 * (binom.logpmf(x, n, p) - binom.logpmf(x, n, x / n))
    result = kupiec(n, x, p)
    assert result.statistic == pytest.approx(expected) and 0.01 < result.p_value < 0.05


def test_kupiec_accepts_the_right_number_and_rejects_too_many_or_too_few():
    assert kupiec(1000, 10, 0.01).p_value == pytest.approx(1.0)
    assert kupiec(1000, 25, 0.01).p_value < 0.001                          # VaR too small
    assert kupiec(1000, 1, 0.01).p_value < 0.01                            # VaR needlessly large
    assert np.isfinite(kupiec(250, 0, 0.01).statistic)                     # no violation at all


def test_kupiec_rejects_a_correct_var_about_five_times_in_a_hundred():
    rng = np.random.default_rng(2)
    counts = rng.binomial(2500, 0.05, size=4000)
    rejected = np.mean([kupiec(2500, int(x), 0.05).p_value < 0.05 for x in counts])
    assert 0.035 < rejected < 0.065


def test_christoffersen_detects_clustered_violations():
    rng = np.random.default_rng(3)
    independent = rng.random(5000) < 0.05
    assert christoffersen(independent).p_value > 0.05

    clustered = np.zeros(5000, dtype=bool)
    for t in range(1, 5000):                                               # a violation is likely to be followed by another
        clustered[t] = rng.random() < (0.5 if clustered[t - 1] else 0.027)
    assert abs(clustered.mean() - 0.05) < 0.015                            # about the right number of violations
    assert christoffersen(clustered).p_value < 0.001

    assert np.isnan(christoffersen(np.zeros(300, dtype=bool)).p_value)     # undefined without violations


def test_conditional_coverage_adds_the_two_statistics():
    hits = np.zeros(1000, dtype=bool)
    hits[[100, 101, 102, 500, 501, 800, 801, 802, 803, 900]] = True
    coverage, independence = kupiec(1000, int(hits.sum()), 0.01), christoffersen(hits)
    joint = conditional_coverage(coverage, independence)
    assert coverage.p_value == pytest.approx(1.0) and independence.p_value < 0.001
    assert joint.statistic == pytest.approx(coverage.statistic + independence.statistic) and joint.p_value < 0.001


def test_quantile_loss_formula_and_minimum_at_the_true_quantile():
    # VaR 2%: a 1% loss costs p * 1%; a 5% loss is a violation and costs (1 - p) * 3%
    assert np.allclose(quantile_loss([-0.01, -0.05], [0.02, 0.02], 0.05), [0.05 * 0.01, 0.95 * 0.03])
    returns = np.random.default_rng(4).normal(0, 0.01, 200_000)
    true_var = 0.01 * 2.3263
    losses = {scale: quantile_loss(returns, np.full(len(returns), true_var * scale), 0.01).mean() for scale in (0.7, 1.0, 1.4)}
    assert losses[1.0] < losses[0.7] and losses[1.0] < losses[1.4]
    assert (quantile_loss(returns, np.full(len(returns), true_var), 0.01) >= 0).all()


def test_traffic_light_reproduces_the_basel_zones_for_250_days():
    zones = [traffic_light(250, x) for x in range(12)]
    assert zones == ["green"] * 5 + ["yellow"] * 5 + ["red"] * 2


def test_backtest_tables_agree_with_each_other_and_with_a_direct_count():
    data = simulated()
    var = value_at_risk(data, "good", "normal", 95)
    by_series, by_year = backtest(data, var, 95)

    assert list(by_series["ticker"]) == ["A", "B", "C"] and by_series["n"].sum() == len(data)
    direct = (data["next_ret"] < -var).groupby(data["ticker"]).sum()
    assert (by_series.set_index("ticker")["violations"] == direct).all()
    assert (by_year.groupby("ticker")["violations"].sum() == direct).all()
    assert set(by_year["zone"]) <= {"green", "yellow", "red"}
    assert by_series["kupiec_p"].between(0, 1).all() and (by_series["mean_var"] > 0).all()

    daily = daily_quantile_loss(data, var, 95)
    assert len(daily) == data["return_date"].nunique()
    assert daily.mean() == pytest.approx(quantile_loss(data["next_ret"], var, 0.05).mean())


def test_a_model_that_follows_volatility_beats_one_that_does_not():
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2010-01-01", periods=4000)
    vol = np.where((np.arange(4000) // 100) % 2 == 0, 0.005, 0.03)        # calm and turbulent periods of 100 days
    data = pd.concat([pd.DataFrame({"ticker": t, "date": dates, "return_date": dates + pd.offsets.BDay(1),
                                    "next_ret": vol * rng.normal(size=4000), "good": vol ** 2,
                                    "flat": np.full(4000, np.mean(vol ** 2))}) for t in ("A", "B")], ignore_index=True)
    data["year"] = data["return_date"].dt.year
    good, _ = backtest(data, value_at_risk(data, "good", "normal", 95), 95)
    flat, _ = backtest(data, value_at_risk(data, "flat", "normal", 95), 95)

    assert (good["quantile_loss"] < flat["quantile_loss"]).all()
    assert (good["kupiec_p"] > 0.01).all() and (good["independence_p"] > 0.01).all()
    assert (flat["independence_p"] < 0.001).all()                          # its violations all fall in the turbulent periods


def test_run_produces_every_result_table_on_the_same_sample():
    data = simulated(n_days=1300).rename(columns={"good": "har_x", "flat": "naive"})
    history = pd.bdate_range(end=data["date"].min() - pd.offsets.BDay(1), periods=260)
    rng = np.random.default_rng(5)
    returns = pd.concat(
        [pd.concat([pd.DataFrame({"ticker": t, "date": history, "ret_d": rng.normal(0, 0.01, 260)}),
                    rows[["ticker", "return_date", "next_ret"]].rename(columns={"return_date": "date", "next_ret": "ret_d"})])
         for t, rows in data.groupby("ticker")], ignore_index=True)

    tables = run(data, ["har_x", "naive"], returns)
    series, yearly, tests, quantiles = (tables[k] for k in ["var_backtest", "var_backtest_yearly", "var_tests", "var_quantiles"])

    assert len(series) == (2 * 2 + 1) * 2 * 3                              # (2 models x 2 methods + benchmark) x 2 levels x 3 series
    assert series.groupby(["model", "method", "confidence"])["n"].sum().nunique() == 1          # one common sample
    assert yearly["year"].min() == data["year"].min() + 1                  # the first year only calibrates
    assert set(series["model"]) == {"har_x", "naive", HS_MODEL}
    assert set(quantiles["model"]) == {"har_x", "naive"} and quantiles["pooled"].isin([True, False]).all()
    reference = tests[tests["model"] == "har_x"]
    assert (reference["loss_vs_reference"] == 0).all() and reference["p_value"].isna().all()
    assert (tests.loc[tests["model"] == "naive", "loss_vs_reference"] > 0).all()                 # ignoring volatility costs
