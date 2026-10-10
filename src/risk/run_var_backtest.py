"""Step 10: backtest the Value-at-Risk implied by every model's 1-day volatility forecast.

Run from the project root:

    python -m src.risk.run_var_backtest

For every stored model, two quantile methods (normal, empirical) and two
confidence levels (95%, 99%), it builds the 1-day VaR, counts the violations
and runs the tests of src/risk/backtest.py. A benchmark without a volatility
model (historical simulation) is added. Results go to the tables
`var_quantiles`, `var_backtest`, `var_backtest_yearly` and `var_tests`, and the
daily VaR of a few models (for the Power BI dashboard) to `var_daily`.
Takes about a minute.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.stats import diebold_mariano
from src.evaluation.store import replace_table
from src.risk.backtest import backtest, daily_quantile_loss
from src.risk.var import (CONFIDENCE_LEVELS, HS_METHOD, HS_MODEL, METHODS, apply_quantiles, empirical_quantiles,
                          historical_simulation, load_var_data, value_at_risk)

REFERENCE = "har_x"
SERIES_COLUMNS = ["model", "method", "confidence", "ticker", "n", "violations", "mean_var", "quantile_loss",
                  "kupiec_stat", "kupiec_p", "independence_p", "cond_coverage_p"]
YEARLY_COLUMNS = ["model", "method", "confidence", "ticker", "year", "n", "violations", "mean_var", "quantile_loss", "zone"]
QUANTILE_COLUMNS = ["model", "confidence", "ticker", "year", "quantile", "n_obs", "pooled"]
DAILY_MODELS = ("har_x", "ensemble_mean")          # daily VaR kept for the dashboard (empirical multiplier)
DAILY_COLUMNS = ["ticker", "date", "confidence", "ret"] + [f"var_{m}" for m in (*DAILY_MODELS, HS_MODEL)]

logger = logging.getLogger(__name__)


def evaluation_sample(data: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Series:
    """Rows on which every VaR exists: after the first test year, and with enough history for the benchmark."""
    return (data["year"] > data["year"].min()) & benchmark.notna().all(axis=1)


def run(data: pd.DataFrame, models: list[str], returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All VaR backtests. `data` comes from load_var_data, `returns` has ticker, date, ret_d."""
    benchmark = pd.DataFrame(index=data.index)
    for confidence in CONFIDENCE_LEVELS:
        simulated = historical_simulation(returns, confidence)
        benchmark[confidence] = data[["ticker", "date"]].merge(simulated, on=["ticker", "date"], how="left")["var"].to_numpy()
    keep = evaluation_sample(data, benchmark)
    sample = data[keep]
    logger.info("Evaluation sample: %d observations, %d series, %d to %d",
                len(sample), sample["ticker"].nunique(), sample["year"].min(), sample["year"].max())

    by_series, by_year, quantiles, losses, daily = [], [], [], {}, []

    def evaluate(model: str, method: str, confidence: int, var: pd.Series) -> None:
        series_table, yearly_table = backtest(sample, var[keep], confidence)
        by_series.append(series_table.assign(model=model, method=method, confidence=confidence))
        by_year.append(yearly_table.assign(model=model, method=method, confidence=confidence))
        losses[(model, method, confidence)] = daily_quantile_loss(sample, var[keep], confidence)

    for confidence in CONFIDENCE_LEVELS:
        day = pd.DataFrame({"ticker": sample["ticker"], "date": sample["return_date"], "confidence": confidence,
                            "ret": sample["next_ret"], f"var_{HS_MODEL}": benchmark.loc[keep, confidence]})
        for model in models:
            for method in METHODS:
                if method == "empirical":
                    table = empirical_quantiles(data, model, confidence)
                    quantiles.append(table.assign(model=model, confidence=confidence))
                    var = apply_quantiles(data, model, table)
                    if model in DAILY_MODELS:
                        day[f"var_{model}"] = var[keep]
                else:
                    var = value_at_risk(data, model, method, confidence)
                evaluate(model, method, confidence, var)
        evaluate(HS_MODEL, HS_METHOD, confidence, benchmark[confidence])
        daily.append(day.reindex(columns=DAILY_COLUMNS))
        logger.info("%d%% VaR: %d models done", confidence, len(models))

    tests = []
    for (model, method, confidence), loss in losses.items():
        reference = losses[(REFERENCE, "empirical" if method == HS_METHOD else method, confidence)]
        result = diebold_mariano(loss.to_numpy(), reference.to_numpy(), horizon=1)
        tests.append({"model": model, "method": method, "confidence": confidence, "reference": REFERENCE,
                      "n_dates": len(loss), "quantile_loss": float(loss.mean()),
                      "loss_vs_reference": float(loss.mean() / reference.mean() - 1),
                      "dm_stat": result.statistic if model != REFERENCE else None,
                      "p_value": result.p_value if model != REFERENCE else None})

    return {"var_quantiles": pd.concat(quantiles, ignore_index=True)[QUANTILE_COLUMNS],
            "var_backtest": pd.concat(by_series, ignore_index=True)[SERIES_COLUMNS],
            "var_backtest_yearly": pd.concat(by_year, ignore_index=True)[YEARLY_COLUMNS],
            "var_tests": pd.DataFrame(tests),
            "var_daily": pd.concat(daily, ignore_index=True)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    engine = get_engine()
    data, models = load_var_data(engine)
    if REFERENCE not in models:
        raise RuntimeError("Run the models of Steps 5 to 9 first.")
    returns = pd.read_sql("SELECT ticker, date, ret_d FROM features ORDER BY ticker, date", engine, parse_dates=["date"])
    logger.info("%d forecasts of %d models", len(data), len(models))

    tables = run(data, models, returns)
    for name, frame in tables.items():
        replace_table(engine, name, frame)

    overall = pd.read_sql("SELECT model, method, confidence, violation_rate, mean_var, quantile_loss, pass_kupiec, "
                          "pass_independence FROM var_backtest_overall", engine)
    for column in ["violation_rate", "mean_var", "pass_kupiec", "pass_independence"]:
        overall[column] = (overall[column] * 100).round(2)
    overall["quantile_loss"] = (overall["quantile_loss"] * 1e4).round(3)
    print("\nViolation rate (%), average VaR (%), quantile loss (basis points), share of series passing each test (%):")
    for (confidence, method), rows in overall.groupby(["confidence", "method"], sort=False):
        print(f"\n--- {confidence}% VaR, {method} ---")
        print(rows.drop(columns=["confidence", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
