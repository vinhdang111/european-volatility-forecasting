"""Model comparison: forecast combinations, statistical tests and controlled experiments.

1. Builds two forecast combinations (average and median of the five models
   using the full feature set) and stores them like any other model.
2. Tests whether the differences between models are statistically significant
   (Diebold-Mariano tests for every pair, Model Confidence Set).
3. Re-runs HAR-X with feature groups added or removed (ablation), with the
   stock / index flag, and without the series of each region (unseen markets).

Results go to the tables `model_tests`, `model_confidence_set`,
`ensemble_variants` and `experiment_scores`. All models of the previous steps must have been run.

Usage (from the project root; a few minutes)
--------------------------------------------
    python -m src.data.setup_db                # creates the three result tables
    python -m src.evaluation.run_comparison
"""
from __future__ import annotations

import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.backtest import run_backtest
from src.evaluation.experiments import (ENSEMBLE_MEMBERS, ablation_variants, build_ensembles, ensemble_variant_checks,
                                        held_out_models, score_by_segment)
from src.evaluation.stats import diebold_mariano, model_confidence_set
from src.evaluation.store import (common_sample_sql, load_daily_losses, member_forecasts_sql, refresh_scores,
                                  replace_table, save_forecasts, stored_models)
from src.features.build_features import HORIZONS
from src.models.har import HarX

KEYS = ["horizon", "ticker", "date"]

logger = logging.getLogger(__name__)


def pairwise_tests(daily_losses: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Diebold-Mariano test of every ordered pair of models, for each horizon."""
    rows = []
    for horizon, day in daily_losses.groupby("horizon"):
        for model in models:
            for reference in models:
                if model == reference:
                    continue
                result = diebold_mariano(day[model].to_numpy(), day[reference].to_numpy(), horizon=int(horizon))
                rows.append({"horizon": int(horizon), "model": model, "reference": reference, "n_dates": result.n_obs,
                             "mean_loss_diff": result.mean_difference, "dm_stat": result.statistic, "p_value": result.p_value})
    return pd.DataFrame(rows)


def confidence_sets(daily_losses: pd.DataFrame, models: list[str], alpha: float = 0.10) -> pd.DataFrame:
    """Model Confidence Set of each horizon."""
    parts = []
    for horizon, day in daily_losses.groupby("horizon"):
        result = model_confidence_set(day[models], horizon=int(horizon), alpha=alpha)
        result.insert(0, "horizon", int(horizon))
        parts.append(result)
    out = pd.concat(parts, ignore_index=True)
    out["eliminated"] = out["eliminated"].astype("Int64")
    return out[["horizon", "model", "mean_loss", "mcs_p_value", "eliminated", "in_set"]]


def on_common_sample(predictions: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """Keep the forecasts made on the observations used to score the stored models."""
    return predictions.merge(common, on=KEYS)


def run_ablation(data: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """Scores of the HAR-X variants: overall, and by asset type (to measure the bias on indices)."""
    predictions = on_common_sample(run_backtest(data, ablation_variants(), horizons=HORIZONS), common)
    asset_type = predictions["ticker"].map(data.drop_duplicates("ticker").set_index("ticker")["asset_type"])
    scores = pd.concat([score_by_segment(predictions), score_by_segment(predictions, asset_type)], ignore_index=True)
    return scores.assign(experiment="ablation")


def run_unseen_regions(data: pd.DataFrame, common: pd.DataFrame) -> pd.DataFrame:
    """For each region: HAR-X estimated without its series against HAR-X estimated with them, on the same rows."""
    universe = data.drop_duplicates("ticker")[["ticker", "country"]]
    held_out = held_out_models(universe)
    predictions = on_common_sample(run_backtest(data, [HarX(), *held_out], horizons=HORIZONS), common)
    reference = predictions[predictions["model"] == "har_x"]

    parts = []
    for model in held_out:
        without = predictions[predictions["model"] == model.name].assign(model="trained without the region")
        with_region = reference.merge(without[KEYS], on=KEYS).assign(model="trained with the region")
        both = pd.concat([without, with_region], ignore_index=True)
        parts.append(score_by_segment(both, pd.Series(model.region, index=both.index)))
    return pd.concat(parts, ignore_index=True).assign(experiment="unseen_region")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    engine = get_engine()

    # 1. Forecast combinations, stored and scored like the other models
    members = ", ".join(ENSEMBLE_MEMBERS)
    forecasts = pd.read_sql(f"SELECT horizon, ticker, date, {members} FROM forecasts", engine, parse_dates=["date"])
    ensembles = build_ensembles(forecasts)
    save_forecasts(engine, ensembles, family="ensemble", descriptions={
        "ensemble_mean": "Average of the variance forecasts of HAR-X, LightGBM, the hybrid, the LSTM and the Transformer.",
        "ensemble_median": "Median of the variance forecasts of HAR-X, LightGBM, the hybrid, the LSTM and the Transformer."})
    refresh_scores(engine)
    models = stored_models(engine)

    # 2. Statistical tests on the daily losses
    daily_losses = load_daily_losses(engine)
    logger.info("Daily losses: %d dates per horizon, %d models", daily_losses.groupby("horizon").size().min(), len(models))
    replace_table(engine, "model_tests", pairwise_tests(daily_losses, models))
    replace_table(engine, "model_confidence_set", confidence_sets(daily_losses, models))

    # Robustness of the combination to the choice of its members
    member_forecasts = pd.read_sql(member_forecasts_sql(ENSEMBLE_MEMBERS, models), engine, parse_dates=["date"])
    replace_table(engine, "ensemble_variants", ensemble_variant_checks(member_forecasts))

    # 3. Controlled experiments around HAR-X
    data = pd.read_sql("SELECT f.*, u.asset_type, u.country FROM features f JOIN universe u USING (ticker)",
                       engine, parse_dates=["date"])
    common = pd.read_sql(common_sample_sql(models), engine, parse_dates=["date"])
    experiments = pd.concat([run_ablation(data, common), run_unseen_regions(data, common)], ignore_index=True)
    replace_table(engine, "experiment_scores",
                  experiments[["experiment", "variant", "horizon", "segment", "n", "qlike", "log_mse", "mae_vol", "bias"]])

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.4f}".format)
    versus_harx = pd.read_sql("SELECT horizon, model, mean_loss_diff, dm_stat, p_value FROM model_tests "
                              "WHERE reference = 'har_x' ORDER BY horizon, mean_loss_diff", engine)
    print("\nDiebold-Mariano tests against HAR-X (negative difference = better than HAR-X):")
    print(versus_harx.to_string(index=False))


if __name__ == "__main__":
    main()
