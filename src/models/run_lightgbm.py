"""Walk-forward backtest of the machine-learning models: LightGBM and the HAR-X + LightGBM hybrid.

Reads the `features` table, tunes the hyperparameters with Optuna inside the
training window, produces out-of-sample forecasts for every test year from
2006, stores them in `forecasts`, stores hyperparameters and SHAP importances
in `model_parameters` and refreshes `model_scores`.

Usage (from the project root; 20 to 40 minutes depending on the machine)
-----------------------------------------------------------------------
    python -m src.models.run_lightgbm
    python -m src.models.run_lightgbm --trials 5      # faster, less thorough search
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.backtest import run_backtest
from src.evaluation.store import load_overall_scores, parameters_long, refresh_scores, save_forecasts, save_parameters
from src.features.build_features import HORIZONS
from src.models.boosting import Lgbm, LgbmHybrid

ML_MODELS = [Lgbm, LgbmHybrid]

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials per hyperparameter search (default 15)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    engine = get_engine()
    data = pd.read_sql("SELECT f.*, u.asset_type FROM features f JOIN universe u USING (ticker)",
                       engine, parse_dates=["date"])
    logger.info("Loaded %d feature rows", len(data))

    models = [model(n_trials=args.trials) for model in ML_MODELS]
    predictions = run_backtest(data, models, horizons=HORIZONS)
    descriptions = {model.name: (model.__doc__ or "").strip().splitlines()[0] for model in models}
    save_forecasts(engine, predictions, family="machine_learning", descriptions=descriptions)
    save_parameters(engine, pd.concat([parameters_long(model.history_) for model in models], ignore_index=True))
    refresh_scores(engine)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print("\nOut-of-sample scores, lower is better (view: model_scores_overall):")
    print(load_overall_scores(engine).to_string(index=False))


if __name__ == "__main__":
    main()
