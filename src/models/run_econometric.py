"""Walk-forward backtest of the econometric models: GARCH, GJR-GARCH, HAR and HAR-X.

Reads the `features` table, produces out-of-sample forecasts for every test
year from 2006, stores them in `forecasts`, stores the estimated parameters in
`model_parameters` and refreshes `model_scores` (all stored models are
re-scored on their common sample).

Usage (from the project root; takes a few minutes)
--------------------------------------------------
    python -m src.models.run_econometric
"""
from __future__ import annotations

import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.backtest import run_backtest
from src.evaluation.store import load_overall_scores, parameters_long, refresh_scores, save_forecasts, save_parameters
from src.features.build_features import HORIZONS
from src.models.garch import Garch, GjrGarch
from src.models.har import Har, HarX

ECONOMETRIC_MODELS = [Garch, GjrGarch, Har, HarX]

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    engine = get_engine()
    data = pd.read_sql("SELECT * FROM features", engine, parse_dates=["date"])
    logger.info("Loaded %d feature rows", len(data))

    models = [model() for model in ECONOMETRIC_MODELS]
    predictions = run_backtest(data, models, horizons=HORIZONS)
    descriptions = {model.name: (model.__doc__ or "").strip().splitlines()[0] for model in models}
    save_forecasts(engine, predictions, family="econometric", descriptions=descriptions)
    save_parameters(engine, pd.concat([parameters_long(model.history_) for model in models], ignore_index=True))
    refresh_scores(engine)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print("\nOut-of-sample scores, lower is better (view: model_scores_overall):")
    print(load_overall_scores(engine).to_string(index=False))


if __name__ == "__main__":
    main()
