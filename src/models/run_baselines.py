"""Walk-forward backtest of the baseline models.

Reads the `features` table, produces out-of-sample forecasts for every test
year from 2006, stores them in `forecasts` and refreshes `model_scores`.

Usage (from the project root)
-----------------------------
    python -m src.models.run_baselines
"""
from __future__ import annotations

import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.backtest import run_backtest
from src.evaluation.store import load_overall_scores, refresh_scores, save_forecasts
from src.features.build_features import HORIZONS
from src.models.baselines import BASELINES

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    engine = get_engine()
    data = pd.read_sql("SELECT * FROM features", engine, parse_dates=["date"])
    logger.info("Loaded %d feature rows", len(data))

    predictions = run_backtest(data, [model() for model in BASELINES], horizons=HORIZONS)
    descriptions = {model.name: (model.__doc__ or "").strip().splitlines()[0] for model in BASELINES}
    save_forecasts(engine, predictions, family="baseline", descriptions=descriptions)
    refresh_scores(engine)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print("\nOut-of-sample scores, lower is better (view: model_scores_overall):")
    print(load_overall_scores(engine).to_string(index=False))


if __name__ == "__main__":
    main()
