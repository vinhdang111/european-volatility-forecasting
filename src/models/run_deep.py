"""Walk-forward backtest of the deep-learning models: LSTM and Transformer.

Reads the `features` table, trains the networks (on a GPU if PyTorch finds one,
otherwise on the CPU), produces out-of-sample forecasts for every test year
from 2006, stores them in `forecasts`, stores the training records in
`model_parameters` and refreshes `model_scores`.

Needs PyTorch:  python -m pip install -r requirements-dl.txt

Usage (from the project root; a few hours on a CPU, much faster on a GPU)
------------------------------------------------------------------------
    python -m src.models.run_deep                                 # one training per test year (21 per model)
    python -m src.models.run_deep --refit-every 3 --epochs 6      # faster: one training every three years
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.db import get_engine
from src.evaluation.backtest import run_backtest
from src.evaluation.store import load_overall_scores, parameters_long, refresh_scores, save_forecasts, save_parameters
from src.features.build_features import HORIZONS
from src.models.deep import MAX_EPOCHS, REFIT_EVERY, Lstm, Transformer

DEEP_MODELS = [Lstm, Transformer]

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refit-every", type=int, default=REFIT_EVERY, help="test years between two trainings")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="maximum number of epochs per training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    engine = get_engine()
    data = pd.read_sql("SELECT f.*, u.asset_type FROM features f JOIN universe u USING (ticker)",
                       engine, parse_dates=["date"])
    logger.info("Loaded %d feature rows", len(data))

    models = [model(refit_every=args.refit_every, max_epochs=args.epochs) for model in DEEP_MODELS]
    logger.info("Training on: %s", models[0].device)
    predictions = run_backtest(data, models, horizons=HORIZONS)
    descriptions = {model.name: (model.__doc__ or "").strip().splitlines()[0] for model in models}
    save_forecasts(engine, predictions, family="deep_learning", descriptions=descriptions)
    save_parameters(engine, pd.concat([parameters_long(model.history_) for model in models], ignore_index=True))
    refresh_scores(engine)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:,.4f}".format)
    print("\nOut-of-sample scores, lower is better (view: model_scores_overall):")
    print(load_overall_scores(engine).to_string(index=False))


if __name__ == "__main__":
    main()
