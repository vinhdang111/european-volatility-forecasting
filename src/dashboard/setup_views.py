"""Step 11: create the views read by the Power BI dashboard (sql/powerbi_views.sql).

Run from the project root, after the models (Steps 5 to 9) and the VaR
backtest (Step 10):

    python -m src.dashboard.setup_views

Then open Power BI Desktop and follow powerbi/README.md.
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import Engine

from src.db import SQL_DIR, get_engine
from src.evaluation.store import stored_models

VIEWS_SQL = SQL_DIR / "powerbi_views.sql"
REQUIRED_MODELS = ["ewma", "gjr_garch", "har_x", "transformer", "ensemble_mean"]        # lines of pbi_forecast_paths
SCORED_MODELS = ["naive", "historical_mean", "monthly_average", "riskmetrics", "ewma", "garch", "gjr_garch", "har",
                 "har_x", "lgbm", "lgbm_hybrid", "lstm", "transformer", "ensemble_median", "ensemble_mean"]  # pbi_regime_scores
REQUIRED_TABLES = ["model_scores", "model_tests", "model_confidence_set", "var_backtest", "var_backtest_yearly",
                   "var_tests", "var_daily"]
VIEWS = ["pbi_series", "pbi_models", "pbi_horizons", "pbi_confidence", "pbi_calendar", "pbi_model_scores",
         "pbi_model_tests", "pbi_regime_scores", "pbi_forecast_paths", "pbi_var_backtest", "pbi_var_yearly",
         "pbi_var_tests", "pbi_var_daily"]

logger = logging.getLogger(__name__)


def missing_inputs(engine: Engine) -> list[str]:
    """Models and result tables the views need but that are not in the database yet."""
    stored = stored_models(engine)
    missing = [f"model {m}" for m in dict.fromkeys(REQUIRED_MODELS + SCORED_MODELS) if m not in stored]
    for table in REQUIRED_TABLES:
        rows = pd.read_sql(f"SELECT count(*) AS n FROM {table}", engine)["n"].iloc[0]
        if rows == 0:
            missing.append(f"table {table}")
    return missing


def create_views(engine: Engine) -> None:
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(VIEWS_SQL.read_text(encoding="utf-8"))
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    engine = get_engine()
    missing = missing_inputs(engine)
    if missing:
        raise RuntimeError("Run Steps 5 to 10 first (python -m src.risk.run_var_backtest creates var_daily). "
                           "Missing: " + ", ".join(missing))
    create_views(engine)
    print("\nViews for Power BI (rows):")
    for view in VIEWS:
        rows = pd.read_sql(f"SELECT count(*) AS n FROM {view}", engine)["n"].iloc[0]
        print(f"  {view:<22} {rows:>12,}")


if __name__ == "__main__":
    main()
