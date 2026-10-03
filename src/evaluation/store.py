"""Store forecasts and scores in PostgreSQL.

`forecasts` is a wide table: one row per (horizon, ticker, date) and one REAL
column per model. Saving a model adds its column if needed and upserts the rows.
The view `forecasts_long` unpivots it to one row per model.
"""
from __future__ import annotations

import io
import logging
import re

import pandas as pd
from sqlalchemy import Engine

from src.db import SQL_DIR

KEYS = ["horizon", "ticker", "date"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL builders (pure functions, so the generated statements can be tested)
# ---------------------------------------------------------------------------
def check_model_name(name: str) -> str:
    """Model names become column names: lowercase letters, digits and underscores only."""
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,40}", name) or name in KEYS:
        raise ValueError(f"Invalid model name: {name!r}")
    return name


def add_columns_sql(models: list[str]) -> str:
    return "\n".join(f"ALTER TABLE forecasts ADD COLUMN IF NOT EXISTS {check_model_name(m)} REAL;" for m in models)


def stage_table_sql(models: list[str]) -> str:
    columns = ", ".join(f"{check_model_name(m)} REAL" for m in models)
    return f"CREATE TEMP TABLE forecasts_stage (horizon SMALLINT, ticker TEXT, date DATE, {columns}) ON COMMIT DROP;"


def upsert_sql(models: list[str]) -> str:
    columns = ", ".join(models)
    updates = ", ".join(f"{m} = EXCLUDED.{m}" for m in models)
    return (f"INSERT INTO forecasts (horizon, ticker, date, {columns})\n"
            f"SELECT horizon, ticker, date, {columns} FROM forecasts_stage\n"
            f"ON CONFLICT (horizon, ticker, date) DO UPDATE SET {updates};")


def clear_stale_sql(models: list[str]) -> str:
    """Forecasts stored by an earlier run of these models for rows that are no longer produced."""
    set_null = ", ".join(f"{m} = NULL" for m in models)
    any_set = " OR ".join(f"f.{m} IS NOT NULL" for m in models)
    return (f"UPDATE forecasts f SET {set_null}\n"
            f"WHERE ({any_set})\n"
            f"  AND NOT EXISTS (SELECT 1 FROM forecasts_stage s\n"
            f"                  WHERE s.horizon = f.horizon AND s.ticker = f.ticker AND s.date = f.date);")


def long_view_sql(models: list[str]) -> str:
    values = ", ".join(f"('{check_model_name(m)}', f.{m})" for m in models)
    return (f"CREATE OR REPLACE VIEW forecasts_long AS\n"
            f"SELECT v.model, f.horizon, f.ticker, f.date, v.pred_var\n"
            f"FROM forecasts f\n"
            f"CROSS JOIN LATERAL (VALUES {values}) AS v (model, pred_var)\n"
            f"WHERE v.pred_var IS NOT NULL;")


FORECAST_VS_ACTUAL_SQL = """
CREATE OR REPLACE VIEW forecast_vs_actual AS
SELECT f.model,
       f.horizon,
       f.ticker,
       u.name,
       u.asset_type,
       u.country,
       u.sector,
       f.date,
       sqrt(f.pred_var::double precision * 252) * 100 AS pred_vol_pct,
       sqrt(exp(CASE f.horizon WHEN 1 THEN x.target_1d
                               WHEN 5 THEN x.target_5d
                               ELSE x.target_22d END) * 252) * 100 AS actual_vol_pct
FROM forecasts_long f
JOIN features x USING (ticker, date)
JOIN universe u USING (ticker);
"""

REGISTER_MODEL_SQL = """
INSERT INTO models (model, family, description, updated_at)
VALUES (%s, %s, %s, now())
ON CONFLICT (model) DO UPDATE SET family = EXCLUDED.family, description = EXCLUDED.description, updated_at = now();
"""


def to_wide(predictions: pd.DataFrame) -> pd.DataFrame:
    """Long predictions (model, horizon, ticker, date, pred_var) -> one column per model."""
    wide = (predictions.pivot(index=KEYS, columns="model", values="pred_var")
            .reset_index().rename_axis(columns=None))
    wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")
    return wide


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def save_forecasts(engine: Engine, predictions: pd.DataFrame, family: str,
                   descriptions: dict[str, str] | None = None) -> None:
    """Store (or replace) the forecasts of the models present in `predictions`."""
    if (predictions["pred_var"] <= 0).any() or predictions["pred_var"].isna().any():
        raise ValueError("Forecast variances must be positive and not missing")
    models = sorted(predictions["model"].unique())
    wide = to_wide(predictions)
    buffer = io.StringIO()
    wide[KEYS + models].to_csv(buffer, index=False, header=False, na_rep="", float_format="%.9g")
    buffer.seek(0)

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            for model in models:
                cur.execute(REGISTER_MODEL_SQL, (model, family, (descriptions or {}).get(model)))
            cur.execute(add_columns_sql(models))
            cur.execute(stage_table_sql(models))
            cur.copy_expert(f"COPY forecasts_stage ({', '.join(KEYS + models)}) FROM STDIN WITH (FORMAT csv, NULL '')", buffer)
            cur.execute(upsert_sql(models))
            cur.execute(clear_stale_sql(models))
            cur.execute("SELECT model FROM models ORDER BY model")
            all_models = [row[0] for row in cur.fetchall()]
            cur.execute(long_view_sql(all_models))
            cur.execute(FORECAST_VS_ACTUAL_SQL)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    logger.info("Saved %d forecasts (%d rows) for: %s", len(predictions), len(wide), ", ".join(models))


PARAMETER_COLUMNS = ["model", "horizon", "train_end", "series", "parameter", "value"]


def parameters_long(history: list[dict]) -> pd.DataFrame:
    """Estimated parameters of every fit, one row per parameter.

    `history` is a model's `history_`: one dict per fit (or per fit and ticker)
    with the keys model, horizon, train_end, optionally ticker, and one key per
    parameter. Models pooled over all series get the series 'ALL'.
    """
    frame = pd.DataFrame(history)
    frame["series"] = frame["ticker"] if "ticker" in frame else "ALL"
    ids = ["model", "horizon", "train_end", "series"]
    values = [c for c in frame.columns if c not in ids and c != "ticker"]
    long = frame.melt(id_vars=ids, value_vars=values, var_name="parameter", value_name="value")
    long["value"] = long["value"].astype(float)
    return long.dropna(subset=["value"])[PARAMETER_COLUMNS]


def save_parameters(engine: Engine, parameters: pd.DataFrame) -> None:
    """Replace the stored parameters of the models present in `parameters`."""
    models = sorted(parameters["model"].unique())
    out = parameters[PARAMETER_COLUMNS].copy()
    out["train_end"] = pd.to_datetime(out["train_end"]).dt.strftime("%Y-%m-%d")
    buffer = io.StringIO()
    out.to_csv(buffer, index=False, header=False, float_format="%.10g")
    buffer.seek(0)

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute("DELETE FROM model_parameters WHERE model = ANY(%s)", (models,))
            cur.copy_expert(f"COPY model_parameters ({', '.join(PARAMETER_COLUMNS)}) FROM STDIN WITH (FORMAT csv)", buffer)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    logger.info("Saved %d parameter values for: %s", len(out), ", ".join(models))


def refresh_scores(engine: Engine) -> None:
    """Recompute model_scores for all stored models (sql/score_models.sql)."""
    sql = (SQL_DIR / "score_models.sql").read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(sql)
        raw.commit()
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()
    logger.info("model_scores refreshed")


def load_overall_scores(engine: Engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM model_scores_overall", engine)
