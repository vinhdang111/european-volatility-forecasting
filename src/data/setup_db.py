"""Create the project database, its tables and load the universe.

Safe to run several times: existing objects are kept, the universe table is
updated from `data/universe.csv` (insert new tickers, update changed ones).

Usage (from the project root)
-----------------------------
    python -m src.data.setup_db
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from sqlalchemy import Engine, text

from src.db import PROJECT_ROOT, SQL_DIR, database_name, get_engine

UNIVERSE_PATH = PROJECT_ROOT / "data" / "universe.csv"

logger = logging.getLogger(__name__)

UPSERT_UNIVERSE = text("""
    INSERT INTO universe (ticker, name, asset_type, country, exchange, sector, currency, valid_from, note, updated_at)
    VALUES (:ticker, :name, :asset_type, :country, :exchange, :sector, :currency, :valid_from, :note, now())
    ON CONFLICT (ticker) DO UPDATE SET
        name       = EXCLUDED.name,
        asset_type = EXCLUDED.asset_type,
        country    = EXCLUDED.country,
        exchange   = EXCLUDED.exchange,
        sector     = EXCLUDED.sector,
        currency   = EXCLUDED.currency,
        valid_from = EXCLUDED.valid_from,
        note       = EXCLUDED.note,
        updated_at = now()
""")


def create_database() -> None:
    """Create the project database on the server if it does not exist yet."""
    name = database_name()
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        raise ValueError(f"Invalid database name: {name!r} (use lowercase letters, digits, _)")

    admin = get_engine(database="postgres")
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}).scalar()
        if exists:
            logger.info("Database '%s' already exists", name)
        else:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
            logger.info("Created database '%s'", name)
    admin.dispose()


def create_schema(engine: Engine) -> None:
    """Run sql/schema.sql (all statements use IF NOT EXISTS / OR REPLACE)."""
    schema_sql = (SQL_DIR / "schema.sql").read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(schema_sql)
        raw.commit()
    finally:
        raw.close()
    logger.info("Schema created / up to date")


def load_universe_csv() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH, dtype=str)
    duplicated = universe["ticker"][universe["ticker"].duplicated()]
    if not duplicated.empty:
        raise ValueError(f"Duplicated tickers in universe.csv: {list(duplicated)}")
    return universe


def sync_universe(engine: Engine) -> pd.DataFrame:
    """Insert or update every ticker of data/universe.csv in the universe table."""
    universe = load_universe_csv()
    records = universe.astype(object).where(universe.notna(), None).to_dict("records")
    with engine.begin() as conn:
        conn.execute(UPSERT_UNIVERSE, records)
    logger.info("Universe table synchronised: %d tickers", len(records))
    return universe


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    create_database()
    engine = get_engine()
    create_schema(engine)
    sync_universe(engine)
    logger.info("Database ready. Next step: python -m src.data.fetch_prices")


if __name__ == "__main__":
    main()
