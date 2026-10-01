"""Database connection helpers.

Connection settings are read from a `.env` file in the project root
(never committed). Copy `.env.example` to `.env` and fill in your password.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import URL, Engine, create_engine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQL_DIR = PROJECT_ROOT / "sql"

load_dotenv(PROJECT_ROOT / ".env")


def database_name() -> str:
    return os.getenv("PGDATABASE", "volatility_db")


def get_engine(database: str | None = None) -> Engine:
    """SQLAlchemy engine for the project database (or another database on the same server)."""
    password = os.getenv("PGPASSWORD")
    if not password:
        raise RuntimeError(
            "PGPASSWORD is not set. Copy .env.example to .env in the project root "
            "and fill in your PostgreSQL password."
        )
    url = URL.create(
        "postgresql+psycopg2",
        username=os.getenv("PGUSER", "postgres"),
        password=password,
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        database=database or database_name(),
    )
    return create_engine(url)
