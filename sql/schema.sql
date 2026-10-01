-- =====================================================================
-- European Equity Volatility Forecasting: database schema
-- Run automatically by `python -m src.data.setup_db` (safe to re-run).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Universe: one row per series (stocks, indices, external series)
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS universe (
    ticker      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    asset_type  TEXT NOT NULL CHECK (asset_type IN ('stock', 'index', 'external')),
    country     TEXT NOT NULL,
    exchange    TEXT,
    sector      TEXT,                          -- NULL for indices and VIX
    currency    TEXT NOT NULL,                 -- 'GBp' = pence (London listings)
    note        TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- 2. Raw prices: exactly as received from Yahoo Finance.
--    Deliberately NO sanity constraints (e.g. high >= low): raw data
--    must be stored even when it is wrong, so that errors can be
--    detected, logged and fixed in the cleaning step.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices_raw (
    ticker         TEXT NOT NULL REFERENCES universe (ticker),
    date           DATE NOT NULL,
    open           DOUBLE PRECISION,
    high           DOUBLE PRECISION,
    low            DOUBLE PRECISION,
    close          DOUBLE PRECISION,
    adj_close      DOUBLE PRECISION,
    volume         BIGINT,
    downloaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, date)
);

-- ---------------------------------------------------------------------
-- 3. Download audit trail: one row per run + one row per ticker per run
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS download_runs (
    run_id       SERIAL PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    start_date   DATE NOT NULL,
    end_date     DATE,                         -- NULL = up to the run date
    n_tickers    INTEGER,
    n_ok         INTEGER,
    n_rows       INTEGER
);

CREATE TABLE IF NOT EXISTS download_log (
    run_id             INTEGER NOT NULL REFERENCES download_runs (run_id) ON DELETE CASCADE,
    ticker             TEXT NOT NULL REFERENCES universe (ticker),
    status             TEXT NOT NULL CHECK (status IN ('OK', 'NO DATA')),
    first_date         DATE,
    last_date          DATE,
    years_of_history   NUMERIC(4, 1),
    n_rows             INTEGER,
    missing_ohlc_rows  INTEGER,
    zero_volume_rows   INTEGER,
    jumps_over_50pct   INTEGER,
    PRIMARY KEY (run_id, ticker)
);

-- ---------------------------------------------------------------------
-- 4. Convenience view: quality report of the most recent download run
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW latest_download_report AS
SELECT u.ticker,
       u.name,
       u.asset_type,
       u.country,
       l.status,
       l.first_date,
       l.last_date,
       l.years_of_history,
       l.n_rows,
       l.missing_ohlc_rows,
       l.zero_volume_rows,
       l.jumps_over_50pct
FROM universe u
LEFT JOIN download_log l
       ON l.ticker = u.ticker
      AND l.run_id = (SELECT max(run_id) FROM download_runs)
ORDER BY l.years_of_history NULLS FIRST, u.ticker;
