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
    valid_from  DATE,                          -- ignore earlier data (unreliable history)
    note        TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Added in Step 2 (keeps databases created in Step 1 up to date)
ALTER TABLE universe ADD COLUMN IF NOT EXISTS valid_from DATE;

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

-- =====================================================================
-- Step 2: cleaned prices and cleaning audit trail
-- Rebuilt from prices_raw by `python -m src.data.clean_prices`.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 5. Clean prices: every row satisfies basic price logic.
--    open/high/low are NULL when the source only provides a close
--    (no intraday range available).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prices_clean (
    ticker     TEXT NOT NULL REFERENCES universe (ticker),
    date       DATE NOT NULL,
    open       DOUBLE PRECISION,
    high       DOUBLE PRECISION,
    low        DOUBLE PRECISION,
    close      DOUBLE PRECISION NOT NULL,
    adj_close  DOUBLE PRECISION,
    volume     BIGINT,
    PRIMARY KEY (ticker, date),
    CONSTRAINT positive_close     CHECK (close > 0),
    CONSTRAINT positive_adj_close CHECK (adj_close IS NULL OR adj_close > 0),
    CONSTRAINT positive_low       CHECK (low IS NULL OR low > 0),
    CONSTRAINT range_all_or_none  CHECK ((open IS NULL) = (high IS NULL) AND (high IS NULL) = (low IS NULL)),
    CONSTRAINT high_above_low     CHECK (high >= low),
    CONSTRAINT open_within_range  CHECK (open BETWEEN low AND high),
    CONSTRAINT close_within_range CHECK (high IS NULL OR close BETWEEN low AND high),
    CONSTRAINT non_negative_volume CHECK (volume IS NULL OR volume >= 0)
);

-- ---------------------------------------------------------------------
-- 6. Cleaning log: one row per change made to the raw data
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cleaning_log (
    log_id     SERIAL PRIMARY KEY,
    ticker     TEXT NOT NULL REFERENCES universe (ticker),
    date       DATE NOT NULL,
    rule       TEXT NOT NULL,
    action     TEXT NOT NULL CHECK (action IN ('dropped', 'repaired', 'set_null')),
    detail     TEXT,
    logged_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS cleaning_log_ticker_idx ON cleaning_log (ticker, date);

-- ---------------------------------------------------------------------
-- 7. Views: what the cleaning did, and what is left per ticker
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW cleaning_summary AS
SELECT rule,
       action,
       count(*)                AS n_rows,
       count(DISTINCT ticker)  AS n_tickers,
       min(date)               AS first_date,
       max(date)               AS last_date
FROM cleaning_log
GROUP BY rule, action
ORDER BY n_rows DESC;

CREATE OR REPLACE VIEW data_coverage AS
SELECT u.ticker,
       u.name,
       u.asset_type,
       u.country,
       r.n_raw,
       c.n_clean,
       r.n_raw - c.n_clean                           AS n_dropped,
       c.n_with_range,
       c.first_date,
       c.last_date
FROM universe u
JOIN (SELECT ticker, count(*) AS n_raw FROM prices_raw GROUP BY ticker) r USING (ticker)
JOIN (SELECT ticker,
             count(*)           AS n_clean,
             count(high)        AS n_with_range,
             min(date)          AS first_date,
             max(date)          AS last_date
      FROM prices_clean GROUP BY ticker) c USING (ticker)
ORDER BY n_dropped DESC, u.ticker;
