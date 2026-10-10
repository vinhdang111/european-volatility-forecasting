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
    range_valid_from DATE,                     -- ignore earlier open/high/low (close still used)
    note        TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Added in Step 2 (keeps databases created in Step 1 up to date)
ALTER TABLE universe ADD COLUMN IF NOT EXISTS valid_from DATE;
-- Added in Step 4
ALTER TABLE universe ADD COLUMN IF NOT EXISTS range_valid_from DATE;

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

-- =====================================================================
-- Step 4: modelling dataset
-- Rebuilt from prices_clean by `python -m src.features.build_features`.
-- A row dated t contains what is known at the close of day t (features)
-- and what happens afterwards (targets).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 8. Features and targets: one row per stock / index and trading day
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS features (
    ticker            TEXT NOT NULL REFERENCES universe (ticker),
    date              DATE NOT NULL,

    -- Daily variance (decimal units) and how it was measured
    rv                DOUBLE PRECISION,
    rv_source         TEXT CHECK (rv_source IN ('range', 'close')),

    -- Own volatility: day, week, month, quarter, long-run level (logs of daily variance)
    log_rv_d          DOUBLE PRECISION,
    log_rv_w          DOUBLE PRECISION,
    log_rv_m          DOUBLE PRECISION,
    log_rv_q          DOUBLE PRECISION,
    log_rv_lt         DOUBLE PRECISION,

    -- Returns and leverage effect
    ret_d             DOUBLE PRECISION,
    ret_w             DOUBLE PRECISION,
    ret_m             DOUBLE PRECISION,
    neg_ret_d         DOUBLE PRECISION,
    down_share_m      DOUBLE PRECISION,

    -- Instability of volatility, trading activity
    vol_of_vol_m      DOUBLE PRECISION,
    log_volume_ratio  DOUBLE PRECISION,

    -- Market-wide volatility (median European stock)
    mkt_log_rv_d      DOUBLE PRECISION,
    mkt_log_rv_w      DOUBLE PRECISION,
    mkt_log_rv_m      DOUBLE PRECISION,

    -- VIX of the last US session strictly before `date`
    log_vix           DOUBLE PRECISION,
    vix_chg_5d        DOUBLE PRECISION,

    -- Calendar: 0 = Monday ... 4 = Friday
    dow               SMALLINT CHECK (dow BETWEEN 0 AND 6),

    -- Targets: log of the average daily variance over the next 1 / 5 / 22 trading days
    target_1d         DOUBLE PRECISION,
    target_5d         DOUBLE PRECISION,
    target_22d        DOUBLE PRECISION,

    PRIMARY KEY (ticker, date),
    CONSTRAINT positive_rv CHECK (rv IS NULL OR rv > 0),
    CONSTRAINT down_share_is_a_share CHECK (down_share_m IS NULL OR down_share_m BETWEEN 0 AND 1)
);

-- ---------------------------------------------------------------------
-- 9. View: features joined with the static attributes of each series,
--    plus volatility expressed as an annualised percentage for reporting
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW model_dataset AS
SELECT f.*,
       u.name,
       u.asset_type,
       u.country,
       u.sector,
       sqrt(f.rv * 252) * 100              AS vol_d_pct,
       sqrt(exp(f.log_rv_m) * 252) * 100   AS vol_m_pct,
       sqrt(exp(f.target_1d) * 252) * 100  AS target_1d_vol_pct,
       sqrt(exp(f.target_5d) * 252) * 100  AS target_5d_vol_pct,
       sqrt(exp(f.target_22d) * 252) * 100 AS target_22d_vol_pct
FROM features f
JOIN universe u USING (ticker);

-- =====================================================================
-- Step 5: forecasts and model scores
-- Written by the model runners (e.g. `python -m src.models.run_baselines`).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 10. Out-of-sample forecasts, one row per horizon, series and day.
--     Each model adds ONE COLUMN (created by src/evaluation/store.py when
--     the model is first saved): the forecast, made at the close of `date`,
--     of the average daily variance over the following `horizon` trading days.
--     A wide table is about seven times smaller than one row per model.
--     The view `forecasts_long` (also created by store.py) gives the same
--     data with one row per model, for scoring and dashboards.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    horizon   SMALLINT NOT NULL CHECK (horizon IN (1, 5, 22)),
    ticker    TEXT NOT NULL REFERENCES universe (ticker),
    date      DATE NOT NULL,
    PRIMARY KEY (horizon, ticker, date)
);

-- Registry of the models that have a column in `forecasts`
CREATE TABLE IF NOT EXISTS models (
    model        TEXT PRIMARY KEY,
    family       TEXT NOT NULL,
    description  TEXT,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- 11. Scores: average losses per model, horizon, series and test year,
--     computed by sql/score_models.sql on the observations that every
--     model has forecast (common sample).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_scores (
    model      TEXT NOT NULL REFERENCES models (model),
    horizon    SMALLINT NOT NULL,
    ticker     TEXT NOT NULL REFERENCES universe (ticker),
    test_year  SMALLINT NOT NULL,
    n          INTEGER NOT NULL,
    qlike      DOUBLE PRECISION NOT NULL,
    log_mse    DOUBLE PRECISION NOT NULL,
    mae_vol    DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model, horizon, ticker, test_year)
);

-- Overall ranking of the models at each horizon
CREATE OR REPLACE VIEW model_scores_overall AS
SELECT s.horizon,
       s.model,
       m.family,
       sum(s.n)                                   AS n_forecasts,
       sum(s.qlike * s.n) / sum(s.n)              AS qlike,
       sum(s.log_mse * s.n) / sum(s.n)            AS log_mse,
       sum(s.mae_vol * s.n) / sum(s.n)            AS mae_vol,
       rank() OVER (PARTITION BY s.horizon ORDER BY sum(s.qlike * s.n) / sum(s.n)) AS rank_qlike
FROM model_scores s
JOIN models m USING (model)
GROUP BY s.horizon, s.model, m.family
ORDER BY s.horizon, rank_qlike;

-- =====================================================================
-- Step 6: estimated parameters
-- ---------------------------------------------------------------------
-- 12. Parameters of every model that estimates some, for each walk-forward
--     fit: `train_end` is the last day of the training window, `series` is
--     a ticker, or 'ALL' for a model pooled over all series.
--     Examples: GARCH alpha / beta / persistence per ticker, HAR coefficients.
-- =====================================================================
CREATE TABLE IF NOT EXISTS model_parameters (
    model      TEXT NOT NULL REFERENCES models (model),
    horizon    SMALLINT NOT NULL,
    train_end  DATE NOT NULL,
    series     TEXT NOT NULL,
    parameter  TEXT NOT NULL,
    value      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model, horizon, train_end, series, parameter)
);

-- =====================================================================
-- Step 9: statistical comparison of the models and controlled experiments
-- Written by `python -m src.evaluation.run_comparison`.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 13. Diebold-Mariano tests between every pair of models, on the daily
--     average QLIKE (one observation per date, Newey-West variance).
--     mean_loss_diff = loss(model) - loss(reference): negative means that
--     `model` is the more accurate of the two.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_tests (
    horizon         SMALLINT NOT NULL,
    model           TEXT NOT NULL REFERENCES models (model),
    reference       TEXT NOT NULL REFERENCES models (model),
    n_dates         INTEGER NOT NULL,
    mean_loss_diff  DOUBLE PRECISION NOT NULL,
    dm_stat         DOUBLE PRECISION,
    p_value         DOUBLE PRECISION,
    PRIMARY KEY (horizon, model, reference)
);

-- ---------------------------------------------------------------------
-- 14. Model Confidence Set: the models that cannot be distinguished from
--     the best one. `eliminated` is the order in which a model was removed
--     (NULL for the last model standing).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_confidence_set (
    horizon      SMALLINT NOT NULL,
    model        TEXT NOT NULL REFERENCES models (model),
    mean_loss    DOUBLE PRECISION NOT NULL,
    mcs_p_value  DOUBLE PRECISION NOT NULL,
    eliminated   SMALLINT,
    in_set       BOOLEAN NOT NULL,
    PRIMARY KEY (horizon, model)
);

-- ---------------------------------------------------------------------
-- 15. Scores of the controlled experiments around HAR-X:
--     experiment = 'ablation' (feature groups added or removed),
--                  'unseen_region' (model estimated without the region),
--     segment    = 'all', an asset type, or a region.
--     bias = average of actual / forecast (1 = right on average).
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_scores (
    experiment  TEXT NOT NULL,
    variant     TEXT NOT NULL,
    horizon     SMALLINT NOT NULL,
    segment     TEXT NOT NULL,
    n           INTEGER NOT NULL,
    qlike       DOUBLE PRECISION NOT NULL,
    log_mse     DOUBLE PRECISION NOT NULL,
    mae_vol     DOUBLE PRECISION NOT NULL,
    bias        DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (experiment, variant, horizon, segment)
);

-- ---------------------------------------------------------------------
-- 16. Robustness of the forecast combination to the choice of its members
--     (the five members overlap: the hybrid is HAR-X corrected by LightGBM).
--     qlike_vs_harx / qlike_vs_main: relative difference in QLIKE;
--     p-values from Diebold-Mariano tests on the daily average QLIKE.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ensemble_variants (
    variant        TEXT NOT NULL,
    horizon        SMALLINT NOT NULL,
    n_members      SMALLINT NOT NULL,
    n              INTEGER NOT NULL,
    qlike          DOUBLE PRECISION NOT NULL,
    mae_vol        DOUBLE PRECISION NOT NULL,
    qlike_vs_harx  DOUBLE PRECISION NOT NULL,
    p_vs_harx      DOUBLE PRECISION,
    qlike_vs_main  DOUBLE PRECISION NOT NULL,
    p_vs_main      DOUBLE PRECISION,
    PRIMARY KEY (variant, horizon)
);

-- =====================================================================
-- Step 10: Value-at-Risk backtest
-- Written by `python -m src.risk.run_var_backtest`.
-- The 1-day VaR of a model is  - quantile * forecast volatility,  with
--   method = 'normal'     quantile of the normal distribution,
--            'empirical'  quantile of the model's own past standardised returns,
--            'historical' benchmark without a volatility model
--                         (model = 'historical_simulation': quantile of the last 250 returns).
-- confidence is 95 or 99 (percent). mean_var and quantile_loss are in return
-- units (0.03 = 3%).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 17. Empirical quantiles: the quantile of standardised returns used for
--     each model, series and year, estimated on the earlier test years.
--     pooled = TRUE when the series had too little history of its own and
--     the quantile of all series together was used.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_quantiles (
    model       TEXT NOT NULL REFERENCES models (model),
    confidence  SMALLINT NOT NULL CHECK (confidence IN (95, 99)),
    ticker      TEXT NOT NULL REFERENCES universe (ticker),
    year        SMALLINT NOT NULL,
    quantile    DOUBLE PRECISION NOT NULL,
    n_obs       INTEGER NOT NULL,
    pooled      BOOLEAN NOT NULL,
    PRIMARY KEY (model, confidence, ticker, year)
);

-- ---------------------------------------------------------------------
-- 18. Backtest per series over the whole test period.
--     kupiec_p: test of the number of violations; independence_p:
--     Christoffersen test of clustering; cond_coverage_p: both together.
--     A p-value below 0.05 means that the VaR fails the test.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_backtest (
    model            TEXT NOT NULL,
    method           TEXT NOT NULL CHECK (method IN ('normal', 'empirical', 'historical')),
    confidence       SMALLINT NOT NULL CHECK (confidence IN (95, 99)),
    ticker           TEXT NOT NULL REFERENCES universe (ticker),
    n                INTEGER NOT NULL,
    violations       INTEGER NOT NULL,
    mean_var         DOUBLE PRECISION NOT NULL,
    quantile_loss    DOUBLE PRECISION NOT NULL,
    kupiec_stat      DOUBLE PRECISION NOT NULL,
    kupiec_p         DOUBLE PRECISION NOT NULL,
    independence_p   DOUBLE PRECISION,
    cond_coverage_p  DOUBLE PRECISION,
    PRIMARY KEY (model, method, confidence, ticker)
);

-- ---------------------------------------------------------------------
-- 19. Backtest per series and calendar year. zone is the Basel traffic
--     light (green / yellow / red) for the number of violations of the year.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_backtest_yearly (
    model          TEXT NOT NULL,
    method         TEXT NOT NULL CHECK (method IN ('normal', 'empirical', 'historical')),
    confidence     SMALLINT NOT NULL CHECK (confidence IN (95, 99)),
    ticker         TEXT NOT NULL REFERENCES universe (ticker),
    year           SMALLINT NOT NULL,
    n              INTEGER NOT NULL,
    violations     INTEGER NOT NULL,
    mean_var       DOUBLE PRECISION NOT NULL,
    quantile_loss  DOUBLE PRECISION NOT NULL,
    zone           TEXT NOT NULL CHECK (zone IN ('green', 'yellow', 'red')),
    PRIMARY KEY (model, method, confidence, ticker, year)
);

-- ---------------------------------------------------------------------
-- 20. Diebold-Mariano tests on the daily average quantile loss, each model
--     against a reference (HAR-X with the same method).
--     loss_vs_reference: relative difference, negative = better VaR.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_tests (
    model              TEXT NOT NULL,
    method             TEXT NOT NULL,
    confidence         SMALLINT NOT NULL,
    reference          TEXT NOT NULL,
    n_dates            INTEGER NOT NULL,
    quantile_loss      DOUBLE PRECISION NOT NULL,
    loss_vs_reference  DOUBLE PRECISION NOT NULL,
    dm_stat            DOUBLE PRECISION,
    p_value            DOUBLE PRECISION,
    PRIMARY KEY (model, method, confidence)
);

-- ---------------------------------------------------------------------
-- 21. Daily VaR of a few models, next to the return it had to cover
--     (input of the Power BI dashboard). date = day of the return;
--     var_* = VaR with the empirical multiplier (positive, 0.03 = 3%),
--     var_historical_simulation = benchmark without a volatility model.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS var_daily (
    ticker                     TEXT NOT NULL REFERENCES universe (ticker),
    date                       DATE NOT NULL,
    confidence                 SMALLINT NOT NULL CHECK (confidence IN (95, 99)),
    ret                        DOUBLE PRECISION NOT NULL,
    var_har_x                  DOUBLE PRECISION,
    var_ensemble_mean          DOUBLE PRECISION,
    var_historical_simulation  DOUBLE PRECISION,
    PRIMARY KEY (ticker, date, confidence)
);

-- Overall result of each VaR: violation rate, average size, and the share of
-- the series that pass each test at the 5% level
CREATE OR REPLACE VIEW var_backtest_overall AS
SELECT b.model,
       b.method,
       b.confidence,
       m.family,
       count(*)                                              AS n_series,
       sum(b.n)                                              AS n,
       sum(b.violations)                                     AS violations,
       sum(b.violations)::double precision / sum(b.n)        AS violation_rate,
       (1 - b.confidence / 100.0)::double precision          AS expected_rate,
       sum(b.mean_var * b.n) / sum(b.n)                      AS mean_var,
       sum(b.quantile_loss * b.n) / sum(b.n)                 AS quantile_loss,
       avg((b.kupiec_p >= 0.05)::int)::double precision        AS pass_kupiec,
       avg((b.independence_p >= 0.05)::int)::double precision  AS pass_independence,
       avg((b.cond_coverage_p >= 0.05)::int)::double precision AS pass_cond_coverage
FROM var_backtest b
LEFT JOIN models m USING (model)
GROUP BY b.model, b.method, b.confidence, m.family
ORDER BY b.confidence, b.method, quantile_loss;
