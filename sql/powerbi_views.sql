-- =====================================================================
-- Views read by the Power BI dashboard (powerbi/).
-- Created by `python -m src.dashboard.setup_views`, after Steps 5 to 10
-- (they read the forecasts and the results of the comparison and of the
-- VaR backtest). Safe to re-run.
--
-- Star schema: five dimension views (pbi_series, pbi_models,
-- pbi_horizons, pbi_confidence, pbi_calendar) and fact views that point
-- to them through ticker, model, horizon, confidence and date.
-- Labels and sort orders are defined here once, so that every page of
-- the dashboard uses the same names as the README and the notebooks.
-- =====================================================================

-- Views are dropped first: CREATE OR REPLACE cannot change column types.
DROP VIEW IF EXISTS pbi_series, pbi_models, pbi_horizons, pbi_confidence, pbi_calendar,
                    pbi_model_scores, pbi_model_tests, pbi_regime_scores, pbi_forecast_paths,
                    pbi_var_backtest, pbi_var_yearly, pbi_var_tests, pbi_var_daily CASCADE;

-- ---------------------------------------------------------------------
-- Dimensions
-- ---------------------------------------------------------------------

-- The 59 forecast series (50 stocks, 9 indices)
CREATE VIEW pbi_series AS
SELECT u.ticker,
       u.name,
       u.name || ' (' || u.ticker || ')'                    AS series_label,
       initcap(u.asset_type)                                 AS asset_type,
       u.country,
       coalesce(u.sector, 'Index')                           AS sector
FROM universe u
WHERE u.asset_type IN ('stock', 'index')
  AND EXISTS (SELECT 1 FROM forecasts f WHERE f.ticker = u.ticker);

-- Models, their family and a fixed display order (plus the VaR benchmark)
CREATE VIEW pbi_models AS
SELECT m.model, l.model_label, l.family, l.family_label, l.family_order, l.model_order
FROM (SELECT model FROM models UNION SELECT 'historical_simulation') m
JOIN (VALUES
    ('naive',                 'Naive (last value)',    'baseline',         'Baselines',             1,  1),
    ('historical_mean',       'Historical mean',       'baseline',         'Baselines',             1,  2),
    ('monthly_average',       'Monthly average',       'baseline',         'Baselines',             1,  3),
    ('riskmetrics',           'RiskMetrics',           'baseline',         'Baselines',             1,  4),
    ('ewma',                  'EWMA (range-based)',    'baseline',         'Baselines',             1,  5),
    ('garch',                 'GARCH(1,1)',            'econometric',      'Econometric models',    2,  6),
    ('gjr_garch',             'GJR-GARCH',             'econometric',      'Econometric models',    2,  7),
    ('har',                   'HAR',                   'econometric',      'Econometric models',    2,  8),
    ('har_x',                 'HAR-X',                 'econometric',      'Econometric models',    2,  9),
    ('lgbm',                  'LightGBM',              'machine_learning', 'Machine learning',      3, 10),
    ('lgbm_hybrid',           'HAR-X + LightGBM',      'machine_learning', 'Machine learning',      3, 11),
    ('lstm',                  'LSTM',                  'deep_learning',    'Deep learning',         4, 12),
    ('transformer',           'Transformer',           'deep_learning',    'Deep learning',         4, 13),
    ('ensemble_median',       'Ensemble (Median)',     'ensemble',         'Combinations',          5, 14),
    ('ensemble_mean',         'Ensemble (Mean)',       'ensemble',         'Combinations',          5, 15),
    ('historical_simulation', 'Historical simulation', 'benchmark',        'No volatility model',   6, 16)
) AS l (model, model_label, family, family_label, family_order, model_order) USING (model);

CREATE VIEW pbi_horizons AS
SELECT * FROM (VALUES (1::smallint, '1 day', 1), (5::smallint, '1 week (5 days)', 2), (22::smallint, '1 month (22 days)', 3))
       AS h (horizon, horizon_label, horizon_order);

CREATE VIEW pbi_confidence AS
SELECT * FROM (VALUES (95::smallint, '95% VaR', 0.05::double precision), (99::smallint, '99% VaR', 0.01::double precision))
       AS c (confidence, confidence_label, promised_rate);

-- One row per calendar day of the test period (mark it as the date table in Power BI)
CREATE VIEW pbi_calendar AS
SELECT d::date                                AS date,
       extract(year FROM d)::int              AS year,
       extract(month FROM d)::int             AS month,
       to_char(d, 'Mon YYYY')                 AS month_label,
       (extract(year FROM d)::int * 100 + extract(month FROM d)::int) AS month_key
FROM generate_series((SELECT date_trunc('year', min(date)) FROM forecasts),
                     (SELECT date_trunc('year', max(date)) + interval '1 year - 1 day' FROM forecasts),
                     interval '1 day') AS d;

-- ---------------------------------------------------------------------
-- Model comparison (Steps 5 to 9)
-- ---------------------------------------------------------------------

-- Losses per model, horizon, series and test year (common sample)
CREATE VIEW pbi_model_scores AS
SELECT model, horizon, ticker, test_year, n, qlike, mae_vol
FROM model_scores;

-- Full-sample tests against HAR-X: Diebold-Mariano p-value and membership
-- of the 90% Model Confidence Set
CREATE VIEW pbi_model_tests AS
SELECT o.model,
       o.horizon,
       o.qlike,
       o.qlike / r.qlike - 1                                 AS qlike_vs_harx,
       t.p_value,
       CASE WHEN t.p_value < 0.01 THEN '***' WHEN t.p_value < 0.05 THEN '**'
            WHEN t.p_value < 0.10 THEN '*' ELSE '' END       AS significance,
       CASE WHEN c.mcs_p_value >= 0.10 THEN 'Yes' ELSE 'No' END AS in_confidence_set
FROM model_scores_overall o
JOIN model_scores_overall r ON r.horizon = o.horizon AND r.model = 'har_x'
LEFT JOIN model_tests t ON t.horizon = o.horizon AND t.model = o.model AND t.reference = 'har_x'
LEFT JOIN model_confidence_set c ON c.horizon = o.horizon AND c.model = o.model;

-- QLIKE and MAE of every model in calm and in stress periods, pooled over
-- the 59 series. Stress periods are the market crises of src/viz.py
-- (CRISES); every other day is calm. Same common sample as model_scores:
-- only the days on which all fifteen models have a forecast.
CREATE VIEW pbi_regime_scores AS
WITH complete AS (
    SELECT f.*,
           exp(CASE f.horizon WHEN 1 THEN x.target_1d WHEN 5 THEN x.target_5d ELSE x.target_22d END) AS actual,
           CASE WHEN f.date BETWEEN '2000-03-01' AND '2003-03-31'      -- Dot-com bust
                  OR f.date BETWEEN '2007-08-01' AND '2009-06-30'      -- Financial crisis
                  OR f.date BETWEEN '2011-07-01' AND '2012-07-31'      -- Euro debt crisis
                  OR f.date BETWEEN '2020-02-15' AND '2020-06-30'      -- COVID-19
                  OR f.date BETWEEN '2022-02-15' AND '2022-10-31'      -- Energy & rate shock
                THEN 'Stress periods' ELSE 'Calm periods' END AS regime
    FROM forecasts f
    JOIN features x USING (ticker, date)
    WHERE f.naive IS NOT NULL
      AND f.historical_mean IS NOT NULL
      AND f.monthly_average IS NOT NULL
      AND f.riskmetrics IS NOT NULL
      AND f.ewma IS NOT NULL
      AND f.garch IS NOT NULL
      AND f.gjr_garch IS NOT NULL
      AND f.har IS NOT NULL
      AND f.har_x IS NOT NULL
      AND f.lgbm IS NOT NULL
      AND f.lgbm_hybrid IS NOT NULL
      AND f.lstm IS NOT NULL
      AND f.transformer IS NOT NULL
      AND f.ensemble_median IS NOT NULL
      AND f.ensemble_mean IS NOT NULL
)
SELECT v.model,
       c.horizon,
       c.regime,
       count(*)                                                          AS n,
       avg(c.actual / v.forecast - ln(c.actual / v.forecast) - 1)        AS qlike,
       avg(abs(sqrt(c.actual * 252) - sqrt(v.forecast * 252)) * 100)     AS mae_vol
FROM complete c
CROSS JOIN LATERAL (VALUES
    ('naive', c.naive::double precision),
    ('historical_mean', c.historical_mean::double precision),
    ('monthly_average', c.monthly_average::double precision),
    ('riskmetrics', c.riskmetrics::double precision),
    ('ewma', c.ewma::double precision),
    ('garch', c.garch::double precision),
    ('gjr_garch', c.gjr_garch::double precision),
    ('har', c.har::double precision),
    ('har_x', c.har_x::double precision),
    ('lgbm', c.lgbm::double precision),
    ('lgbm_hybrid', c.lgbm_hybrid::double precision),
    ('lstm', c.lstm::double precision),
    ('transformer', c.transformer::double precision),
    ('ensemble_median', c.ensemble_median::double precision),
    ('ensemble_mean', c.ensemble_mean::double precision)
) AS v (model, forecast)
WHERE c.actual IS NOT NULL
GROUP BY v.model, c.horizon, c.regime;

-- Forecast and realised volatility through time, for the realised value and
-- five models. One row per series, forecast date, horizon and line;
-- realised_pct repeats the realised value on every row so that errors can be
-- computed for any period selected in the dashboard.
-- Volatilities are annualised, in percent.
CREATE VIEW pbi_forecast_paths AS
WITH base AS (
    SELECT f.ticker, f.date, f.horizon,
           sqrt(exp(CASE f.horizon WHEN 1 THEN x.target_1d WHEN 5 THEN x.target_5d ELSE x.target_22d END) * 252) * 100 AS realised,
           sqrt(f.ewma * 252) * 100          AS ewma,
           sqrt(f.gjr_garch * 252) * 100     AS gjr_garch,
           sqrt(f.har_x * 252) * 100         AS har_x,
           sqrt(f.transformer * 252) * 100   AS transformer,
           sqrt(f.ensemble_mean * 252) * 100 AS ensemble_mean
    FROM forecasts f
    JOIN features x USING (ticker, date)
    WHERE CASE f.horizon WHEN 1 THEN x.target_1d WHEN 5 THEN x.target_5d ELSE x.target_22d END IS NOT NULL
      AND f.ewma IS NOT NULL AND f.gjr_garch IS NOT NULL AND f.har_x IS NOT NULL
      AND f.transformer IS NOT NULL AND f.ensemble_mean IS NOT NULL
)
SELECT b.ticker, b.date, b.horizon, l.line, l.line_order,
       round(l.vol::numeric, 2)::double precision      AS vol_pct,
       round(b.realised::numeric, 2)::double precision AS realised_pct
FROM base b
CROSS JOIN LATERAL (VALUES
    ('Realised',             0, b.realised),
    ('EWMA (range-based)',   1, b.ewma),
    ('GJR-GARCH',            2, b.gjr_garch),
    ('HAR-X',                3, b.har_x),
    ('Transformer',          4, b.transformer),
    ('Ensemble (Mean)',      5, b.ensemble_mean)
) AS l (line, line_order, vol);

-- ---------------------------------------------------------------------
-- Value-at-Risk backtest (Step 10)
-- ---------------------------------------------------------------------

-- One row per model, method, confidence level and series
CREATE VIEW pbi_var_backtest AS
SELECT model, method, confidence, ticker, n, violations, mean_var, quantile_loss,
       kupiec_p, independence_p,
       CASE WHEN kupiec_p >= 0.05 THEN 1 ELSE 0 END                                   AS passes_kupiec,
       CASE WHEN independence_p IS NULL THEN NULL WHEN independence_p >= 0.05 THEN 1 ELSE 0 END AS passes_independence
FROM var_backtest;

-- One row per model, method, confidence level, series and calendar year;
-- full_year marks the years used for the Basel traffic light
CREATE VIEW pbi_var_yearly AS
SELECT model, method, confidence, ticker, year, n, violations, zone,
       CASE zone WHEN 'green' THEN 1 WHEN 'yellow' THEN 2 ELSE 3 END AS zone_order,
       initcap(zone) || ' zone'                                       AS zone_label,
       (n >= 240)                                                     AS full_year
FROM var_backtest_yearly;

-- Full-sample Diebold-Mariano tests of the quantile loss against HAR-X
-- (empirical multiplier, and the historical-simulation benchmark)
CREATE VIEW pbi_var_tests AS
SELECT model, confidence, loss_vs_reference AS quantile_loss_vs_harx, p_value,
       CASE WHEN p_value < 0.01 THEN '***' WHEN p_value < 0.05 THEN '**'
            WHEN p_value < 0.10 THEN '*' ELSE '' END AS significance
FROM var_tests
WHERE method IN ('empirical', 'historical');

-- Daily return and three 99% / 95% VaR limits, in percent of the position
CREATE VIEW pbi_var_daily AS
SELECT ticker, date, confidence,
       ret * 100                        AS return_pct,
       var_har_x * 100                  AS var_har_x_pct,
       var_ensemble_mean * 100          AS var_ensemble_pct,
       var_historical_simulation * 100  AS var_historical_pct
FROM var_daily;
