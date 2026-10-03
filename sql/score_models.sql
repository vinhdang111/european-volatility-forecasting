-- =====================================================================
-- Recompute model_scores from the forecasts (view forecasts_long: one row per model).
-- Run by src/evaluation/store.py after any model has written forecasts.
--
-- Losses (actual and forecast are average daily variances):
--   QLIKE    actual/forecast - ln(actual/forecast) - 1
--   log MSE  (ln actual - ln forecast)^2
--   MAE      |actual vol - forecast vol| in annualised volatility points
--
-- Common sample: an observation (horizon, ticker, date) is scored only if
-- EVERY model has a forecast for it, so that all models are compared on
-- exactly the same days.
-- =====================================================================
TRUNCATE model_scores;

INSERT INTO model_scores (model, horizon, ticker, test_year, n, qlike, log_mse, mae_vol)
WITH n_models AS (
    SELECT count(*) AS n FROM models
),
joined AS (
    SELECT f.model,
           f.horizon,
           f.ticker,
           f.date,
           f.pred_var::double precision AS forecast,
           exp(CASE f.horizon WHEN 1 THEN x.target_1d
                              WHEN 5 THEN x.target_5d
                              ELSE x.target_22d END) AS actual,
           count(*) OVER (PARTITION BY f.horizon, f.ticker, f.date) AS models_with_forecast
    FROM forecasts_long f
    JOIN features x USING (ticker, date)
)
SELECT model,
       horizon,
       ticker,
       extract(year FROM date)::smallint                               AS test_year,
       count(*)                                                        AS n,
       avg(actual / forecast - ln(actual / forecast) - 1)              AS qlike,
       avg((ln(actual) - ln(forecast)) ^ 2)                            AS log_mse,
       avg(abs(sqrt(actual * 252) - sqrt(forecast * 252)) * 100)       AS mae_vol
FROM joined, n_models
WHERE actual IS NOT NULL
  AND models_with_forecast = n_models.n
GROUP BY model, horizon, ticker, extract(year FROM date);
