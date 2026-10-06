"""From a volatility forecast to a Value-at-Risk (VaR).

The 1-day VaR at 99% is the loss that should be exceeded on only 1% of the
days. With a forecast of tomorrow's volatility it is

    VaR(t+1) = - quantile * forecast volatility(t+1)

where `quantile` is the 1% quantile of the *standardised* returns (return
divided by forecast volatility). The mean daily return is taken as zero, as is
usual at a one-day horizon. Two choices of quantile are compared:

* **normal**: the quantile of the normal distribution (-1.645 at 95%, -2.326 at
  99%). The textbook formula; it ignores that returns have fat tails.
* **empirical** (filtered historical simulation): the quantile actually observed
  in the past standardised returns of the same model. It is re-estimated once a
  year, for each series, on the test years before the one being forecast, so a
  VaR never uses information from its own future. A series with fewer than
  `MIN_OBS_SERIES` past observations uses the quantile of all series together.
  The first test year only serves to estimate the first quantiles.

A benchmark that uses no volatility model is added: **historical simulation**,
the quantile of the last 250 daily returns of the series, still widely used by
banks.

The VaR is expressed as a positive number: 0.03 means a loss of 3%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sqlalchemy import Engine

from src.evaluation.store import check_model_name, stored_models

CONFIDENCE_LEVELS = (95, 99)
METHODS = ("normal", "empirical")
MIN_OBS_SERIES = 750          # about three years of a series' own standardised returns
HS_WINDOW = 250               # trading days used by historical simulation
HS_MIN_OBS = 200
HS_MODEL = "historical_simulation"
HS_METHOD = "historical"


def tail_probability(confidence: int) -> float:
    """Share of days on which the loss should exceed the VaR: 0.01 at 99%."""
    return 1.0 - confidence / 100.0


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def var_data_sql(models: list[str]) -> str:
    """1-day variance forecasts of every model next to the return of the following trading day.

    Same sample as the model comparison: the rows that every model has forecast.
    """
    columns = ", ".join(f"f.{check_model_name(m)}::double precision AS {m}" for m in models)
    complete = " AND ".join(f"f.{m} IS NOT NULL" for m in models)
    return (f"WITH next_day AS (\n"
            f"    SELECT ticker, date,\n"
            f"           lead(date)  OVER (PARTITION BY ticker ORDER BY date) AS return_date,\n"
            f"           lead(ret_d) OVER (PARTITION BY ticker ORDER BY date) AS next_ret\n"
            f"    FROM features)\n"
            f"SELECT f.ticker, f.date, d.return_date, d.next_ret, {columns}\n"
            f"FROM forecasts f\n"
            f"JOIN features x USING (ticker, date)\n"
            f"JOIN next_day d USING (ticker, date)\n"
            f"WHERE f.horizon = 1 AND x.target_1d IS NOT NULL AND d.next_ret IS NOT NULL AND {complete}\n"
            f"ORDER BY f.ticker, f.date")


def load_var_data(engine: Engine) -> tuple[pd.DataFrame, list[str]]:
    """Forecasts and next-day returns, sorted by series and date, with the year of the return."""
    models = stored_models(engine)
    data = pd.read_sql(var_data_sql(models), engine, parse_dates=["date", "return_date"])
    data["year"] = data["return_date"].dt.year
    return data, models


# ---------------------------------------------------------------------------
# Quantiles of the standardised returns
# ---------------------------------------------------------------------------
def standardised_returns(data: pd.DataFrame, model: str) -> pd.Series:
    """Next-day return divided by the volatility the model forecast for that day."""
    return data["next_ret"] / np.sqrt(data[model])


def empirical_quantiles(data: pd.DataFrame, model: str, confidence: int,
                        min_obs: int = MIN_OBS_SERIES) -> pd.DataFrame:
    """Quantile of the past standardised returns, for every series and year.

    The quantile used in year Y is estimated on the years before Y only: on the
    series' own history when it has at least `min_obs` observations, otherwise
    on all series together (`pooled`). The first year has no quantile.
    Returns the columns ticker, year, quantile, n_obs, pooled.
    """
    p = tail_probability(confidence)
    frame = pd.DataFrame({"ticker": data["ticker"].to_numpy(), "year": data["year"].to_numpy(),
                          "z": standardised_returns(data, model).to_numpy()})
    rows = []
    for year in sorted(frame["year"].unique())[1:]:
        past = frame[frame["year"] < year]
        pooled = float(past["z"].quantile(p))
        own = past.groupby("ticker")["z"].agg(quantile=lambda z: z.quantile(p), n_obs="size")
        table = pd.DataFrame({"ticker": frame.loc[frame["year"] == year, "ticker"].unique(), "year": year})
        table = table.merge(own, left_on="ticker", right_index=True, how="left")
        table["n_obs"] = table["n_obs"].fillna(0).astype(int)
        table["pooled"] = table["n_obs"] < min_obs
        table.loc[table["pooled"], ["quantile", "n_obs"]] = [pooled, len(past)]
        rows.append(table)
    columns = ["ticker", "year", "quantile", "n_obs", "pooled"]
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def apply_quantiles(data: pd.DataFrame, model: str, quantiles: pd.DataFrame) -> pd.Series:
    """VaR = - quantile(series, year) * forecast volatility. Missing where there is no quantile yet."""
    lookup = quantiles.set_index(["ticker", "year"])["quantile"]
    quantile = lookup.reindex(pd.MultiIndex.from_frame(data[["ticker", "year"]])).to_numpy(dtype=float)
    return pd.Series(-quantile * np.sqrt(data[model].to_numpy()), index=data.index)


def value_at_risk(data: pd.DataFrame, model: str, method: str, confidence: int,
                  min_obs: int = MIN_OBS_SERIES) -> pd.Series:
    """1-day VaR (positive number) of every row of `data` for one model, method and confidence level."""
    if method == "normal":
        return -norm.ppf(tail_probability(confidence)) * np.sqrt(data[model])
    if method == "empirical":
        return apply_quantiles(data, model, empirical_quantiles(data, model, confidence, min_obs))
    raise ValueError(f"Unknown method: {method!r}")


# ---------------------------------------------------------------------------
# Benchmark without a volatility model
# ---------------------------------------------------------------------------
def historical_simulation(returns: pd.DataFrame, confidence: int, window: int = HS_WINDOW,
                          min_obs: int = HS_MIN_OBS) -> pd.DataFrame:
    """VaR for the next day = minus the quantile of the last `window` daily returns of the series.

    `returns` has the columns ticker, date, ret_d. The VaR on a row uses the
    returns up to and including that date, and applies to the following day.
    """
    p = tail_probability(confidence)
    ordered = returns.sort_values(["ticker", "date"])
    quantile = (ordered.groupby("ticker", sort=False)["ret_d"]
                .transform(lambda r: r.rolling(window, min_periods=min_obs).quantile(p)))
    return pd.DataFrame({"ticker": ordered["ticker"], "date": ordered["date"], "var": -quantile})
