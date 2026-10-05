"""Controlled experiments around the best linear model (HAR-X).

HAR-X is cheap to estimate (a linear regression), so it can be re-run many
times with one thing changed at a time. Three questions:

* **Ablation**: which groups of features carry the information? Each group is
  added alone to the plain HAR, and removed alone from the full HAR-X.
* **Stock / index flag**: the tree and neural models receive `is_index` and
  HAR-X does not. Giving it to HAR-X separates what comes from that piece of
  information from what comes from non-linearity.
* **Unseen markets**: is the pooled model specific to the series it was trained
  on? For each region, a model is estimated without any series of that region
  and then used to forecast them.

Forecast combinations are also built here: the average and the median of the
five models that use the full feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import add_is_index
from src.models.har import HAR_FEATURES, Har, HarX

FEATURE_GROUPS = {
    "slow volatility (quarter, long run)": ["log_rv_q", "log_rv_lt"],
    "returns and leverage": ["ret_d", "ret_w", "ret_m", "neg_ret_d", "down_share_m"],
    "instability and volume": ["vol_of_vol_m", "log_volume_ratio"],
    "market-wide volatility": ["mkt_log_rv_d", "mkt_log_rv_w", "mkt_log_rv_m"],
    "VIX": ["log_vix", "vix_chg_5d"],
}

# Stocks and the national index of each region are held out together
REGIONS = {
    "France": ["France"],
    "Germany": ["Germany"],
    "United Kingdom": ["United Kingdom"],
    "Switzerland": ["Switzerland"],
    "Benelux": ["Netherlands", "Belgium", "Luxembourg"],
    "Southern Europe": ["Spain", "Italy"],
    "Nordics": ["Sweden", "Denmark", "Finland", "Norway"],
}

ENSEMBLE_MEMBERS = ["har_x", "lgbm", "lgbm_hybrid", "lstm", "transformer"]


class LinearVariant(Har):
    """A pooled linear regression on a chosen list of features (same estimation as HAR and HAR-X)."""

    def __init__(self, name: str, features: list[str]) -> None:
        super().__init__()
        self.name = name
        self.features = list(features)

    def prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        return add_is_index(data) if "is_index" in self.features else data


def ablation_variants() -> list[LinearVariant]:
    """HAR, HAR plus one group, HAR-X, HAR-X minus one group, HAR-X plus the stock / index flag."""
    all_extra = [f for group in FEATURE_GROUPS.values() for f in group]
    variants = [LinearVariant("HAR", HAR_FEATURES)]
    variants += [LinearVariant(f"HAR + {name}", HAR_FEATURES + features) for name, features in FEATURE_GROUPS.items()]
    variants += [LinearVariant("HAR-X", HAR_FEATURES + all_extra)]
    variants += [LinearVariant(f"HAR-X - {name}", HAR_FEATURES + [f for f in all_extra if f not in features])
                 for name, features in FEATURE_GROUPS.items()]
    variants += [LinearVariant("HAR-X + is_index", HAR_FEATURES + all_extra + ["is_index"])]
    return variants


class HeldOutHarX(HarX):
    """HAR-X estimated without the series of one region, used to forecast only those series."""

    def __init__(self, region: str, tickers: list[str]) -> None:
        super().__init__()
        self.name = f"without {region}"
        self.region = region
        self.held_out = set(tickers)

    def fit(self, train: pd.DataFrame, horizon: int) -> "HeldOutHarX":
        return super().fit(train[~train["ticker"].isin(self.held_out)], horizon)

    def predict(self, test: pd.DataFrame, horizon: int) -> pd.Series:
        return super().predict(test, horizon).where(test["ticker"].isin(self.held_out))


def region_tickers(universe: pd.DataFrame) -> dict[str, list[str]]:
    """Tickers (stocks and indices) of each region; `universe` has the columns ticker and country."""
    return {region: sorted(universe.loc[universe["country"].isin(countries), "ticker"])
            for region, countries in REGIONS.items()}


def held_out_models(universe: pd.DataFrame) -> list[HeldOutHarX]:
    return [HeldOutHarX(region, tickers) for region, tickers in region_tickers(universe).items()]


def build_ensembles(forecasts: pd.DataFrame, members: list[str] = ENSEMBLE_MEMBERS) -> pd.DataFrame:
    """Combine the members' variance forecasts: equal-weight average and median.

    `forecasts` has the columns horizon, ticker, date and one column per member.
    A combination is only produced where every member has a forecast. Returns
    long predictions (model, horizon, ticker, date, pred_var).
    """
    complete = forecasts.dropna(subset=members)
    keys = complete[["horizon", "ticker", "date"]].reset_index(drop=True)
    values = complete[members].to_numpy(dtype=float)
    combinations = {"ensemble_mean": values.mean(axis=1), "ensemble_median": np.median(values, axis=1)}
    return pd.concat([keys.assign(model=name, pred_var=combined) for name, combined in combinations.items()],
                     ignore_index=True)[["model", "horizon", "ticker", "date", "pred_var"]]


def score_by_segment(predictions: pd.DataFrame, segment: pd.Series | None = None) -> pd.DataFrame:
    """Average losses per model and horizon (and per value of `segment`, aligned with `predictions`).

    Adds `bias`, the average of actual / forecast (1 = right on average, below 1 = forecasts too high).
    """
    from src.evaluation.metrics import add_losses

    scored = add_losses(predictions)
    scored["bias"] = scored["actual_var"] / scored["pred_var"]
    scored["segment"] = "all" if segment is None else np.asarray(segment)
    grouped = scored.groupby(["model", "horizon", "segment"], observed=True)
    out = grouped[["qlike", "log_mse", "mae_vol", "bias"]].mean()
    out.insert(0, "n", grouped.size())
    return out.reset_index().rename(columns={"model": "variant"})
