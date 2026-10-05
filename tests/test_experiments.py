"""Unit tests for the controlled experiments and the comparison pipeline (no database needed)."""
import numpy as np
import pandas as pd
import pytest

from src.db import PROJECT_ROOT
from src.evaluation.experiments import (ENSEMBLE_MEMBERS, FEATURE_GROUPS, REGIONS, HeldOutHarX, LinearVariant,
                                        ablation_variants, build_ensembles, held_out_models, region_tickers,
                                        score_by_segment)
from src.evaluation.run_comparison import confidence_sets, on_common_sample, pairwise_tests
from src.evaluation.store import common_sample_sql, daily_losses_sql, replace_table
from src.features.build_features import build_features
from src.models.har import EXTRA_FEATURES, HAR_FEATURES, Har, HarX
from tests.test_features import make_panel


@pytest.fixture(scope="module")
def dataset():
    return build_features(make_panel(n_days=900), min_market_stocks=2)


@pytest.fixture(scope="module")
def split(dataset):
    dates = dataset["date"].sort_values().unique()
    train = dataset[dataset["date"] <= dates[700]]
    test = dataset[dataset["date"] > dates[700]].dropna(subset=HAR_FEATURES)
    return train, test


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
def test_feature_groups_cover_exactly_the_extra_features_of_harx():
    grouped = [f for group in FEATURE_GROUPS.values() for f in group]
    assert sorted(grouped) == sorted(EXTRA_FEATURES) and len(grouped) == len(set(grouped))


def test_ablation_variants_add_and_remove_one_group_at_a_time():
    variants = {v.name: v.features for v in ablation_variants()}
    assert len(variants) == 2 + 2 * len(FEATURE_GROUPS) + 1
    assert variants["HAR"] == HAR_FEATURES
    assert sorted(variants["HAR-X"]) == sorted(HAR_FEATURES + EXTRA_FEATURES)
    for name, features in FEATURE_GROUPS.items():
        assert sorted(variants[f"HAR + {name}"]) == sorted(HAR_FEATURES + features)
        assert sorted(variants[f"HAR-X - {name}"]) == sorted(set(variants["HAR-X"]) - set(features))
    assert set(variants["HAR-X + is_index"]) - set(variants["HAR-X"]) == {"is_index"}


def test_linear_variant_with_the_har_features_is_the_har_model(split):
    train, test = split
    variant = LinearVariant("HAR", HAR_FEATURES).fit(train, 5).predict(test, 5)
    assert np.allclose(variant, Har().fit(train, 5).predict(test, 5))


def test_is_index_variant_gets_its_feature_from_the_asset_type(dataset, split):
    model = LinearVariant("HAR-X + is_index", HAR_FEATURES + EXTRA_FEATURES + ["is_index"])
    data = model.prepare(dataset)
    assert set(data.loc[data["is_index"] == 1, "ticker"]) == {"^IDX"}
    dates = data["date"].sort_values().unique()
    forecast = model.fit(data[data["date"] <= dates[700]], 5).predict(data[data["date"] > dates[700]].dropna(subset=HAR_FEATURES), 5)
    assert forecast.notna().all() and (forecast > 0).all()


# ---------------------------------------------------------------------------
# Unseen regions
# ---------------------------------------------------------------------------
def test_every_stock_and_national_index_belongs_to_exactly_one_region():
    universe = pd.read_csv(PROJECT_ROOT / "data" / "universe.csv")
    tickers = region_tickers(universe)
    assigned = [t for group in tickers.values() for t in group]
    assert len(assigned) == len(set(assigned))
    assert set(universe["ticker"]) - set(assigned) == {"^STOXX50E", "^VIX"}       # euro-area index and VIX: no region
    assert set(tickers) == set(REGIONS) and all(len(group) >= 5 for group in tickers.values())
    assert [m.region for m in held_out_models(universe)] == list(REGIONS)


def test_held_out_model_never_trains_on_the_region_and_only_forecasts_it(split):
    train, test = split
    model = HeldOutHarX("Testland", ["S1.PA", "^IDX"]).fit(train, 5)
    expected = HarX().fit(train[~train["ticker"].isin(["S1.PA", "^IDX"])], 5)
    assert np.allclose(model.coef_, expected.coef_)
    assert not np.allclose(model.coef_, HarX().fit(train, 5).coef_)                 # the full model is different

    forecast = model.predict(test, 5)
    held_out = test["ticker"].isin(["S1.PA", "^IDX"])
    assert forecast[held_out].notna().all() and forecast[~held_out].isna().all()


# ---------------------------------------------------------------------------
# Forecast combinations
# ---------------------------------------------------------------------------
def test_ensembles_are_the_mean_and_the_median_of_the_members():
    forecasts = pd.DataFrame({"horizon": [5, 5, 5], "ticker": ["A", "A", "B"],
                              "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
                              "har_x": [1.0, 2.0, 1.0], "lgbm": [2.0, 2.0, 1.0], "lgbm_hybrid": [3.0, 2.0, 1.0],
                              "lstm": [4.0, 2.0, np.nan], "transformer": [10.0, 2.0, 1.0]})
    out = build_ensembles(forecasts)
    assert set(out["model"]) == {"ensemble_mean", "ensemble_median"}
    first = out[(out["ticker"] == "A") & (out["date"] == "2024-01-02")].set_index("model")["pred_var"]
    assert first["ensemble_mean"] == 4.0 and first["ensemble_median"] == 3.0       # the median ignores the outlier
    assert "B" not in set(out["ticker"])                                           # a member is missing: no combination
    assert list(ENSEMBLE_MEMBERS) == ["har_x", "lgbm", "lgbm_hybrid", "lstm", "transformer"]


# ---------------------------------------------------------------------------
# Scoring and tests
# ---------------------------------------------------------------------------
def test_score_by_segment_reports_bias_and_segments():
    predictions = pd.DataFrame({"model": "m", "horizon": 5, "ticker": ["A", "A", "B", "B"],
                                "actual_var": [1.0, 1.0, 4.0, 4.0], "pred_var": [2.0, 2.0, 4.0, 4.0]})
    overall = score_by_segment(predictions).iloc[0]
    assert overall["segment"] == "all" and overall["n"] == 4 and np.isclose(overall["bias"], 0.75)
    by_ticker = score_by_segment(predictions, predictions["ticker"]).set_index("segment")
    assert np.isclose(by_ticker.loc["A", "bias"], 0.5) and by_ticker.loc["B", "qlike"] == 0.0


def test_common_sample_keeps_only_the_scored_observations():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    predictions = pd.DataFrame({"model": "m", "horizon": 5, "ticker": "A", "date": dates, "pred_var": [1.0, 2.0]})
    common = pd.DataFrame({"horizon": [5], "ticker": ["A"], "date": dates[:1]})
    assert on_common_sample(predictions, common)["pred_var"].tolist() == [1.0]


def synthetic_daily_losses():
    rng = np.random.default_rng(0)
    parts = []
    for horizon in (1, 5):
        common = rng.normal(0, 0.2, 800)
        parts.append(pd.DataFrame({"horizon": horizon, "date": pd.bdate_range("2020-01-01", periods=800), "n": 50,
                                   "good": 1.0 + common + rng.normal(0, 0.05, 800),
                                   "also_good": 1.0 + common + rng.normal(0, 0.05, 800),
                                   "bad": 1.5 + common + rng.normal(0, 0.05, 800)}))
    return pd.concat(parts, ignore_index=True)


def test_pairwise_tests_cover_every_ordered_pair_and_are_antisymmetric():
    tests = pairwise_tests(synthetic_daily_losses(), ["good", "also_good", "bad"])
    assert len(tests) == 2 * 3 * 2
    row = tests.set_index(["horizon", "model", "reference"])
    assert row.loc[(5, "good", "bad"), "dm_stat"] < -5 and row.loc[(5, "good", "bad"), "p_value"] < 0.001
    assert np.isclose(row.loc[(5, "bad", "good"), "dm_stat"], -row.loc[(5, "good", "bad"), "dm_stat"])
    assert row.loc[(5, "good", "also_good"), "p_value"] > 0.01


def test_confidence_sets_keep_the_good_models_at_each_horizon():
    sets = confidence_sets(synthetic_daily_losses(), ["good", "also_good", "bad"]).set_index(["horizon", "model"])
    for horizon in (1, 5):
        assert sets.loc[(horizon, "good"), "in_set"] and sets.loc[(horizon, "also_good"), "in_set"]
        assert not sets.loc[(horizon, "bad"), "in_set"] and sets.loc[(horizon, "bad"), "eliminated"] == 1


# ---------------------------------------------------------------------------
# SQL builders
# ---------------------------------------------------------------------------
def test_daily_losses_query_scores_every_model_on_the_common_sample():
    sql = daily_losses_sql(["har_x", "lstm"])
    assert "AS har_x" in sql and "AS lstm" in sql
    assert "f.har_x IS NOT NULL AND f.lstm IS NOT NULL" in sql and "GROUP BY f.horizon, f.date" in sql
    assert "f.har_x IS NOT NULL AND f.lstm IS NOT NULL" in common_sample_sql(["har_x", "lstm"])
    with pytest.raises(ValueError):
        daily_losses_sql(["har_x; DROP TABLE forecasts"])


def test_replace_table_rejects_unsafe_table_names():
    with pytest.raises(ValueError):
        replace_table(None, "model_tests; DROP TABLE x", pd.DataFrame())
