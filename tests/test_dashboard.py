"""Consistency checks of the Power BI inputs (the views themselves are tested on the database)."""
import json
import re

from src.dashboard.setup_views import REQUIRED_MODELS, VIEWS, VIEWS_SQL
from src.db import PROJECT_ROOT

POWERBI = PROJECT_ROOT / "powerbi"


def test_setup_script_lists_exactly_the_views_of_the_sql_file():
    sql = VIEWS_SQL.read_text(encoding="utf-8")
    created = re.findall(r"CREATE VIEW (\w+)", sql)
    assert created == VIEWS
    dropped = re.search(r"DROP VIEW IF EXISTS (.*?) CASCADE;", sql, re.S).group(1)
    assert sorted(v.strip() for v in dropped.replace("\n", " ").split(",")) == sorted(VIEWS)


def test_forecast_paths_use_the_required_models_only():
    sql = VIEWS_SQL.read_text(encoding="utf-8")
    paths = sql[sql.index("CREATE VIEW pbi_forecast_paths"):sql.index("CREATE VIEW pbi_var_backtest")]
    used = set(re.findall(r"f\.(\w+) \* 252", paths))
    assert used == set(REQUIRED_MODELS)


def test_theme_is_valid_json_with_the_project_colours():
    theme = json.loads((POWERBI / "theme.json").read_text(encoding="utf-8"))
    assert theme["dataColors"][:2] == ["#003399", "#FFCC00"]                 # EU blue and gold


def test_measures_only_reference_tables_of_the_guide():
    guide = (POWERBI / "README.md").read_text(encoding="utf-8")
    tables = set(re.findall(r"\| `pbi_\w+` \| ([\w ]+) \|", guide))
    assert len(tables) == len(VIEWS)
    measures = (POWERBI / "measures.dax").read_text(encoding="utf-8")
    quoted = set(re.findall(r"'([\w ]+)'\[", measures))
    bare = set(re.findall(r"(?<!['\w])([A-Z]\w+)\[", measures))
    assert quoted | bare <= tables


def test_regime_view_uses_the_crisis_periods_and_all_scored_models():
    from src.dashboard.setup_views import SCORED_MODELS
    from src.viz import CRISES

    sql = VIEWS_SQL.read_text(encoding="utf-8")
    view = sql[sql.index("CREATE VIEW pbi_regime_scores"):sql.index("CREATE VIEW pbi_forecast_paths")]
    windows = re.findall(r"f\.date BETWEEN '([\d-]+)' AND '([\d-]+)'", view)
    assert windows == list(CRISES.values())
    assert re.findall(r"\('(\w+)', c\.\w+::double precision\)", view) == SCORED_MODELS
    models = sql[sql.index("CREATE VIEW pbi_models"):sql.index("CREATE VIEW pbi_horizons")]
    assert set(SCORED_MODELS) == set(re.findall(r"\('(\w+)',\s+'", models)) - {"historical_simulation"}
