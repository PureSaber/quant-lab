import json
from copy import deepcopy

import pytest

from quant_lab.research import (
    candidates,
    compare_results,
    execute_study,
    initialize_recipe,
    validate_recipe,
)


@pytest.fixture
def recipe():
    return {
        "schema_version": "quant.research-recipe/v1",
        "study_id": "test-study",
        "hypothesis": "Fixed momentum improves net returns",
        "mode": "exploratory",
        "backend": "equity",
        "inputs": {"bundle": "example"},
        "interval": {"start": "2024-01-01", "end": "2024-06-01"},
        "factors": {"momentum_20d": 1, "volatility_20d": -1},
        "strategy": {
            "family": "rank",
            "frequency": "weekly",
            "top_n": 2,
            "max_weight": 0.25,
            "cash_buffer": 0.5,
            "trend_window": 60,
        },
        "costs": {
            "initial_capital": 100000,
            "commission": 0.0003,
            "min_commission": 5,
            "stamp_tax": 0.0005,
            "slippage": 0.001,
            "participation_rate": 0.01,
        },
        "diagnostics": {"single_factors": True, "cost_multipliers": [2], "signal_delays": [1]},
        "variants": [],
    }


def test_all_candidates_registered_before_execution_and_resume(recipe, tmp_path):
    calls = []

    def executor(spec, candidate, out):
        from quant_lab.trials import TrialRegistry

        registered = TrialRegistry(tmp_path / "experiments.db").definition(spec["study_id"])
        assert len(registered["definition"]["parameters"]) == len(candidates(spec))
        calls.append(candidate["name"])
        (out / "returns.csv").write_text("return\n.01\n")
        return {"metrics": {"total_return": 0.01}, "comparison": {"currency": "CNY"}}

    first = execute_study(
        recipe,
        tmp_path,
        identity={"code": "a" * 40},
        data_identity={"snapshot": "x"},
        executor=executor,
    )
    assert first["completed"] == len(candidates(recipe))
    second = execute_study(
        recipe,
        tmp_path,
        identity={"code": "a" * 40},
        data_identity={"snapshot": "x"},
        executor=executor,
    )
    assert first == second
    assert len(calls) == len(candidates(recipe))
    assert compare_results(first["results"])["comparable"]
    path = next(tmp_path.glob("attempts/*/returns.csv"))
    path.write_text("changed")
    with pytest.raises(ValueError, match="artifact"):
        execute_study(
            recipe,
            tmp_path,
            identity={"code": "a" * 40},
            data_identity={"snapshot": "x"},
            executor=executor,
        )


def test_failure_is_preserved_and_changed_definition_is_rejected(recipe, tmp_path):
    def fail(*args):
        raise ValueError("data gap")

    report = execute_study(
        recipe, tmp_path, identity={"code": "a"}, data_identity={}, executor=fail
    )
    assert report["failed"] == len(candidates(recipe))
    assert all(e["status"] == "failed" for e in report["attempts"][1::2])
    changed = deepcopy(recipe)
    changed["hypothesis"] = "Different hypothesis"
    with pytest.raises(ValueError, match="changed"):
        execute_study(changed, tmp_path, identity={"code": "a"}, data_identity={}, executor=fail)


@pytest.mark.parametrize(
    "patch",
    [
        {"command": "arbitrary shell"},
        {"factors": {"momentum_20d": 0}},
        {"mode": "automatic"},
        {"interval": {"start": "2025-01-01", "end": "2024-01-01"}},
        {"variants": [{"name": "base"}]},
        {"diagnostics": {"signal_delays": [-1]}},
        {"backend_parameters": {"exec": "bad"}},
    ],
)
def test_closed_recipe_rejects_unsafe_or_ambiguous_requests(recipe, patch):
    with pytest.raises((ValueError, TypeError)):
        candidates({**recipe, **patch})


def test_init_never_reuses_past_holdout(recipe, tmp_path):
    import yaml

    template = tmp_path / "template.yaml"
    template.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="future"):
        initialize_recipe(
            template,
            tmp_path / "new.yaml",
            study_id="new",
            holdout_start="2020-01-01",
            holdout_end="2020-02-01",
        )
    new = initialize_recipe(template, tmp_path / "new.yaml", study_id="new")
    assert new["mode"] == "exploratory" and "holdout" not in new
    with pytest.raises(FileExistsError):
        initialize_recipe(template, tmp_path / "new.yaml", study_id="new")


def test_incompatible_comparisons_and_nonfinite_costs(recipe):
    base = {"status": "completed", "candidate": {"name": "base"}, "comparison": {"cost": 1}}
    other = {**base, "candidate": {"name": "stress"}, "comparison": {"cost": 2}}
    assert not compare_results([base, other])["comparable"]
    recipe["costs"]["slippage"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_recipe(recipe)
    json.dumps(compare_results([]), allow_nan=False)


@pytest.mark.parametrize(
    "patch",
    [
        {"risk": {"max_estimated_cost_rate": 0.01}},
        {"risk": {"drawdown_action": "liquidate"}},
        {"risk": {"max_drawdown": 0.1, "drawdown_action": "resume"}},
        {"neutralization": ["industry"]},
        {"risk_model": {"model_kind": "statistical_proxy"}},
        {"allocation": {"mode": "cost_aware"}, "risk_model": {"model_kind": "fundamental_style"}},
        {
            "allocation": {"mode": "cost_aware"},
            "risk_model": {"model_kind": "statistical_proxy", "max_tracking_error": 0.1},
        },
        {"risk": {"max_industry_weight": 0.5, "industry_field": "sector"}},
    ],
)
def test_risk_extensions_reject_unwired_or_ambiguous_configuration(recipe, patch):
    with pytest.raises(ValueError):
        validate_recipe({**recipe, **patch})


def test_risk_model_and_pit_industry_contract(recipe):
    recipe.update(
        allocation={"mode": "cost_aware"},
        required_history={"sector": "classification", "industry": "classification"},
        neutralization=["industry"],
        risk={
            "max_drawdown": 0.1,
            "drawdown_action": "liquidate",
            "max_industry_weight": 0.5,
            "industry_field": "sector",
            "max_estimated_cost_rate": 0.01,
            "estimated_cost_rate_per_turnover": 0.001,
        },
        risk_model={
            "model_kind": "statistical_proxy",
            "factor_bounds": {"market": [0, 0.9]},
            "benchmark_weights": {"A": 0.5, "B": 0.5},
            "active_factor_bounds": {"market": [-0.5, 0]},
            "max_tracking_error": 0.3,
        },
    )
    assert validate_recipe(recipe) == recipe
