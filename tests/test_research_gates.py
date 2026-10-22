from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from test_research_integrity import evidence, objective

from quant_lab.objectives import evaluate_objective
from quant_lab.research_gates import _stat, detectability, evaluate_gates, validate_plan
from quant_lab.trials import TrialRegistry


def setup_plan():
    rng = np.random.default_rng(22)
    benchmark = rng.normal(0, 0.01, 180)
    strategy = 0.5 * benchmark + rng.normal(0, 0.0005, 180)
    rows = lambda dates: [
        {"date": str(d.date()), "strategy": float(s), "benchmark": float(b)}
        for d, s, b in zip(dates, strategy, benchmark)
    ]
    goal = objective()
    goal["primary"] = {"metric": "volatility_ratio", "direction": "at_most", "threshold": 0.8}
    dates = pd.bdate_range("2027-01-01", periods=180)
    plan = {
        "schema": "quant.research-gate-plan/v1",
        "sample_kind": "prospective",
        "sessions": [str(d.date()) for d in dates],
        "review_date": str(dates[-1].date()),
        "calibration": rows(pd.bdate_range("2025-01-01", periods=180)),
        "confidence": 0.95,
        "power": 0.8,
        "planning_effect": 0.6,
        "block_lengths": [3, 9],
        "repetitions": 199,
        "seed": 22,
    }
    ev = {
        **evidence(),
        "volatility_ratio": _stat(np.column_stack([strategy, benchmark]), "volatility_ratio"),
        "paired_returns": rows(dates),
        "evidence_status": "validated",
    }
    return goal, plan, ev


def review(goal, plan, ev):
    return evaluate_gates(
        goal,
        evaluate_objective(goal, ev),
        plan=plan,
        registered_at="2026-10-22T00:00:00Z",
        evidence=ev,
        as_of="2028-01-01",
    )


def test_point_estimate_never_implies_a_b_or_c():
    goal, _, ev = setup_plan()
    point = evaluate_objective(goal, ev)
    assert point["evidence_kind"] == "point_estimate_only"
    gates = evaluate_gates(goal, point)
    assert gates["gate_b"]["status"] == "not_preregistered"
    assert gates["gate_c"]["status"] == "not_authorized"


def test_pairing_and_power_with_independent_noise():
    goal, plan, ev = setup_plan()
    gates = review(goal, plan, ev)
    assert gates["gate_b"]["status"] == "meets_objective"
    assert gates["gate_b"]["intervals"][0]["upper"] < 0.8
    assert not gates["automatic_promotion"]
    plan["calibration"][0]["strategy"] = 0.9
    assert detectability(plan, goal)["required_sessions"] > len(plan["sessions"])
    assert review(goal, plan, ev)["gate_b"]["reason"] == "window_below_declared_detectability"


@pytest.mark.parametrize("status", ["provisional", "unknown", None])
def test_provisional_unknown_and_unlabelled_cannot_pass(status):
    goal, plan, ev = setup_plan()
    ev["evidence_status"] = status
    assert review(goal, plan, ev)["gate_b"]["status"] == "insufficient_evidence"


def test_missing_rows_tampered_points_and_retrospective_samples():
    goal, plan, ev = setup_plan()
    damaged = deepcopy(ev)
    damaged["paired_returns"].pop(10)
    assert review(goal, plan, damaged)["gate_b"]["reason"] == "frozen_session_coverage_mismatch"
    with pytest.raises(ValueError, match="point estimate"):
        review(goal, plan, {**ev, "volatility_ratio": 0.1})
    plan["sample_kind"] = "retrospective"
    assert review(goal, plan, ev)["gate_b"]["status"] == "retrospective_only"


def test_registration_is_frozen_and_uses_actual_registry_time(tmp_path):
    goal, plan, _ = setup_plan()
    registry = TrialRegistry(tmp_path / "trials.db")
    definition = {
        "objective": goal,
        "research_plan": plan,
        "hypothesis": "risk reduction",
        "parameters": [{"x": 1}],
        "selection_rule": "report all",
        "code_identity": "abc",
    }
    at = datetime(2026, 10, 22, tzinfo=timezone.utc)
    digest = registry.register("one", definition, now=at)
    assert (
        registry.register("one", definition, now=datetime(2028, 1, 1, tzinfo=timezone.utc))
        == digest
    )
    with pytest.raises(ValueError, match="future sessions"):
        registry.register("late", definition, now=datetime(2028, 1, 1, tzinfo=timezone.utc))
    changed = deepcopy(definition)
    changed["research_plan"]["power"] = 0.9
    with pytest.raises(ValueError, match="definition changed"):
        registry.register("one", changed, now=at)


def test_zero_uncertainty_invalid_dates_and_boolean_settings():
    goal, plan, _ = setup_plan()
    plan["power"] = True
    with pytest.raises(ValueError):
        validate_plan(plan, goal)
    plan["power"] = 0.8
    plan["calibration"] = [{**r, "strategy": r["benchmark"] * 0.5} for r in plan["calibration"]]
    with pytest.raises(ValueError, match="degenerate"):
        detectability(plan, goal)
