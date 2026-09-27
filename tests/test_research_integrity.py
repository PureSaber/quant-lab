import sqlite3
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from quant_lab.counterfactuals import attribution, intervention_plan
from quant_lab.invariants import (
    assert_cash_conservation,
    assert_fixed_order_cost_monotonic,
    assert_fx_consistency,
    assert_prefix_invariant,
    assert_split_wealth,
)
from quant_lab.nested import freeze_transfer, nested_selection, validate_transfer
from quant_lab.objectives import evaluate_objective, validate_objective
from quant_lab.trials import TrialRegistry


def objective():
    return {
        "mechanism": "slow information diffusion",
        "benchmark_id": "passive",
        "primary": {"metric": "net_excess_return", "direction": "at_least", "threshold": 0.01},
        "capital_range": [1000, 100000],
        "max_drawdown": 0.2,
        "max_turnover": 10,
        "max_cost_rate": 0.03,
        "min_capacity": 100000,
        "data_conditions": ["pit_complete"],
        "stop_conditions": ["structural_break"],
    }


def evidence():
    return {
        "benchmark_id": "passive",
        "net_excess_return": 0.02,
        "drawdown_magnitude": 0.1,
        "turnover": 5,
        "cost_rate": 0.01,
        "capacity": 500000,
        "capital": 10000,
        "data_conditions": {"pit_complete": True},
        "stop_conditions": {"structural_break": False},
    }


def test_ex_ante_goals_explain_different_decisions_without_changing_results():
    goal = objective()
    assert evaluate_objective(goal, evidence())["status"] == "meets_objective"
    goal["max_drawdown"] = 0.05
    failed = evaluate_objective(goal, evidence())
    assert failed["status"] == "failed"
    assert [r["category"] for r in failed["checks"] if r["passed"] is False] == ["risk"]
    assert (
        evaluate_objective(objective(), {"benchmark_id": "passive"})["status"]
        == "insufficient_evidence"
    )
    with pytest.raises(ValueError):
        evaluate_objective(goal, {**evidence(), "benchmark_id": "cash"})
    for field, value in (
        ("capital_range", [-1, 100]),
        ("primary", {}),
        ("max_drawdown", 10),
        ("stop_conditions", []),
    ):
        with pytest.raises(ValueError):
            validate_objective({**goal, field: value})


def test_family_preregistration_retains_failures_and_objectives_are_immutable(tmp_path):
    registry = TrialRegistry(tmp_path / "all-studies.db")
    basis = {
        "currency": "USD",
        "benchmark_id": "cash",
        "cost_policy": "net",
        "periods_per_year": 252,
        "sample_kind": "historical",
    }
    family = {"study_ids": ["s1", "s2"], "hypothesis": "same economic mechanism", "basis": basis}
    digest = registry.register_family("family", family)
    assert registry.register_family("family", family) == digest
    spec = {
        "hypothesis": "h",
        "parameters": [{"window": 10}],
        "code_identity": "abc",
        "selection_rule": "inner folds only",
        "family_id": "family",
        "objective": objective(),
    }
    registry.register("s1", spec)
    trial = registry.start("s1", {"window": 10})
    registry.finish(trial, "failed", {"reason": "missing terminal values"})
    registry.start("s1", {"window": 10})
    inventory = registry.family_inventory("family")
    assert inventory["attempt_count"] == 2
    assert inventory["studies"][1]["status"] == "not_registered"
    assert inventory["studies"][0]["attempts"][0]["status"] == "failed"
    changed = deepcopy(spec)
    changed["objective"]["max_drawdown"] = 0.9
    with pytest.raises(ValueError, match="changed"):
        registry.register("s1", changed)
    with pytest.raises(ValueError, match="changed"):
        registry.register_family("family", {**family, "study_ids": ["s1"]})
    with pytest.raises(ValueError, match="before"):
        registry.register_family("posthoc", family)
    with registry.connect() as db, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("UPDATE research_families SET definition='{}'")


def test_nested_outer_labels_cannot_change_its_selection_and_inner_fit_is_bounded():
    data = pd.DataFrame(
        {"label": np.sin(np.arange(80))}, index=pd.date_range("2020-01-01", periods=80)
    )
    seen = []

    def fit(frame, recipe):
        seen.append((frame.index.min(), frame.index.max()))
        return recipe["direction"]

    def score(model, frame):
        return model * frame.label.mean()

    recipes = {
        "plus": {"direction": 1, "neutralization": []},
        "minus": {"direction": -1, "neutralization": []},
    }
    kwargs = {
        "candidates": recipes,
        "fit": fit,
        "evaluate": score,
        "outer_train": 40,
        "outer_test": 10,
        "inner_train": 10,
        "inner_test": 5,
        "embargo": 2,
    }
    result = nested_selection(data, **kwargs)
    altered = data.copy()
    altered.iloc[42:52, 0] += 50
    changed = nested_selection(altered, **kwargs)
    assert result["folds"][0]["selection_sha256"] == changed["folds"][0]["selection_sha256"]
    assert result["folds"][0]["outer_score"] != changed["folds"][0]["outer_score"]
    for fold in result["folds"]:
        assert pd.Timestamp(fold["outer_train_end"]) < pd.Timestamp(fold["test_start"])
        assert all(
            pd.Timestamp(inner["train_end"]) < pd.Timestamp(inner["validation_start"])
            for inner in fold["inner_folds"]
        )


def test_transfer_recipe_is_frozen_with_market_rules_not_just_ticker_substitution():
    frozen = freeze_transfer(
        {"window": 20},
        source_market="US",
        target_market="HK",
        registered_at="2026-01-01T00:00:00Z",
        target_start="2026-02-01T00:00:00Z",
        currency_policy="HKD native vs HKD benchmark",
        cost_policy="HK fees v2",
        rule_policy="HK cash lot and settlement v2",
    )
    assert (
        validate_transfer(frozen, {"window": 20}, target_market="HK", at="2026-03-01T00:00:00Z")
        == frozen
    )
    with pytest.raises(ValueError, match="retune"):
        validate_transfer(frozen, {"window": 10}, target_market="HK", at="2026-03-01T00:00:00Z")
    with pytest.raises(ValueError, match="hash"):
        validate_transfer(
            {**frozen, "cost_policy": "free"},
            {"window": 20},
            target_market="HK",
            at="2026-03-01T00:00:00Z",
        )


def test_paired_attribution_is_preregistered_reconciled_and_not_causal():
    base = {
        "factors": {"momentum": 1},
        "cost_multiplier": 1,
        "signal_delay": 0,
        "risk": {"max_drawdown": 0.2},
        "strategy": {"frequency": "monthly"},
    }
    benchmarks = {
        "passive": {"mode": "buy_hold"},
        "same_risk_constrained": {"mode": "buy_hold", "risk": base["risk"]},
        "cash": {"mode": "cash", "interest": 0},
    }
    plan = intervention_plan(base, {"fees": 0, "delay": 1, "risk_latch": {}}, benchmarks=benchmarks)
    assert plan["variants"]["fees"]["risk"] == base["risk"]
    data = pd.DataFrame(
        {
            "base": [0.01, -0.02, 0.03, 0],
            "fees": [0.011, -0.019, 0.031, 0.001],
            "delay": [0, 0.01, -0.02, 0.03],
            "risk_latch": [0.01, -0.02, 0.03, 0],
            "passive": [0.02, 0, 0.02, 0],
            "same_risk_constrained": [0.01, 0, 0.02, 0],
            "cash": [0, 0, 0, 0],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    result = attribution(plan, data)
    assert result["return_gap"] == pytest.approx(
        sum(result["one_at_a_time_effects"].values())
        + result["unexplained_and_interaction_residual"]
    )
    with pytest.raises(ValueError, match="changed"):
        attribution({**plan, "base": {}}, data)
    with pytest.raises(ValueError, match="all base"):
        attribution(plan, data.drop(columns="fees"))


def test_financial_invariants_and_deliberate_fault_injection():
    flows = [
        {"event_id": "buy", "currency": "USD", "amount": "-100"},
        {"event_id": "dividend", "currency": "USD", "amount": "5"},
    ]
    assert_cash_conservation({"USD": "1000"}, {"USD": "905"}, flows)
    assert_split_wealth(10, 100, 20, 50)
    assert_fx_consistency(100, "7.8", 780)
    orders = [{"symbol": "A", "quantity": 10}]
    assert_fixed_order_cost_monotonic(1100, 1090, low_orders=orders, high_orders=orders)
    assert_prefix_invariant([1, 2], [1, 2])
    with pytest.raises(AssertionError, match="duplicate"):
        assert_cash_conservation({"USD": 1000}, {"USD": 910}, flows + [flows[-1]])
    with pytest.raises(AssertionError, match="FX"):
        assert_fx_consistency(100, "7.8", 100 / 7.8)
    with pytest.raises(AssertionError, match="split"):
        assert_split_wealth(10, 100, 20, 100)
    with pytest.raises(AssertionError, match="future"):
        assert_prefix_invariant([1, 2], [1, 3])
    with pytest.raises(AssertionError, match="improved"):
        assert_fixed_order_cost_monotonic(1090, 1100, low_orders=orders, high_orders=orders)
    with pytest.raises(ValueError, match="identical"):
        assert_fixed_order_cost_monotonic(1090, 1100, low_orders=orders, high_orders=[])


def test_registered_transfer_is_immutable_and_has_one_terminal_evaluation(tmp_path):
    from datetime import datetime, timezone

    from quant_lab.nested import register_transfer

    registry = TrialRegistry(tmp_path / "r.db")
    spec = {
        "hypothesis": "transfer",
        "parameters": [{"window": 20}],
        "code_identity": "fixed",
        "selection_rule": "frozen source market",
    }
    policies = {
        "source_market": "US",
        "target_market": "HK",
        "target_start": "2027-02-01T00:00:00Z",
        "currency_policy": "native HKD",
        "cost_policy": "HK fees",
        "rule_policy": "HK lot rules",
    }
    now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    register_transfer(
        registry, "t", spec, {"window": 20}, target_end="2027-03-01", now=now, **policies
    )
    with pytest.raises(ValueError, match="changed"):
        register_transfer(
            registry, "t", spec, {"window": 10}, target_end="2027-03-01", now=now, **policies
        )
    assert registry.definition("t")["definition"]["transfer_holdout"]["recipe"] == {"window": 20}
    evidence = {
        "start": "2027-02-01",
        "end": "2027-03-01",
        "code_identity": "fixed",
        "input_sha256": "a" * 64,
    }
    after = datetime(2027, 3, 2, tzinfo=timezone.utc)
    registry.seal_holdout("t", evidence, now=after)
    with pytest.raises(sqlite3.IntegrityError):
        registry.seal_holdout("t", evidence, now=after)


def test_small_n_hurdle_is_continuous_and_negative_risk_evidence_is_rejected():
    from quant_lab.selection import expected_max_sharpe

    values = [expected_max_sharpe(1, n) for n in np.linspace(1, 4, 301)]
    assert np.diff(values).min() >= 0 and np.diff(values).max() < 0.01
    for metric in ("drawdown_magnitude", "cost_rate", "capacity", "capital"):
        with pytest.raises(ValueError, match="nonnegative"):
            evaluate_objective(objective(), {**evidence(), metric: -1})
