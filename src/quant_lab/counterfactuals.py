"""Preregistered one-factor interventions and exact additive return-gap accounting."""

from copy import deepcopy

import numpy as np

from quant_lab.research import digest
from quant_lab.selection import matrix

# The entire subtree is one economic intervention, not an arbitrary recipe patch.
PATHS = {
    "signal": ("factors",),
    "allocation": ("allocation",),
    "risk_latch": ("risk",),
    "frequency": ("strategy", "frequency"),
    "fees": ("cost_multiplier",),
    "delay": ("signal_delay",),
}
BENCHMARKS = {"passive", "same_risk_constrained", "cash"}


def intervention_plan(base, changes, *, benchmarks):
    if not changes or set(changes) - set(PATHS) or set(benchmarks) != BENCHMARKS:
        raise ValueError(
            "supported interventions and three distinct benchmark definitions required"
        )
    if any(not isinstance(v, dict) or not v for v in benchmarks.values()):
        raise ValueError("benchmarks require explicit executable definitions")
    if len({digest(v) for v in benchmarks.values()}) != 3:
        raise ValueError("passive, same-risk and cash benchmarks must be separately defined")
    variants = {}
    for dimension, replacement in changes.items():
        variant = deepcopy(base)
        target = variant
        path = PATHS[dimension]
        for part in path[:-1]:
            target = target[part]
        if target.get(path[-1]) == replacement:
            raise ValueError("intervention must change its declared dimension")
        target[path[-1]] = deepcopy(replacement)
        variants[dimension] = variant
    plan = {
        "schema": "quant.paired-interventions/v1",
        "base": deepcopy(base),
        "variants": variants,
        "benchmarks": deepcopy(benchmarks),
    }
    return {**plan, "sha256": digest(plan)}


def validate_plan(plan):
    if digest({k: v for k, v in plan.items() if k != "sha256"}) != plan.get("sha256"):
        raise ValueError("preregistered intervention plan was changed")
    if set(plan.get("variants", {})) - set(PATHS):
        raise ValueError("unknown intervention dimension")
    changes = {}
    for dimension, variant in plan["variants"].items():
        value = variant
        for field in PATHS[dimension]:
            value = value[field]
        changes[dimension] = value
    if intervention_plan(plan["base"], changes, benchmarks=plan["benchmarks"]) != plan:
        raise ValueError("each intervention must change only one declared dimension")
    return plan


def attribution(plan, net_returns, *, reference="passive"):
    validate_plan(plan)
    if reference not in BENCHMARKS:
        raise ValueError("choose an explicitly defined benchmark")
    data = matrix(net_returns)
    expected = {"base", *plan["variants"], *BENCHMARKS}
    if set(data) != expected or (data < -1).any().any():
        raise ValueError("all base, intervention and benchmark net returns must be present")
    terminal = (1 + data).prod() - 1
    effects = {name: float(terminal[name] - terminal["base"]) for name in plan["variants"]}
    gap = float(terminal[reference] - terminal["base"])
    residual = gap - sum(effects.values())
    # A residual is NOT an identified interaction effect or causal attribution.
    if not np.isclose(sum(effects.values()) + residual, gap, atol=1e-12):
        raise ArithmeticError("counterfactual accounting did not reconcile")
    return {
        "plan_sha256": plan["sha256"],
        "reference": reference,
        "return_gap": gap,
        "one_at_a_time_effects": effects,
        "unexplained_and_interaction_residual": residual,
        "total_returns": terminal.to_dict(),
        "sessions": len(data),
        "definition": "each effect is intervention terminal return minus baseline terminal return",
        "limitations": [
            "model-dependent counterfactuals, not causal identification",
            "residual includes omitted effects and interactions",
            "better unconstrained backtest does not authorize weaker live risk limits",
        ],
    }
