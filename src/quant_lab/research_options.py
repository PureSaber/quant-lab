"""Closed, dependency-free contracts for optional research capabilities."""

from __future__ import annotations

import math
import re


def validate_validation(value: dict) -> dict:
    fields = {
        "method", "train_sessions", "test_sessions", "embargo_sessions", "expanding",
        "selection_metric", "direction_policy", "direction_horizon", "fdr_alpha",
    }
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("Unknown validation options")
    result = {
        "method": "walk_forward", "embargo_sessions": 1, "expanding": True,
        "selection_metric": "sharpe", "direction_policy": "fixed",
        "direction_horizon": 1, "fdr_alpha": 0.05, **value,
    }
    if result["method"] != "walk_forward":
        raise ValueError("validation method must be walk_forward")
    for key in ("train_sessions", "test_sessions", "embargo_sessions"):
        if type(result.get(key)) is not int or result[key] < (2 if key != "embargo_sessions" else 1):
            raise ValueError(f"Invalid validation {key}")
    if type(result["expanding"]) is not bool:
        raise ValueError("expanding must be boolean")
    if result["selection_metric"] not in {"sharpe", "total_return"}:
        raise ValueError("Unknown training selection metric")
    if result["direction_policy"] not in {"fixed", "train_ic"}:
        raise ValueError("Unknown direction policy")
    if type(result["direction_horizon"]) is not int or result["direction_horizon"] not in {1, 5, 20}:
        raise ValueError("direction_horizon must be 1, 5 or 20")
    if result["embargo_sessions"] < result["direction_horizon"]:
        raise ValueError("Embargo must cover the declared label horizon")
    if result["direction_policy"] == "train_ic" and result["train_sessions"] < result["direction_horizon"] + 2:
        raise ValueError("Training window cannot produce two mature direction labels")
    alpha = result["fdr_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("Invalid FDR alpha")
    return result


def validate_options(recipe: dict) -> None:
    keys = ("factor_expressions", "validation", "allocation", "execution")
    if recipe["backend"] != "equity" and any(key in recipe for key in keys):
        raise ValueError("Research extensions require the equity backend")
    expressions = recipe.get("factor_expressions", {})
    if not isinstance(expressions, dict) or len(expressions) > 32:
        raise ValueError("factor_expressions must be a mapping of at most 32 expressions")
    for name, expression in expressions.items():
        if (
            not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", name)
            or not isinstance(expression, str) or not expression.strip() or len(expression) > 2048
        ):
            raise ValueError("Invalid factor expression name or text")
    if "validation" in recipe:
        validate_validation(recipe["validation"])
    if "allocation" in recipe:
        validate_allocation(recipe["allocation"])
    if "execution" in recipe:
        validate_execution(recipe["execution"], recipe.get("required_history", {}))


def validate_allocation(value):
    fields = {"mode", "lookback", "min_observations", "covariance_shrinkage", "risk_aversion",
              "turnover_penalty", "max_turnover"}
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("Unknown allocation options")
    result = {"lookback": 20, "min_observations": 10, "covariance_shrinkage": 0.2,
              "risk_aversion": 5.0, "turnover_penalty": 0.0, "max_turnover": 1.0, **value}
    if result.get("mode") not in {"equal", "inverse_vol", "cost_aware"}:
        raise ValueError("Unknown allocation mode")
    if any(type(result[k]) is not int or result[k] < 2 for k in ("lookback", "min_observations")):
        raise ValueError("Invalid allocation observations")
    if result["min_observations"] > result["lookback"]:
        raise ValueError("Allocation min_observations exceeds lookback")
    for key in ("covariance_shrinkage", "risk_aversion", "turnover_penalty", "max_turnover"):
        number = result[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError("Allocation values must be finite numbers")
        if number < 0:
            raise ValueError("Allocation values cannot be negative")
    if result["covariance_shrinkage"] > 1 or result["risk_aversion"] == 0 or result["max_turnover"] > 2:
        raise ValueError("Allocation constraint out of range")
    return result


def validate_execution(value, required_history):
    fields = {"mode", "universe_field", "status_fields", "max_retry_sessions"}
    if not isinstance(value, dict) or set(value) - fields or value.get("mode") not in {"fixed", "dynamic"}:
        raise ValueError("Unknown execution options")
    result = {"universe_field": None, "max_retry_sessions": 5, **value}
    retry = result["max_retry_sessions"]
    if type(retry) is not int or not 0 <= retry <= 252:
        raise ValueError("Invalid execution retry sessions")
    universe = result["universe_field"]
    if result["mode"] == "dynamic":
        if not isinstance(universe, str) or not universe.strip() or required_history.get(universe) != "universe":
            raise ValueError("Dynamic execution requires universe history mapping")
    elif universe is not None:
        raise ValueError("Fixed execution cannot declare universe_field")
    statuses = result.get("status_fields")
    if not isinstance(statuses, dict) or set(statuses) != {
        "listed", "delisted", "tradable", "limit_up", "limit_down"
    }:
        raise ValueError("Execution requires five explicit status fields")
    for field in statuses.values():
        if not isinstance(field, str) or not field.strip() or required_history.get(field) != "status":
            raise ValueError("Execution status requires status history mapping")
    if len(set(statuses.values())) != 5:
        raise ValueError("Execution status fields must be distinct")
    return result
