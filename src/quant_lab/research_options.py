"""Closed, dependency-free contracts for optional research capabilities."""

from __future__ import annotations

import math
import re

from quant_lab.risk_options import validate_estimation, validate_exposure_pipeline


def validate_validation(value: dict) -> dict:
    fields = {
        "method",
        "train_sessions",
        "test_sessions",
        "embargo_sessions",
        "expanding",
        "selection_metric",
        "direction_policy",
        "direction_horizon",
        "fdr_alpha",
        "account_policy",
    }
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("Unknown validation options")
    result = {
        "method": "walk_forward",
        "embargo_sessions": 1,
        "expanding": True,
        "selection_metric": "sharpe",
        "direction_policy": "fixed",
        "direction_horizon": 1,
        "fdr_alpha": 0.05,
        "account_policy": "independent",
        **value,
    }
    if result["method"] != "walk_forward":
        raise ValueError("validation method must be walk_forward")
    if result["account_policy"] not in {"independent", "continuous"}:
        raise ValueError("account_policy must be independent or continuous")
    for key in ("train_sessions", "test_sessions", "embargo_sessions"):
        if type(result.get(key)) is not int or result[key] < (
            2 if key != "embargo_sessions" else 1
        ):
            raise ValueError(f"Invalid validation {key}")
    if type(result["expanding"]) is not bool:
        raise ValueError("expanding must be boolean")
    if result["selection_metric"] not in {"sharpe", "total_return"}:
        raise ValueError("Unknown training selection metric")
    if result["direction_policy"] not in {"fixed", "train_ic"}:
        raise ValueError("Unknown direction policy")
    if type(result["direction_horizon"]) is not int or result["direction_horizon"] not in {
        1,
        5,
        20,
    }:
        raise ValueError("direction_horizon must be 1, 5 or 20")
    if result["embargo_sessions"] < result["direction_horizon"]:
        raise ValueError("Embargo must cover the declared label horizon")
    if (
        result["direction_policy"] == "train_ic"
        and result["train_sessions"] < result["direction_horizon"] + 2
    ):
        raise ValueError("Training window cannot produce two mature direction labels")
    alpha = result["fdr_alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("Invalid FDR alpha")
    return result


def validate_options(recipe: dict) -> None:
    keys = (
        "factor_expressions",
        "validation",
        "allocation",
        "execution",
        "neutralization",
        "risk_model",
    )
    if recipe["backend"] != "equity" and any(key in recipe for key in keys):
        raise ValueError("Research extensions require the equity backend")
    expressions = recipe.get("factor_expressions", {})
    if not isinstance(expressions, dict) or len(expressions) > 32:
        raise ValueError("factor_expressions must be a mapping of at most 32 expressions")
    for name, expression in expressions.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", name)
            or not isinstance(expression, str)
            or not expression.strip()
            or len(expression) > 2048
        ):
            raise ValueError("Invalid factor expression name or text")
    if "validation" in recipe:
        validate_validation(recipe["validation"])
    if "allocation" in recipe:
        validate_allocation(recipe["allocation"])
    if "execution" in recipe:
        validate_execution(recipe["execution"], recipe.get("required_history", {}))
    if "neutralization" in recipe:
        by = recipe["neutralization"]
        if not isinstance(by, list) or not by or len(set(by)) != len(by):
            raise ValueError("neutralization requires distinct industry/market_cap fields")
        for field in by:
            domain = {"industry": "classification", "market_cap": "fundamentals"}.get(field)
            if domain is None or recipe.get("required_history", {}).get(field) != domain:
                raise ValueError("Neutralization requires explicit PIT history mappings")
    if "risk_model" in recipe:
        validate_risk_model(recipe["risk_model"], recipe.get("required_history", {}))
        if recipe.get("allocation", {}).get("mode") not in {
            "equal",
            "cost_aware",
            "inverse_vol",
            "max_sharpe",
            "max_diversification",
            "cvar",
            "evar",
            "cdar",
        }:
            raise ValueError(
                "risk_model does not support dense-only risk_parity or hrp target construction"
            )


def validate_risk(recipe):
    risk = recipe.get("risk", {})
    if risk.get("drawdown_action", "halt") not in {"halt", "liquidate"}:
        raise ValueError("drawdown_action must be halt or liquidate")
    if risk.get("exposure_breach_action", "halt") not in {"halt", "liquidate"}:
        raise ValueError("exposure_breach_action must be halt or liquidate")
    if "drawdown_action" in risk and "max_drawdown" not in risk:
        raise ValueError("drawdown_action requires max_drawdown")
    if "max_estimated_cost_rate" in risk and "estimated_cost_rate_per_turnover" not in risk:
        raise ValueError("Cost limit requires estimated_cost_rate_per_turnover")
    caps = risk.get("max_industry_weight")
    if caps is not None:
        if not _finite(caps) or not 0 <= caps <= 1:
            raise ValueError("Invalid industry weight limit")
        if recipe.get("required_history", {}).get(risk.get("industry_field")) != "classification":
            raise ValueError("Industry limits require a PIT classification field")
        if recipe.get("allocation", {}).get("mode") not in {
            "equal",
            "cost_aware",
            "inverse_vol",
            "max_sharpe",
            "max_diversification",
            "cvar",
            "evar",
            "cdar",
        }:
            raise ValueError(
                "Industry limits do not support dense-only risk_parity or hrp target construction"
            )
    elif "industry_field" in risk:
        raise ValueError("industry_field requires max_industry_weight")


def _finite(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_risk_model(value, histories):
    fields = {
        "model_kind",
        "lookback",
        "min_periods",
        "min_assets_per_period",
        "min_asset_observations",
        "covariance_shrinkage",
        "specific_variance_shrinkage",
        "annualization",
        "exposure_fields",
        "market_factor",
        "factor_bounds",
        "active_factor_bounds",
        "benchmark_weights",
        "max_tracking_error",
        "estimation",
        "exposure_pipeline",
    }
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError("Unknown risk_model options")
    result = {
        "lookback": 60,
        "min_periods": 20,
        "min_assets_per_period": 3,
        "min_asset_observations": 20,
        "covariance_shrinkage": 0.2,
        "specific_variance_shrinkage": 0.2,
        "annualization": 252,
        "exposure_fields": {},
        "market_factor": True,
        "factor_bounds": {},
        "active_factor_bounds": {},
        "benchmark_weights": {},
        "estimation": {},
        **value,
    }
    if result.get("model_kind") not in {"statistical_proxy", "fundamental_style"}:
        raise ValueError("Explicit risk model_kind required")
    for key in (
        "lookback",
        "min_periods",
        "min_assets_per_period",
        "min_asset_observations",
        "annualization",
    ):
        if type(result[key]) is not int or result[key] < 2:
            raise ValueError(f"Invalid risk_model {key}")
    if max(result["min_periods"], result["min_asset_observations"]) > result["lookback"]:
        raise ValueError("Risk model lookback is too short")
    for key in ("covariance_shrinkage", "specific_variance_shrinkage"):
        if not _finite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError("Invalid risk model shrinkage")
    mappings = result["exposure_fields"]
    if not isinstance(mappings, dict) or type(result["market_factor"]) is not bool:
        raise ValueError("Invalid risk model exposure fields")
    for factor, field in mappings.items():
        if (
            not isinstance(factor, str)
            or not factor
            or factor == "market"
            or histories.get(field) not in {"fundamentals", "classification"}
        ):
            raise ValueError("Risk exposures require PIT numeric history fields")
    pipeline = result.get("exposure_pipeline")
    if (
        not mappings
        and pipeline is None
        and (not result["market_factor"] or result["model_kind"] == "fundamental_style")
    ):
        raise ValueError("Fundamental style model requires real descriptors")
    factors = set(mappings) | ({"market"} if result["market_factor"] else set())
    validate_estimation(result["estimation"], result["lookback"])
    if pipeline is not None:
        if mappings:
            raise ValueError("exposure_fields and exposure_pipeline cannot be combined")
        factors = validate_exposure_pipeline(pipeline, histories)
        if "market_factor" in value and value["market_factor"] != pipeline["config"].get(
            "market_factor", True
        ):
            raise ValueError("market_factor differs from the exposure pipeline")
        result["market_factor"] = pipeline["config"].get("market_factor", True)
    estimation = result["estimation"]
    if (
        estimation.get("regression_weighting", "equal") != "equal"
        or estimation.get("industry_constraint", False)
        or estimation.get("specific_target", "global") == "industry"
    ) and pipeline is None:
        raise ValueError("Weighted/industry estimation requires a PIT exposure_pipeline")
    if estimation.get("industry_constraint", False) and not (
        pipeline["config"].get("market_factor", True)
        and pipeline["config"].get("include_industries", True)
    ):
        raise ValueError("Industry constraint requires market and industry factors")
    for key in ("factor_bounds", "active_factor_bounds"):
        bounds = result[key]
        if not isinstance(bounds, dict) or set(bounds) - factors:
            raise ValueError("Risk bounds reference unknown factors")
        for pair in bounds.values():
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(_finite(x) for x in pair)
                or pair[0] > pair[1]
            ):
                raise ValueError("Invalid factor exposure bounds")
    weights = result["benchmark_weights"]
    if not isinstance(weights, dict) or any(
        not isinstance(k, str) or not k or not _finite(v) or v < 0 for k, v in weights.items()
    ):
        raise ValueError("Invalid benchmark weights")
    if weights and abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("Benchmark weights must sum to one")
    if (result["active_factor_bounds"] or "max_tracking_error" in result) and not weights:
        raise ValueError("Active risk limits require benchmark_weights")
    if "max_tracking_error" in result and (
        not _finite(result["max_tracking_error"]) or result["max_tracking_error"] < 0
    ):
        raise ValueError("Invalid max_tracking_error")
    return result


def validate_allocation(value):
    from quant_data_kit.allocation_protocol import validate_research_allocation

    try:
        return validate_research_allocation(value)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def validate_execution(value, required_history):
    fields = {"mode", "universe_field", "status_fields", "max_retry_sessions"}
    if (
        not isinstance(value, dict)
        or set(value) - fields
        or value.get("mode") not in {"fixed", "dynamic"}
    ):
        raise ValueError("Unknown execution options")
    result = {"universe_field": None, "max_retry_sessions": 5, **value}
    retry = result["max_retry_sessions"]
    if type(retry) is not int or not 0 <= retry <= 252:
        raise ValueError("Invalid execution retry sessions")
    universe = result["universe_field"]
    if result["mode"] == "dynamic":
        if (
            not isinstance(universe, str)
            or not universe.strip()
            or required_history.get(universe) != "universe"
        ):
            raise ValueError("Dynamic execution requires universe history mapping")
    elif universe is not None:
        raise ValueError("Fixed execution cannot declare universe_field")
    statuses = result.get("status_fields")
    if not isinstance(statuses, dict) or set(statuses) != {
        "listed",
        "delisted",
        "tradable",
        "limit_up",
        "limit_down",
    }:
        raise ValueError("Execution requires five explicit status fields")
    for field in statuses.values():
        if (
            not isinstance(field, str)
            or not field.strip()
            or required_history.get(field) != "status"
        ):
            raise ValueError("Execution status requires status history mapping")
    if len(set(statuses.values())) != 5:
        raise ValueError("Execution status fields must be distinct")
    return result
