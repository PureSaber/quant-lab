"""Dependency-free recipe contracts for advanced factor-risk research."""

from __future__ import annotations

import math

ESTIMATION_FIELDS = {
    "regression_weighting",
    "industry_constraint",
    "factor_half_life",
    "specific_half_life",
    "newey_west_lags",
    "specific_newey_west_lags",
    "risk_horizon",
    "regime_window",
    "regime_min_periods",
    "regime_half_life",
    "regime_scale_min",
    "regime_scale_max",
    "specific_target",
    "specific_shrinkage_method",
    "specific_prior_dof",
    "specific_min_group_assets",
    "specific_group_fallback",
    "max_condition_number",
}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_estimation(value, lookback):
    if not isinstance(value, dict) or set(value) - ESTIMATION_FIELDS:
        raise ValueError("Unknown risk estimation options")
    choices = {
        "regression_weighting": {"equal", "market_cap", "sqrt_market_cap"},
        "specific_target": {"global", "industry"},
        "specific_shrinkage_method": {"fixed", "bayesian"},
        "specific_group_fallback": {"error", "global"},
    }
    for name, allowed in choices.items():
        if name in value and value[name] not in allowed:
            raise ValueError(f"Invalid risk estimation {name}")
    if type(value.get("industry_constraint", False)) is not bool:
        raise ValueError("industry_constraint must be boolean")
    for name in ("factor_half_life", "specific_half_life", "regime_half_life"):
        if value.get(name) is not None and (not _number(value[name]) or value[name] <= 0):
            raise ValueError(f"Invalid risk estimation {name}")
    for name, minimum in (
        ("newey_west_lags", 0),
        ("specific_newey_west_lags", 0),
        ("risk_horizon", 1),
        ("regime_window", 0),
        ("regime_min_periods", 2),
        ("specific_min_group_assets", 2),
    ):
        if name in value and (type(value[name]) is not int or value[name] < minimum):
            raise ValueError(f"Invalid risk estimation {name}")
    horizon = value.get("risk_horizon", 1)
    for name in ("newey_west_lags", "specific_newey_west_lags"):
        lag = value.get(name, 0)
        if lag not in {0, horizon - 1} or lag >= lookback:
            raise ValueError("Newey-West lags must match the horizon and fit the lookback")
    window = value.get("regime_window", 0)
    if window == 1 or (window and window + value.get("regime_min_periods", 20) > lookback):
        raise ValueError("Regime adjustment needs sufficient prior history and window")
    for name in (
        "regime_scale_min",
        "regime_scale_max",
        "specific_prior_dof",
        "max_condition_number",
    ):
        if name in value and (not _number(value[name]) or value[name] <= 0):
            raise ValueError(f"Invalid risk estimation {name}")
    if value.get("max_condition_number", 1e8) <= 1:
        raise ValueError("max_condition_number must exceed one")
    if value.get("regime_scale_min", 0.25) > 1 or value.get("regime_scale_max", 4) < 1:
        raise ValueError("Regime bounds must contain one")
    return value


def validate_exposure_pipeline(value, histories):
    if not isinstance(value, dict) or set(value) != {
        "config",
        "descriptor_fields",
        "market_cap_field",
        "industry_field",
    }:
        raise ValueError("Complete exposure_pipeline contract required")
    config = value["config"]
    allowed = {
        "styles",
        "industry_universe",
        "market_factor",
        "include_industries",
        "standardization_weighting",
        "winsor_method",
        "winsor_limit",
        "missing_policy",
        "max_missing_fraction",
        "min_industry_observations",
        "missing_group_fallback",
        "min_assets",
        "max_descriptor_age_days",
        "max_classification_age_days",
        "orthogonalize",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown exposure pipeline config")
    styles = config.get("styles")
    if not isinstance(styles, dict) or not styles:
        raise ValueError("Exposure pipeline requires explicit styles")
    descriptors = set()
    for name, composite in styles.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or name == "market"
            or name.startswith("industry:")
            or not isinstance(composite, dict)
            or not composite
        ):
            raise ValueError("Invalid exposure style composite")
        if (
            any(
                not isinstance(key, str) or not key.strip() or not _number(weight)
                for key, weight in composite.items()
            )
            or sum(abs(w) for w in composite.values()) == 0
        ):
            raise ValueError("Invalid exposure descriptor weights")
        descriptors.update(key for key, weight in composite.items() if weight)
    fields = value["descriptor_fields"]
    if not isinstance(fields, dict) or set(fields) != descriptors:
        raise ValueError("descriptor_fields must exactly cover active composite descriptors")
    for field in [value["market_cap_field"], *fields.values()]:
        if not isinstance(field, str) or histories.get(field) != "fundamentals":
            raise ValueError("Exposure descriptors and market cap require PIT numeric histories")
    if (
        not isinstance(value["industry_field"], str)
        or histories.get(value["industry_field"]) != "classification"
    ):
        raise ValueError("Exposure industries require PIT classification history")
    industries = config.get("industry_universe")
    if (
        not isinstance(industries, (list, tuple))
        or not industries
        or any(not isinstance(x, str) or not x.strip() for x in industries)
        or len(set(industries)) != len(industries)
    ):
        raise ValueError("A frozen industry_universe is required")
    for key in ("market_factor", "include_industries"):
        if type(config.get(key, True)) is not bool:
            raise ValueError(f"Invalid exposure {key}")
    for key, default, choices in (
        ("standardization_weighting", "market_cap", {"equal", "market_cap", "sqrt_market_cap"}),
        ("winsor_method", "mad", {"mad", "quantile", "none"}),
        ("missing_policy", "error", {"error", "industry_median"}),
        ("missing_group_fallback", "error", {"error", "global"}),
    ):
        if config.get(key, default) not in choices:
            raise ValueError(f"Invalid exposure {key}")
    limit = config.get("winsor_limit", 5.0)
    method = config.get("winsor_method", "mad")
    if (
        not _number(limit)
        or (method == "mad" and limit <= 0)
        or (method == "quantile" and not 0 < limit < 0.5)
    ):
        raise ValueError("Invalid exposure winsor_limit")
    missing = config.get("max_missing_fraction", 0.2)
    if not _number(missing) or not 0 <= missing < 1:
        raise ValueError("Invalid exposure missing fraction")
    for key in ("min_industry_observations", "min_assets"):
        if key in config and (type(config[key]) is not int or config[key] < 2):
            raise ValueError(f"Invalid exposure {key}")
    for key in ("max_descriptor_age_days", "max_classification_age_days"):
        if config.get(key) is not None and (not _number(config[key]) or config[key] < 0):
            raise ValueError(f"Invalid exposure {key}")
    graph = config.get("orthogonalize", {})
    if not isinstance(graph, dict) or set(graph) - set(styles):
        raise ValueError("Invalid exposure orthogonalization graph")
    visited, active = set(), set()

    def visit(name):
        if name in active:
            raise ValueError("Exposure orthogonalization cycle")
        if name in visited:
            return
        parents = graph.get(name, [])
        if (
            not isinstance(parents, (list, tuple))
            or any(p not in styles for p in parents)
            or len(set(parents)) != len(parents)
        ):
            raise ValueError("Invalid orthogonalization controls")
        active.add(name)
        for parent in parents:
            visit(parent)
        active.remove(name)
        visited.add(name)

    for name in styles:
        visit(name)
    factors = set(styles)
    if config.get("market_factor", True):
        factors.add("market")
    if config.get("include_industries", True):
        factors.update("industry:" + name for name in industries)
    return factors
