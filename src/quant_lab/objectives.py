"""Ex-ante economic objectives and separate return/risk/cost/execution decisions."""

from copy import deepcopy
from math import isfinite

FIELDS = {
    "mechanism",
    "benchmark_id",
    "primary",
    "capital_range",
    "max_drawdown",
    "max_turnover",
    "max_cost_rate",
    "min_capacity",
    "data_conditions",
    "stop_conditions",
}
METRICS = {
    "net_excess_return",
    "net_return",
    "sharpe",
    "tracking_error",
    "drawdown_magnitude",
    "net_cashflow_yield",
    "capital_efficiency",
}


def _finite(value):
    return type(value) in {int, float} and isfinite(value)


def validate_objective(value):
    if not isinstance(value, dict) or set(value) != FIELDS:
        raise ValueError("objective requires the complete closed ex-ante investment contract")
    if any(
        not isinstance(value[k], str) or not value[k].strip() for k in ("mechanism", "benchmark_id")
    ):
        raise ValueError("economic mechanism and benchmark identity required")
    primary = value["primary"]
    if (
        not isinstance(primary, dict)
        or set(primary) != {"metric", "direction", "threshold"}
        or primary["metric"] not in METRICS
        or primary["direction"] not in {"at_least", "at_most"}
        or not _finite(primary["threshold"])
    ):
        raise ValueError("one explicit supported primary objective required")
    capital = value["capital_range"]
    if (
        not isinstance(capital, list)
        or len(capital) != 2
        or not all(_finite(v) for v in capital)
        or not 0 < capital[0] <= capital[1]
    ):
        raise ValueError("positive lower/upper capital range required")
    for key in ("max_drawdown", "max_turnover", "max_cost_rate", "min_capacity"):
        if not _finite(value[key]) or value[key] < 0:
            raise ValueError("nonnegative finite risk/cost/capacity budgets required")
    if value["max_drawdown"] > 1 or value["max_cost_rate"] > 1:
        raise ValueError("drawdown and cost rates use fractions, not percentage points")
    for key in ("data_conditions", "stop_conditions"):
        if (
            not isinstance(value[key], list)
            or not value[key]
            or any(not isinstance(v, str) or not v.strip() for v in value[key])
            or len(set(value[key])) != len(value[key])
        ):
            raise ValueError("distinct explicit data and stop condition IDs required")
    return deepcopy(value)


def evaluate_objective(objective, evidence):
    spec = validate_objective(objective)
    for metric in (
        "drawdown_magnitude",
        "turnover",
        "cost_rate",
        "capacity",
        "capital",
        "tracking_error",
    ):
        actual = evidence.get(metric)
        if actual is not None and (not _finite(actual) or actual < 0):
            raise ValueError(f"{metric} must be a nonnegative finite observation or absent")
    if evidence.get("benchmark_id") != spec["benchmark_id"]:
        raise ValueError("result benchmark differs from the registered objective")
    checks = []

    def check(category, metric, threshold, direction, actual):
        passed = None
        if _finite(actual):
            passed = actual >= threshold if direction == "at_least" else actual <= threshold
        checks.append(
            {
                "category": category,
                "metric": metric,
                "actual": actual,
                "threshold": threshold,
                "direction": direction,
                "passed": passed,
            }
        )

    primary = spec["primary"]
    check(
        "risk"
        if primary["metric"] in {"tracking_error", "drawdown_magnitude"}
        else "execution"
        if primary["metric"] == "capital_efficiency"
        else "return",
        primary["metric"],
        primary["threshold"],
        primary["direction"],
        evidence.get(primary["metric"]),
    )
    for category, metric, limit, direction in (
        ("risk", "drawdown_magnitude", "max_drawdown", "at_most"),
        ("cost", "turnover", "max_turnover", "at_most"),
        ("cost", "cost_rate", "max_cost_rate", "at_most"),
        ("execution", "capacity", "min_capacity", "at_least"),
    ):
        check(category, metric, spec[limit], direction, evidence.get(metric))
    check(
        "execution", "capital_lower", spec["capital_range"][0], "at_least", evidence.get("capital")
    )
    check(
        "execution", "capital_upper", spec["capital_range"][1], "at_most", evidence.get("capital")
    )
    for key in spec["data_conditions"]:
        actual = evidence.get("data_conditions", {}).get(key)
        checks.append(
            {
                "category": "data",
                "metric": key,
                "actual": actual,
                "passed": actual if type(actual) is bool else None,
            }
        )
    for key in spec["stop_conditions"]:
        actual = evidence.get("stop_conditions", {}).get(key)
        checks.append(
            {
                "category": "stop",
                "metric": key,
                "actual": actual,
                "passed": not actual if type(actual) is bool else None,
            }
        )
    status = (
        "failed"
        if any(c["passed"] is False for c in checks)
        else "insufficient_evidence"
        if any(c["passed"] is None for c in checks)
        else "meets_objective"
    )
    return {
        "status": status,
        "checks": checks,
        "scope": "research acceptance only; does not authorize investment or relaxing risk controls",
    }
