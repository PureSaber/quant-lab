"""Frozen research plans and paired uncertainty; engineering is a separate gate.

Detectability uses a declared block-bootstrap standard error and square-root-n
extrapolation from the calibration sample. This is a planning approximation,
not a guarantee under a changed return-generating process.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from numbers import Real
from statistics import NormalDist

import numpy as np
import pandas as pd

METRICS = {"mean_daily_excess_return", "volatility_ratio", "downside_ratio", "cvar_ratio"}
FIELDS = {
    "schema",
    "sample_kind",
    "sessions",
    "review_date",
    "calibration",
    "confidence",
    "power",
    "block_lengths",
    "repetitions",
    "seed",
    "planning_effect",
}


def _panel(rows):
    frame = pd.DataFrame(rows)
    if set(frame) != {"date", "strategy", "benchmark"} or len(frame) < 3:
        raise ValueError("paired evidence requires date, strategy, benchmark and at least 3 rows")
    dates = pd.to_datetime(frame.date, errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("paired dates must be unique and increasing")
    if any(str(d) != dts.date().isoformat() for d, dts in zip(frame.date, dates)):
        raise ValueError("paired dates require YYYY-MM-DD")
    values = frame[["strategy", "benchmark"]].to_numpy(dtype=float)
    if any(
        isinstance(v, bool) or not isinstance(v, Real)
        for row in rows
        for v in (row["strategy"], row["benchmark"])
    ):
        raise ValueError("paired returns must be numeric observations, not labels or booleans")
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("paired net simple returns must be finite and greater than -1")
    return frame.date.tolist(), values


def _stat(values, metric):
    if metric == "mean_daily_excess_return":
        return float(np.mean(values[:, 0] - values[:, 1]))
    if metric == "volatility_ratio":
        risk = np.std(values, axis=0, ddof=1)
    elif metric == "downside_ratio":
        risk = np.sqrt(np.mean(np.minimum(values, 0) ** 2, axis=0))
    elif metric == "cvar_ratio":
        # Mean loss in the worst ceil(5% * n) observations; same convention in every replicate.
        risk = -np.sort(values, axis=0)[: max(1, math.ceil(len(values) * 0.05))].mean(axis=0)
    else:
        raise ValueError("unsupported paired metric")
    if risk[1] <= 0 or risk[0] < 0:
        raise ValueError("risk ratio has no positive benchmark risk")
    return float(risk[0] / risk[1])


def _bootstrap(values, metric, block, repetitions, seed):
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, n, size=math.ceil(n / block))
        indices = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        # One common set of indices preserves strategy/benchmark pairing.
        draws.append(_stat(values[indices], metric))
    return np.asarray(draws)


def validate_plan(plan, objective, *, registered_at=None):
    if not isinstance(plan, dict) or set(plan) != FIELDS:
        raise ValueError("research plan requires the complete closed v1 contract")
    if plan["schema"] != "quant.research-gate-plan/v1":
        raise ValueError("unsupported research plan schema")
    if plan["sample_kind"] not in {"retrospective", "prospective"}:
        raise ValueError("declare retrospective or prospective sample use")
    primary = objective["primary"]
    if primary["metric"] not in METRICS:
        raise ValueError("research gate requires an explicit paired return or risk metric")
    direction = "at_least" if primary["metric"] == "mean_daily_excess_return" else "at_most"
    null = 0 if direction == "at_least" else 1
    if primary["direction"] != direction or (
        primary["threshold"] <= 0 if null == 0 else not 0 < primary["threshold"] < 1
    ):
        raise ValueError("objective must declare a positive effect against its null")
    expected = plan["planning_effect"]
    if (
        type(expected) not in {int, float}
        or not math.isfinite(expected)
        or (
            expected <= primary["threshold"]
            if direction == "at_least"
            else not 0 <= expected < primary["threshold"]
        )
    ):
        raise ValueError(
            "planning effect must lie beyond the objective boundary in the desired direction"
        )
    sessions = plan["sessions"]
    if not isinstance(sessions, list) or len(sessions) < 3:
        raise ValueError("freeze the expected observation sessions")
    for day in [*sessions, plan["review_date"]]:
        if not isinstance(day, str) or pd.Timestamp(day).date().isoformat() != day:
            raise ValueError("plan dates require YYYY-MM-DD")
    if sessions != sorted(set(sessions)) or plan["review_date"] < sessions[-1]:
        raise ValueError("session order and scheduled review date are invalid")
    for key in ("confidence", "power"):
        if type(plan[key]) not in {float, int} or not 0.5 < plan[key] < 1:
            raise ValueError(f"{key} must be between .5 and 1")
    if type(plan["repetitions"]) is not int or not 199 <= plan["repetitions"] <= 10000:
        raise ValueError("bootstrap repetitions must be between 199 and 10000")
    if type(plan["seed"]) is not int or plan["seed"] < 0:
        raise ValueError("a nonnegative fixed random seed is required")
    calibration_dates, values = _panel(plan["calibration"])
    blocks = plan["block_lengths"]
    if (
        not isinstance(blocks, list)
        or not blocks
        or len(blocks) != len(set(blocks))
        or any(
            type(b) is not int or b < 1 or b > min(len(values), len(sessions)) // 3 for b in blocks
        )
    ):
        raise ValueError("distinct block lengths require at least three blocks in both samples")
    if registered_at is not None and plan["sample_kind"] == "prospective":
        at = pd.Timestamp(registered_at)
        if at.tzinfo is None:
            raise ValueError("registration must have a timezone")
        local_date = at.tz_convert("Asia/Shanghai").date().isoformat()
        if sessions[0] <= local_date or calibration_dates[-1] >= sessions[0]:
            raise ValueError("future sessions must follow registration and calibration")
        if calibration_dates[-1] >= local_date:
            raise ValueError(
                "daily calibration must precede the registration day; intraday availability is unknown"
            )
    _stat(values, primary["metric"])
    return plan


def detectability(plan, objective):
    validate_plan(plan, objective)
    _, values = _panel(plan["calibration"])
    metric = objective["primary"]["metric"]
    effect = abs(plan["planning_effect"] - objective["primary"]["threshold"])
    quantile = NormalDist().inv_cdf((1 + plan["confidence"]) / 2) + NormalDist().inv_cdf(
        plan["power"]
    )
    reports = []
    for block in plan["block_lengths"]:
        draws = _bootstrap(values, metric, block, plan["repetitions"], plan["seed"])
        error = float(draws.std(ddof=1))
        if error <= 1e-15:
            raise ValueError("degenerate calibration cannot establish detectability")
        required = max(3 * block, math.ceil(len(values) * (quantile * error / effect) ** 2))
        reports.append(
            {"block_length": block, "standard_error": error, "required_sessions": required}
        )
    return {
        "method": "paired circular block bootstrap; square-root-n planning approximation",
        "minimum_detectable_effect": effect,
        "effect_definition": "margin between declared planning alternative and objective boundary",
        "planning_effect": plan["planning_effect"],
        "objective_boundary": objective["primary"]["threshold"],
        "required_sessions": max(r["required_sessions"] for r in reports),
        "sensitivity": reports,
        "limitations": "calibration stationarity assumed; no universal fixed sample threshold",
    }


def evaluate_gates(
    objective, point_review, *, plan=None, registered_at=None, evidence=None, as_of=None
):
    """Evaluate B from frozen dates, paired returns and the original registration.

    A is deliberately owned by native operations. C needs A, B, quote calibration
    and the existing independent-window requirements; this function cannot grant it.
    """
    result = {
        "gate_a": {"status": "external_native_evidence_required"},
        "gate_b": {"status": "not_preregistered"},
        "gate_c": {
            "status": "not_authorized",
            "requires": [
                "native_gate_a",
                "gate_b",
                "real_quote_cost_calibration",
                "existing_model_promotion_rules",
            ],
        },
        "automatic_promotion": False,
    }
    if plan is None:
        return result
    if registered_at is None:
        raise ValueError("gate review requires the immutable registry registration time")
    validate_plan(plan, objective, registered_at=registered_at)
    b = result["gate_b"]
    b["plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if plan["sample_kind"] != "prospective":
        b.update(
            status="retrospective_only",
            reason="historical observations cannot confirm a future objective",
        )
        return result
    calibration = detectability(plan, objective)
    b["detectability"] = calibration
    evidence = evidence or {}
    if evidence.get("evidence_status") != "validated":
        b.update(status="insufficient_evidence", reason="provisional_or_unknown_evidence")
        return result
    clock = pd.Timestamp(as_of or datetime.now(timezone.utc))
    clock = clock.tz_localize("UTC") if clock.tzinfo is None else clock
    today = clock.tz_convert("Asia/Shanghai").date().isoformat()
    last_close = pd.Timestamp(plan["sessions"][-1], tz="Asia/Shanghai") + pd.Timedelta(hours=16)
    if today < plan["review_date"] or clock < last_close or not evidence.get("paired_returns"):
        b.update(status="insufficient_evidence", reason="review_not_due_or_paired_returns_missing")
        return result
    dates, values = _panel(evidence["paired_returns"])
    if dates != plan["sessions"] or dates[-1] > today:
        b.update(status="insufficient_evidence", reason="frozen_session_coverage_mismatch")
        return result
    if len(dates) < calibration["required_sessions"]:
        b.update(status="insufficient_evidence", reason="window_below_declared_detectability")
        return result
    metric = objective["primary"]["metric"]
    estimate = _stat(values, metric)
    actual = evidence.get(metric)
    if type(actual) not in {float, int} or not math.isclose(
        actual, estimate, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("objective point estimate differs from the paired net-return evidence")
    intervals = []
    tail = (1 - plan["confidence"]) / 2
    for block in plan["block_lengths"]:
        draws = _bootstrap(values, metric, block, plan["repetitions"], plan["seed"])
        lower, upper = np.quantile(draws, [tail, 1 - tail])
        intervals.append({"block_length": block, "lower": float(lower), "upper": float(upper)})
    primary = objective["primary"]
    meets = all(
        r["lower"] >= primary["threshold"]
        if primary["direction"] == "at_least"
        else r["upper"] <= primary["threshold"]
        for r in intervals
    )
    rejects = all(
        r["upper"] < primary["threshold"]
        if primary["direction"] == "at_least"
        else r["lower"] > primary["threshold"]
        for r in intervals
    )
    status = (
        "failed"
        if point_review["status"] == "failed" or rejects
        else (
            "meets_objective"
            if meets and point_review["status"] == "meets_objective"
            else "insufficient_evidence"
        )
    )
    b.update(status=status, estimate=estimate, intervals=intervals, observations=len(dates))
    return result
