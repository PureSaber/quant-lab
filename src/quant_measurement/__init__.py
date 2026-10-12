"""Stdlib-only, fail-closed economic measurement declarations (no conversions).

Declarations describe evidence; validation never certifies truth, investability,
or immutable artifact integrity. Consumers must verify their source separately.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import pairwise

SCHEMA = "quant.measurement/v1"
BASIS_FIELDS = frozenset(
    [
        "currency",
        "monetary_unit",
        "return_unit",
        "return_basis",
        "period_start",
        "period_end",
        "frequency",
        "calendar",
        "timezone",
        "annualization_periods",
        "risk_free_rate",
        "flow_policy",
        "cost_policy",
        "valuation_policy",
        "benchmark_id",
        "universe_id",
        "timing_policy",
        "denominator_policy",
        "observation_schedule_id",
        "capital_basis",
        "initial_capital",
        "risk_free_unit",
        "risk_free_compounding",
        "annualization_policy",
        "cost_policy_id",
        "valuation_policy_id",
    ]
)
FIELDS = {"schema", "evidence_kind", "sample_kind", "basis", "provenance", "unsupported_scopes"}
EVIDENCE_KINDS = {"synthetic", "retrospective", "historical_pit", "prospective_observed"}
SAMPLE_KINDS = {"deterministic_fixture", "in_sample", "out_of_sample", "prospective"}


def content_id(value) -> str:
    """SHA256 of UTF-8 sorted-key, compact JSON (ensure_ascii=True, no NaN)."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_observations(measurement, timestamps):
    """Validate the ordered grid including the opening boundary; never align it."""
    value = validate_measurement(measurement)
    grid = list(timestamps)
    basis = value["basis"]
    if len(grid) < 2 or grid[0] != basis["period_start"] or grid[-1] != basis["period_end"]:
        raise ValueError("observation grid must include the exact opening and closing boundaries")
    parsed = [datetime.fromisoformat(token.replace("Z", "+00:00")) for token in grid]
    if any(not token.endswith("Z") for token in grid) or any(
        right <= left for left, right in pairwise(parsed)
    ):
        raise ValueError("observation grid must be strictly increasing UTC timestamps")
    if content_id(grid) != basis["observation_schedule_id"]:
        raise ValueError("observation grid differs from measurement declaration")
    return grid


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _closed(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"{name}: exact fields required: {', '.join(sorted(fields))}")


def _decimal(value, name):
    try:
        result = Decimal(str(value))
        if isinstance(value, bool) or not result.is_finite():
            raise ValueError(f"{name}: finite number required")
        return result
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{name}: finite number required") from exc


def validate_measurement(value: dict) -> dict:
    """Return a detached validated declaration; reject incomplete/unknown schema."""
    _closed(value, FIELDS, "measurement")
    if value["schema"] != SCHEMA:
        raise ValueError("schema: unsupported measurement version")
    if value["evidence_kind"] not in EVIDENCE_KINDS:
        raise ValueError("evidence_kind: explicit supported evidence kind required")
    if value["sample_kind"] not in SAMPLE_KINDS:
        raise ValueError("sample_kind: explicit supported sample kind required")
    basis = value["basis"]
    _closed(basis, BASIS_FIELDS, "basis")
    for key in sorted(BASIS_FIELDS - {"annualization_periods"}):
        if not _text(basis[key]) or basis[key].lower() in {"unknown", "n/a", "unspecified"}:
            raise ValueError(f"basis.{key}: explicit nonempty string required")
    if not re.fullmatch(r"[A-Z]{3}", basis["currency"]):
        raise ValueError("basis.currency: uppercase three-letter currency required")
    if basis["return_unit"] not in {"decimal", "percent"}:
        raise ValueError("basis.return_unit: decimal or percent required")
    if basis["monetary_unit"] not in {"major_currency", "minor_currency"}:
        raise ValueError("basis.monetary_unit: major_currency or minor_currency required")
    if basis["annualization_policy"] == "not_annualized":
        if basis["annualization_periods"] is not None:
            raise ValueError("basis.annualization_periods: must be null when not annualized")
    elif basis["annualization_policy"] == "fixed_periods":
        if basis["frequency"] == "event":
            raise ValueError("basis.frequency: event observations cannot use fixed annualization")
        if (
            type(basis["annualization_periods"]) not in {int, float}
            or _decimal(basis["annualization_periods"], "basis.annualization_periods") <= 0
        ):
            raise ValueError("basis.annualization_periods: positive finite number required")
    else:
        raise ValueError("basis.annualization_policy: not_annualized or fixed_periods required")
    _decimal(basis["risk_free_rate"], "basis.risk_free_rate")
    if _decimal(basis["initial_capital"], "basis.initial_capital") <= 0:
        raise ValueError("basis.initial_capital: positive capital required")
    if basis["risk_free_unit"] not in {"annual_decimal", "per_period_decimal"}:
        raise ValueError("basis.risk_free_unit: explicit rate unit required")
    if basis["risk_free_compounding"] not in {"simple", "effective", "continuous"}:
        raise ValueError("basis.risk_free_compounding: explicit compounding convention required")
    times = []
    for key in ("period_start", "period_end"):
        token = basis[key]
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z", token):
            raise ValueError(f"basis.{key}: UTC ISO timestamp ending Z required")
        times.append(datetime.fromisoformat(token.replace("Z", "+00:00")))
    if times[0] >= times[1]:
        raise ValueError("basis.period_end: must follow period_start")
    for key in ("observation_schedule_id", "cost_policy_id", "valuation_policy_id"):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", basis[key]):
            raise ValueError(f"basis.{key}: exact content hash required")
    provenance = value["provenance"]
    _closed(provenance, {"code_identity", "input_identity"}, "provenance")
    for field, pattern in (
        ("code_identity", r"[0-9a-f]{40}|[0-9a-f]{64}"),
        ("input_identity", r"sha256:[0-9a-f]{64}"),
    ):
        identities = provenance[field]
        if not isinstance(identities, dict) or not identities:
            raise ValueError(f"provenance.{field}: nonempty identity map required")
        for name, identity in identities.items():
            if (
                not _text(name)
                or not isinstance(identity, str)
                or not re.fullmatch(pattern, identity)
            ):
                raise ValueError(f"provenance.{field}: exact content identity required")
    scopes = value["unsupported_scopes"]
    if not isinstance(scopes, list) or not all(_text(v) for v in scopes):
        raise ValueError("unsupported_scopes: explicit string list required")
    if len(scopes) != len(set(scopes)):
        raise ValueError("unsupported_scopes: duplicate scope")
    return deepcopy(value)


def _comparison_basis(value):
    return {
        "basis": value["basis"],
        "evidence_kind": value["evidence_kind"],
        "sample_kind": value["sample_kind"],
        "input_identity": value["provenance"]["input_identity"],
        "unsupported_scopes": sorted(value["unsupported_scopes"]),
    }


def basis_id(value: dict) -> str:
    """Stable equality key, excluding code identities which stay in provenance."""
    value = validate_measurement(value)
    encoded = json.dumps(_comparison_basis(value), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def compare_measurements(measurements, *, statuses=None) -> dict:
    """Require all requested runs, never silently drop failed/missing candidates."""
    values = list(measurements)
    statuses = ["completed"] * len(values) if statuses is None else list(statuses)
    if len(statuses) != len(values):
        raise ValueError("one status per measurement required")
    reasons, valid = [], {}

    def reason(index, field, code, message):
        reasons.append({"index": index, "field": field, "code": code, "message": message})

    if len(values) < 2:
        reason(None, "candidates", "insufficient", "at least two requested runs required")
    for index, (value, status) in enumerate(zip(values, statuses)):
        if status != "completed":
            reason(index, "status", "unavailable", f"run status is {status}")
        if value is None:
            reason(index, "measurement", "unknown", "legacy or missing measurement declaration")
            continue
        try:
            valid[index] = validate_measurement(value)
        except (ValueError, TypeError) as exc:
            reason(index, "measurement", "invalid", str(exc))
    if valid:
        first_index = min(valid)
        first = _comparison_basis(valid[first_index])
        for index, value in valid.items():
            current = _comparison_basis(value)
            for key in sorted(BASIS_FIELDS):
                # Exact values: no date, annualization, unit or economic conversion.
                if content_id(current["basis"][key]) != content_id(first["basis"][key]):
                    reason(index, "basis." + key, "mismatch", f"differs from run {first_index}")
            for key in ("evidence_kind", "sample_kind", "input_identity", "unsupported_scopes"):
                if current[key] != first[key]:
                    reason(index, key, "mismatch", f"differs from run {first_index}")
    comparable = len(values) >= 2 and not reasons
    return {
        "schema": "quant.measurement-comparison/v1",
        "comparable": comparable,
        "reasons": reasons,
        "basis_id": basis_id(valid[0]) if comparable else None,
        "certification": "declared_economic_compatibility_only",
    }
