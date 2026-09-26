"""Closed research recipes, preregistered candidates and resumable execution.

The contract layer accepts a caller-owned executor; it never imports strategy code
or evaluates user Python/shell. Every completed candidate has a byte manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from quant_lab.trials import TrialRegistry, canonical

SCHEMA = "quant.research-recipe/v1"
ROOT_FIELDS = {
    "schema_version",
    "study_id",
    "hypothesis",
    "mode",
    "holdout",
    "backend",
    "inputs",
    "interval",
    "factors",
    "strategy",
    "costs",
    "risk",
    "variants",
    "diagnostics",
    "source",
    "required_history",
    "backend_parameters",
}
STRATEGY_FIELDS = {"family", "frequency", "top_n", "max_weight", "cash_buffer", "trend_window"}
COST_FIELDS = {
    "initial_capital",
    "commission",
    "min_commission",
    "stamp_tax",
    "slippage",
    "participation_rate",
}
VARIANT_FIELDS = {
    "name",
    "factors",
    "strategy",
    "cost_multiplier",
    "signal_delay",
    "backend_parameters",
}


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _closed(value: object, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"Invalid {name} mapping or unknown fields")
    return value


def _number(value, name, *, lower=0, upper=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if (
        value < lower
        or (upper is not None and value > upper)
        or (integer and type(value) is not int)
    ):
        raise ValueError(f"{name} out of range")


def validate_recipe(value: dict) -> dict:
    recipe = deepcopy(_closed(value, ROOT_FIELDS, "recipe"))
    if recipe.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported research recipe schema")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", recipe.get("study_id", "")):
        raise ValueError("study_id must be a safe nonempty identifier")
    if not isinstance(recipe.get("hypothesis"), str) or not recipe["hypothesis"].strip():
        raise ValueError("An explicit research hypothesis is required")
    if recipe.get("mode") not in {"exploratory", "prospective"}:
        raise ValueError("Research mode must be explicit")
    if recipe.get("backend") not in {"equity", "futures_fixture", "crypto_fixture"}:
        raise ValueError("Unknown research backend")
    backend_params = _closed(
        recipe.get("backend_parameters", {}),
        {"entry_basis_bps", "exit_basis_bps", "minimum_funding_rate", "quantity", "passive_limits"}
        if recipe["backend"] == "crypto_fixture"
        else {"quantity_multiplier"}
        if recipe["backend"] == "futures_fixture"
        else set(),
        "backend parameters",
    )
    for field, number in backend_params.items():
        if field == "passive_limits":
            if type(number) is not bool:
                raise ValueError("passive_limits must be boolean")
        else:
            _number(number, field, lower=-1 if field == "minimum_funding_rate" else 0)
    inputs = _closed(
        recipe.get("inputs"), {"bundle", "history", "catalog", "source", "config"}, "inputs"
    )
    if not inputs or any(not isinstance(v, str) or not v.strip() for v in inputs.values()):
        raise ValueError("Explicit input references are required")
    interval = _closed(recipe.get("interval"), {"start", "end"}, "interval")
    for field in ("start", "end"):
        date.fromisoformat(interval.get(field, ""))
    if interval["start"] >= interval["end"]:
        raise ValueError("Research end must follow start")
    factors = recipe.get("factors")
    if (
        not isinstance(factors, dict)
        or not factors
        or any(
            not isinstance(k, str) or type(v) is not int or v not in {-1, 1}
            for k, v in factors.items()
        )
    ):
        raise ValueError("Factors require explicit +1/-1 directions")
    strategy = _closed(recipe.get("strategy"), STRATEGY_FIELDS, "strategy")
    if set(strategy) != STRATEGY_FIELDS:
        raise ValueError("All strategy fields must be explicit")
    if strategy["family"] not in {"rank", "etf_trend", "buy_hold", "spread", "basis"}:
        raise ValueError("Unknown strategy family")
    if strategy["frequency"] not in {"daily", "weekly", "monthly"}:
        raise ValueError("Invalid rebalance frequency")
    for field in ("top_n", "trend_window"):
        _number(strategy[field], field, lower=1, upper=1000, integer=True)
    for field in ("max_weight", "cash_buffer"):
        _number(strategy[field], field, upper=1)
    if strategy["max_weight"] == 0 or strategy["cash_buffer"] == 1:
        raise ValueError("Strategy must permit some investment")
    costs = _closed(recipe.get("costs"), COST_FIELDS, "costs")
    if set(costs) != COST_FIELDS:
        raise ValueError("All cost assumptions must be explicit")
    for field, number in costs.items():
        _number(
            number,
            field,
            lower=0,
            upper=None if field in {"initial_capital", "min_commission"} else 1,
        )
    if costs["initial_capital"] <= 0 or costs["participation_rate"] <= 0:
        raise ValueError("Capital and participation rate must be positive")
    if recipe["backend"] != "equity":
        expected_family = "spread" if recipe["backend"] == "futures_fixture" else "basis"
        expected_factor = (
            "spread_fixture" if recipe["backend"] == "futures_fixture" else "basis_funding_fixture"
        )
        if strategy != {
            "family": expected_family,
            "frequency": "daily",
            "top_n": 2,
            "max_weight": 0.5,
            "cash_buffer": 0.2,
            "trend_window": 20,
        } or factors != {expected_factor: 1}:
            raise ValueError(
                "Fixture strategy fields are descriptors; only backend_parameters may vary"
            )
        if (
            any(costs[k] != 0 for k in ("commission", "min_commission", "stamp_tax", "slippage"))
            or costs["participation_rate"] != 1
        ):
            raise ValueError(
                "Fixture costs use their frozen contract; generic overrides are unsupported"
            )
        if (
            recipe.get("risk")
            or recipe.get("required_history")
            or any(recipe.get("diagnostics", {}).values())
        ):
            raise ValueError("Fixture backends require explicit backend parameter variants")
    _closed(
        recipe.get("risk", {}),
        {
            "max_drawdown",
            "max_single_weight",
            "max_gross_weight",
            "min_cash_weight",
            "max_positions",
            "max_turnover",
            "max_estimated_cost_rate",
            "estimated_cost_rate_per_turnover",
        },
        "risk",
    )
    for field, number in recipe.get("risk", {}).items():
        _number(number, field, upper=1000 if field == "max_positions" else 1)
    diagnostics = _closed(
        recipe.get("diagnostics", {}),
        {"single_factors", "ablations", "cost_multipliers", "signal_delays", "frequencies"},
        "diagnostics",
    )
    for field in ("single_factors", "ablations"):
        if field in diagnostics and type(diagnostics[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    for multiplier in diagnostics.get("cost_multipliers", []):
        _number(multiplier, "cost multiplier", lower=1, upper=100)
    for delay in diagnostics.get("signal_delays", []):
        _number(delay, "signal delay", lower=0, upper=20, integer=True)
    if any(f not in {"daily", "weekly", "monthly"} for f in diagnostics.get("frequencies", [])):
        raise ValueError("Unknown diagnostic frequency")
    if recipe["mode"] == "prospective":
        holdout = _closed(recipe.get("holdout"), {"start", "end"}, "holdout")
        for field in ("start", "end"):
            date.fromisoformat(holdout.get(field, ""))
        if holdout["start"] <= interval["end"] or holdout["end"] <= holdout["start"]:
            raise ValueError("Holdout must follow development and have positive duration")
    elif recipe.get("holdout"):
        raise ValueError("Exploratory studies cannot claim an untouched holdout")
    histories = recipe.get("required_history", {})
    if not isinstance(histories, dict) or any(
        v not in {"universe", "status", "fundamentals", "classification"}
        for v in histories.values()
    ):
        raise ValueError("Invalid required history domains")
    for variant in recipe.get("variants", []):
        _closed(variant, VARIANT_FIELDS, "variant")
        if not isinstance(variant.get("name"), str) or not variant["name"].strip():
            raise ValueError("Variant name required")
        _number(variant.get("cost_multiplier", 1), "cost multiplier", lower=0, upper=100)
        _number(variant.get("signal_delay", 0), "signal delay", upper=20, integer=True)
        changed = deepcopy(recipe)
        changed["variants"] = []
        changed["factors"] = variant.get("factors", factors)
        changed["strategy"] = {**strategy, **variant.get("strategy", {})}
        changed["backend_parameters"] = {**backend_params, **variant.get("backend_parameters", {})}
        validate_recipe(changed)
    canonical(recipe)
    return recipe


def load_recipe(path: Path) -> dict:
    recipe = validate_recipe(yaml.safe_load(path.read_text(encoding="utf-8")))
    for key in ("bundle", "history", "catalog", "config"):
        if key in recipe["inputs"]:
            recipe["inputs"][key] = str((path.parent / recipe["inputs"][key]).resolve())
    return recipe


def candidates(recipe: dict) -> list[dict]:
    recipe = validate_recipe(recipe)
    base = {
        "name": "base",
        "factors": recipe["factors"],
        "strategy": recipe["strategy"],
        "cost_multiplier": 1,
        "signal_delay": 0,
        "backend_parameters": recipe.get("backend_parameters", {}),
    }
    rows = [base]
    if recipe["backend"] == "equity":
        rows.append(
            {
                **deepcopy(base),
                "name": "buy_hold",
                "strategy": {**base["strategy"], "family": "buy_hold"},
            }
        )
    diagnostics = recipe.get("diagnostics", {})
    if diagnostics.get("single_factors") and len(base["factors"]) > 1:
        rows += [
            {**deepcopy(base), "name": "single_" + name, "factors": {name: direction}}
            for name, direction in base["factors"].items()
        ]
    if diagnostics.get("ablations") and len(base["factors"]) > 2:
        rows += [
            {
                **deepcopy(base),
                "name": "without_" + name,
                "factors": {k: v for k, v in base["factors"].items() if k != name},
            }
            for name in base["factors"]
        ]
    rows += [
        {**deepcopy(base), "name": f"cost_{m:g}x", "cost_multiplier": m}
        for m in diagnostics.get("cost_multipliers", [])
        if m != 1
    ]
    rows += [
        {**deepcopy(base), "name": f"delay_{d}", "signal_delay": d}
        for d in diagnostics.get("signal_delays", [])
        if d
    ]
    rows += [
        {
            **deepcopy(base),
            "name": "frequency_" + f,
            "strategy": {**base["strategy"], "frequency": f},
        }
        for f in diagnostics.get("frequencies", [])
        if f != base["strategy"]["frequency"]
    ]
    rows += [
        {**deepcopy(base), **v, "strategy": {**base["strategy"], **v.get("strategy", {})}}
        for v in recipe.get("variants", [])
    ]
    for row in rows:
        row["backend_parameters"] = {**base["backend_parameters"], **row["backend_parameters"]}
    if len(rows) > 64 or len({r["name"] for r in rows}) != len(rows):
        raise ValueError("Candidate names must be unique; at most 64 preregistered candidates")
    for row in rows:
        row["candidate_id"] = digest(row)[:20]
    return rows


@contextmanager
def study_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".research.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def verify_result(path: Path, expected_sha: str) -> dict:
    if file_hash(path) != expected_sha:
        raise ValueError("Cached result hash mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    for relative, sha in result["artifacts"].items():
        target = (path.parent / relative).resolve()
        if path.parent.resolve() not in target.parents or file_hash(target) != sha:
            raise ValueError("Cached artifact missing, changed or outside candidate")
    return result


def execute_study(
    recipe: dict, root: Path, *, identity: dict, data_identity: dict, executor
) -> dict:
    """Register ALL candidates before computation; preserve and resume every attempt."""
    recipe = validate_recipe(recipe)
    planned = candidates(recipe)
    study = recipe["study_id"]
    definition = {
        "hypothesis": recipe["hypothesis"],
        "parameters": planned,
        "code_identity": identity,
        "selection_rule": "report all; no automatic winner promotion",
        "recipe": recipe,
        "data_identity": data_identity,
    }
    if recipe["mode"] == "prospective":
        definition.update(
            holdout_start=recipe["holdout"]["start"], holdout_end=recipe["holdout"]["end"]
        )
    with study_lock(root):
        registry = TrialRegistry(root / "experiments.db")
        registry.register(study, definition)
        history = registry.history(study)
        completed = {}
        started = {e["attempt_id"]: e for e in history if e["status"] == "running"}
        terminals = {e["attempt_id"] for e in history if e["status"] != "running"}
        for attempt, event in started.items():
            if attempt not in terminals:
                registry.finish(
                    attempt,
                    "interrupted",
                    {"reason": "previous writer exited; exclusive lock acquired"},
                )
        for event in history:
            if event["status"] == "completed":
                completed[event["payload"]["candidate_id"]] = event["payload"]
        results = []
        for candidate in planned:
            key = candidate["candidate_id"]
            if key in completed:
                cached = completed[key]
                path = root / cached["result"]
                if root.resolve() not in path.resolve().parents:
                    raise ValueError("Cached result escapes study")
                results.append(verify_result(path, cached["sha256"]))
                continue
            attempt = registry.start(study, candidate)
            out = root / "attempts" / attempt
            out.mkdir(parents=True, exist_ok=False)
            try:
                payload = executor(recipe, candidate, out)
                payload.update(
                    schema_version="quant.research-result/v1",
                    candidate=candidate,
                    study_id=study,
                    attempt_id=attempt,
                    identity=identity,
                    data_identity=data_identity,
                    mode=recipe["mode"],
                    status="completed",
                )
                payload["artifacts"] = {
                    str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                }
                target = out / "result.json"
                target.write_text(canonical(payload), encoding="utf-8")
                registry.finish(
                    attempt,
                    "completed",
                    {
                        "candidate_id": key,
                        "result": str(target.relative_to(root)),
                        "sha256": file_hash(target),
                    },
                )
                results.append(payload)
            except Exception as exc:
                logging.getLogger(__name__).exception("Research candidate failed: %s", key)
                failure = {
                    "candidate": candidate,
                    "attempt_id": attempt,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                (out / "failure.json").write_text(canonical(failure), encoding="utf-8")
                registry.finish(attempt, "failed", failure)
                results.append(failure)
        summary = {
            "schema_version": "quant.research-study/v1",
            "study_id": study,
            "definition_sha256": digest(definition),
            "recipe": recipe,
            "results": results,
            "attempts": registry.history(study),
            "completed": sum(r["status"] == "completed" for r in results),
            "failed": sum(r["status"] == "failed" for r in results),
        }
        temporary = root / "study.json.tmp"
        temporary.write_text(canonical(summary), encoding="utf-8")
        temporary.replace(root / "study.json")
        return summary


def compare_results(results: list[dict]) -> dict:
    """Never silently rank incompatible universes, costs, windows or data vintages."""
    fields = ("data_identity", "comparison", "identity")
    completed = [r for r in results if r.get("status") == "completed"]
    mismatches = []
    if completed:
        first = completed[0]
        for result in completed[1:]:
            for field in fields:
                if result.get(field) != first.get(field):
                    mismatches.append({"candidate": result["candidate"]["name"], "field": field})
    return {
        "comparable": bool(completed) and not mismatches,
        "mismatches": mismatches,
        "results": completed,
    }


def initialize_recipe(
    template: Path,
    output: Path,
    *,
    study_id: str,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> dict:
    recipe = load_recipe(template)
    recipe["study_id"] = study_id
    recipe["mode"] = "prospective" if holdout_start else "exploratory"
    recipe.pop("holdout", None)
    if holdout_start:
        if holdout_start <= datetime.now(timezone.utc).date().isoformat():
            raise ValueError("New prospective studies require a future holdout start")
        recipe["holdout"] = {"start": holdout_start, "end": holdout_end}
    elif holdout_end:
        raise ValueError("holdout_start is required with holdout_end")
    validate_recipe(recipe)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(recipe, handle, allow_unicode=True, sort_keys=False)
    return recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--template", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    init.add_argument("--study-id", required=True)
    init.add_argument("--holdout-start")
    init.add_argument("--holdout-end")
    plan = sub.add_parser("plan")
    plan.add_argument("recipe", type=Path)
    args = parser.parse_args()
    if args.command == "init":
        initialize_recipe(
            args.template,
            args.output,
            study_id=args.study_id,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
        )
    else:
        print(json.dumps(candidates(load_recipe(args.recipe)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
