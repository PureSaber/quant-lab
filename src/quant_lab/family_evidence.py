"""Registry-backed cross-study evidence; every attempt has a matrix column or a failure."""

from pathlib import Path

import pandas as pd

from quant_lab.research import canonical, digest, verify_result
from quant_lab.selection import (
    audit_family,
    bootstrap_means,
    cscv_pbo,
    deflated_sharpe,
    returns_to_losses,
    spa_mcs,
)


def collect_family(registry, family_id):
    inventory = registry.family_inventory(family_id)
    planned, statuses, series, errors = [], {}, {}, []
    comparison = None
    for study in inventory["studies"]:
        if study.get("status") == "not_registered":
            key = study["study_id"] + ":not_registered"
            planned.append(key)
            statuses[key] = "running"
            continue
        starts = {e["attempt_id"]: e for e in study["events"] if e["status"] == "running"}
        attempted = {canonical(e["payload"]["parameters"]) for e in starts.values()}
        for parameter in study["planned"]:
            if canonical(parameter) not in attempted:
                key = study["study_id"] + ":not_attempted:" + digest(parameter)[:12]
                planned.append(key)
                statuses[key] = "running"
        for attempt in study["attempts"]:
            key = attempt["attempt_id"]
            planned.append(key)
            statuses[key] = attempt["status"]
            if attempt["status"] != "completed":
                continue
            try:
                root = Path(starts[key]["payload"]["context"]["artifact_root"]).resolve()
                result_path = (root / attempt["payload"]["result"]).resolve()
                if root not in result_path.parents:
                    raise ValueError("family result path escapes registered artifact root")
                result = verify_result(result_path, attempt["payload"]["sha256"])
                if result["attempt_id"] != key or result["study_id"] != study["study_id"]:
                    raise ValueError("family attempt identity mismatch")
                if "returns.csv" not in result["artifacts"]:
                    raise ValueError("completed attempt has no verified return matrix artifact")
                basis = inventory["definition"]["basis"]
                actual_basis = result.get("measurement_basis")
                if actual_basis != basis:
                    raise ValueError("attempt must explicitly match the family's measurement_basis")
                comparable = {
                    k: result.get("comparison", {}).get(k)
                    for k in ("start", "end", "currency", "costs")
                }
                if any(v is None for v in comparable.values()):
                    raise ValueError("missing common date/currency/cost comparison metadata")
                if comparison is not None and comparison != comparable:
                    raise ValueError("family date/currency/cost policies differ")
                comparison = comparable
                data = pd.read_csv(
                    result_path.parent / "returns.csv", index_col=0, parse_dates=True
                )
                if data.shape[1] != 1:
                    raise ValueError("one candidate return series per attempt required")
                series[key] = data.iloc[:, 0]
            except (KeyError, ValueError, OSError) as exc:
                errors.append({"attempt_id": key, "reason": str(exc)})
    # OUTER concat deliberately exposes missing dates; never an inner join.
    values = pd.DataFrame(series).sort_index()
    audit = audit_family(
        values,
        planned=planned,
        statuses=statuses,
        family_id=family_id,
        basis=inventory["definition"]["basis"],
    )
    audit["collection_errors"] = errors
    audit["inventory"] = inventory
    audit["available"] = audit["available"] and not errors
    return values, audit


def evaluate_family(
    returns,
    audit,
    *,
    benchmark=None,
    blocks=8,
    block_lengths=(5, 10, 20),
    repetitions=1000,
    seed=17,
):
    result = {"schema": "quant.family-evidence/v1", "audit": audit, "methods": {}}
    if not audit["available"]:
        return result
    excess = returns
    if benchmark is not None:
        excess = -returns_to_losses(returns, benchmark)
    methods = {
        "dsr": lambda: deflated_sharpe(excess, periods_per_year=audit["basis"]["periods_per_year"]),
        "cscv": lambda: cscv_pbo(excess, blocks=blocks),
        "bootstrap": lambda: bootstrap_means(
            excess, block_lengths=block_lengths, repetitions=repetitions, seed=seed
        ),
    }
    result["return_basis"] = (
        "net excess over supplied benchmark"
        if benchmark is not None
        else "net return; no risk-free benchmark assumed"
    )
    if benchmark is None:
        methods.pop("dsr")
        result["methods"]["dsr"] = {
            "available": False,
            "reason": "an explicit matched benchmark series is required; zero risk-free rates are not assumed",
        }
    if benchmark is not None:
        methods["spa_mcs"] = lambda: spa_mcs(
            returns_to_losses(returns, benchmark),
            block_length=block_lengths[0],
            repetitions=repetitions,
            seed=seed,
        )
    for name, evaluate in methods.items():
        try:
            result["methods"][name] = {"available": True, "evidence": evaluate()}
        except (ValueError, ImportError) as exc:
            result["methods"][name] = {"available": False, "reason": str(exc)}
    result["policy"] = "report every method and sensitivity; never auto-promote a winning strategy"
    return result
