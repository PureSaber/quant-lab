"""Composable exploratory factor research using immutable inputs and the PIT cache.

The QDK adapter owns all artifact access. Task JSON cannot supply local paths,
Python, shell, fitting code, or arbitrary query expressions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import ipc

from quant_lab import selection, trials
from quant_lab.trials import TrialRegistry, canonical

SCHEMA = "quant.rolling-research/v1"
_LOG = logging.getLogger(__name__)


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def code_identity() -> dict:
    from quant_factors.cache import source_identity

    paths = [Path(__file__), Path(selection.__file__), Path(trials.__file__)]
    return {
        "factors": source_identity(),
        "numpy": np.__version__,
        "lab": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


@dataclass(frozen=True)
class ResolvedMarket:
    table: pa.Table
    input_identities: list[dict]


class ResearchAdapter(Protocol):
    """Trusted injectable boundary; implementations must verify before returning."""

    def resolve(
        self, snapshot: dict, *, columns: list[str], universe: list[str]
    ) -> ResolvedMarket: ...

    def publish(self, value: bytes, *, media_type: str) -> dict: ...


class QdkResearchAdapter:
    """Real data-track CAS/snapshot adapter, with durable input/output ownership."""

    def __init__(self, store, *, owner: str):
        from quant_data_kit.shared_snapshots import SnapshotStore

        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Artifact owner required")
        self.store, self.owner = store, owner
        self.snapshots = SnapshotStore(store)

    def resolve(self, snapshot, *, columns, universe):
        from quant_data_kit.artifacts import ArtifactRef

        ref = ArtifactRef.from_dict(snapshot)
        self.store.retain(self.owner, [ref])
        manifest = self.snapshots.load(ref)
        roles = manifest.document["query_columns"]
        if roles != {
            "date_column": "event_time",
            "symbol_column": "instrument_id",
            "available_at_column": "available_at",
        }:
            raise ValueError(
                "Research snapshot must declare engine event/identity/availability roles"
            )
        # Preserve every event in the window. Dropping unavailable events here would
        # incorrectly change period counts; existing PIT engine emits null instead.
        table = self.snapshots.read_table(ref, columns=columns, symbols=universe)
        inputs = [ref.to_dict(), *[member["artifact"] for member in manifest.document["members"]]]
        return ResolvedMarket(table, inputs)

    def publish(self, value, *, media_type):
        return self.store.put_bytes(value, media_type=media_type, owner=self.owner).to_dict()


def validate_payload(payload: dict) -> dict:
    from quant_factors.cache import SUPPORTED_ALGORITHMS, artifact_identity
    from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec

    required = {"schema_version", "snapshot", "frequency", "factors", "as_of", "universe"}
    if (
        not isinstance(payload, dict)
        or not required <= payload.keys()
        or payload.keys() - required - {"training", "diagnostics", "study_id"}
    ):
        raise ValueError("Invalid rolling research fields")
    value = json.loads(canonical(payload))
    if value["schema_version"] != SCHEMA:
        raise ValueError("Unsupported rolling research schema")
    artifact_identity(value["snapshot"])
    FrequencySpec(**value["frequency"])
    AsOfSpec(**value["as_of"])
    if not isinstance(value["factors"], list) or not value["factors"]:
        raise ValueError("Explicit factors required")
    specs = []
    for raw in value["factors"]:
        spec = FactorSpec(
            **{**raw, "dependencies": [FactorDependency(**d) for d in raw["dependencies"]]}
        )
        if (
            spec.algorithm_id not in SUPPORTED_ALGORITHMS
            or len(spec.dependencies) != 1
            or spec.dependencies[0].role != "market"
        ):
            raise ValueError("Only supported unary market rolling factors are accepted")
        specs.append(spec)
    if len({s.factor_id for s in specs}) != len(specs):
        raise ValueError("Distinct factor IDs required")
    value["factors"] = [s.to_contract() for s in sorted(specs, key=lambda s: s.factor_id)]
    universe = value["universe"]
    if (
        not isinstance(universe, list)
        or not universe
        or any(not isinstance(s, str) or not s.strip() for s in universe)
        or len(set(universe)) != len(universe)
    ):
        raise ValueError("Explicit distinct universe required")
    value["universe"] = sorted(universe)
    diagnostics = value.setdefault("diagnostics", {"hac_lags": 0})
    if (
        not isinstance(diagnostics, dict)
        or set(diagnostics) != {"hac_lags"}
        or type(diagnostics["hac_lags"]) is not int
        or diagnostics["hac_lags"] < 0
    ):
        raise ValueError("diagnostics requires a nonnegative integer hac_lags")
    if "study_id" in value and (
        not isinstance(value["study_id"], str) or not value["study_id"].strip()
    ):
        raise ValueError("study_id must be nonempty")
    if value.get("training") is not None:
        from quant_factors.cache import FactorCache

        FactorCache._validate_training(value["training"], AsOfSpec(**value["as_of"]))
    return value


def _parameters(plan):
    from quant_factors.cache import artifact_identity

    identity = {k: v for k, v in plan.items() if k != "study_id"}
    identity["snapshot"] = artifact_identity(plan["snapshot"])
    return {"request_id": _digest({"plan": identity, "code": code_identity()})}


def register_research(
    registry: TrialRegistry, study_id: str, *, hypothesis: str, payloads: list[dict]
):
    """Declare the complete candidate list through the existing TrialRegistry."""
    plans = [validate_payload(payload) for payload in payloads]
    return registry.register(
        study_id,
        {
            "hypothesis": hypothesis,
            "parameters": [_parameters(plan) for plan in plans],
            "code_identity": _digest(code_identity()),
            "selection_rule": "exploratory descriptive factor diagnostics; no winner selection",
        },
    )


def compute_research(plan: dict, *, adapter: ResearchAdapter, cache, run_id: str):
    """Resolve and evaluate a validated plan. Does not publish results or register trials."""
    from quant_factors.cache import artifact_identity
    from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec

    specs = [
        FactorSpec(**{**raw, "dependencies": [FactorDependency(**d) for d in raw["dependencies"]]})
        for raw in plan["factors"]
    ]
    columns = {"instrument_id", "event_time", "sequence", "event_id", "received_at", "available_at"}
    for spec in specs:
        for dependency in spec.dependencies:
            columns.update((dependency.value_column, dependency.availability_column))
    market = adapter.resolve(plan["snapshot"], columns=sorted(columns), universe=plan["universe"])
    wanted = artifact_identity(plan["snapshot"])
    if wanted not in [artifact_identity(ref) for ref in market.input_identities]:
        raise ValueError("Resolver must bind the requested immutable snapshot")
    return cache.compute(
        market.table,
        FrequencySpec(**plan["frequency"]),
        specs,
        as_of=AsOfSpec(**plan["as_of"]),
        input_identities=market.input_identities,
        universe=plan["universe"],
        run_id=run_id,
        training=plan.get("training"),
    )


def summarize_factors(table: pa.Table, plan: dict) -> dict:
    """Existing HAC statistics on per-event equal-weight factor means, not returns."""
    frame = table.to_pandas()
    if plan.get("training") is not None:
        start = pd.Timestamp(int(plan["training"]["start_ns"]), unit="ns", tz="UTC")
        frame = frame[frame.event_time >= start]
    summary = {}
    for factor in plan["factors"]:
        name = factor["factor_id"]
        values = frame.groupby("event_time", sort=True)[name].mean().dropna()
        lags = plan["diagnostics"]["hac_lags"]
        summary[name] = {
            "non_null_rows": int(frame[name].notna().sum()),
            "total_rows": len(frame),
            "observation_count": len(values),
            "estimand": "equal-weight cross-sectional factor value by event_time; not investment return",
            "hac": selection.hac_mean(values.to_numpy(), lags=lags)
            if len(values) >= 3 and lags < len(values)
            else None,
            "status": "computed"
            if len(values) >= 3 and lags < len(values)
            else "insufficient_observations",
        }
    return summary


def execute_research(
    payload: dict,
    *,
    adapter: ResearchAdapter,
    cache,
    run_id: str,
    registry: TrialRegistry | None = None,
    execution_context: dict | None = None,
) -> dict:
    """Worker-friendly orchestration. Optional trial attempts use existing registry semantics."""
    plan = validate_payload(payload)
    if registry is not None and "study_id" not in plan:
        raise ValueError("Configured TrialRegistry requires a preregistered study_id")
    if registry is None and "study_id" in plan:
        raise ValueError("study_id requires a TrialRegistry")
    attempt = (
        registry.start(
            plan["study_id"],
            _parameters(plan),
            context={**(execution_context or {}), "run_id": run_id},
        )
        if registry
        else None
    )
    try:
        computed = compute_research(plan, adapter=adapter, cache=cache, run_id=run_id)
        summary = summarize_factors(computed.table, plan)
        identity = {
            "factor_computation_id": computed.computation_id,
            "diagnostics": plan["diagnostics"],
            "code": code_identity(),
        }
        research_id = _digest(identity)
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, computed.table.schema) as writer:
            writer.write_table(computed.table)
        factors_ref = adapter.publish(
            sink.getvalue().to_pybytes(), media_type="application/vnd.apache.arrow.stream"
        )
        report = {
            "schema_version": "quant.rolling-research-result/v1",
            "research_id": research_id,
            "identity": identity,
            "factor_identity": computed.metadata["identity"],
            "summary": summary,
            "factors": factors_ref,
        }
        report_ref = adapter.publish(canonical(report).encode(), media_type="application/json")
        outputs = [factors_ref, report_ref]
        from quant_factors.cache import artifact_identity

        for ref in outputs:
            artifact_identity(ref)
        result = {
            "research_id": research_id,
            "computation_id": computed.computation_id,
            "summary": summary,
            "cache": {
                key: computed.metadata[key]
                for key in (
                    "execution_mode",
                    "engine_rows",
                    "recomputed_rows",
                    "reused_rows",
                    "input_rows",
                )
            },
            "factors": factors_ref,
            "report": report_ref,
        }
        if registry:
            registry.finish(attempt, "completed", {"result": result, "outputs": outputs})
            result["trial_attempt_id"] = attempt
        return {"result": result, "outputs": outputs}
    except Exception as exc:
        if registry:
            registry.finish(
                attempt, "failed", {"error_type": type(exc).__name__, "error": str(exc)}
            )
        raise


def _configured_execution(request: dict, context: dict) -> dict:
    from quant_data_kit.artifacts import ArtifactStore
    from quant_factors.cache import FactorCache, artifact_identity

    if request.get("schema_version") != "quant.job-request/v1":
        raise ValueError("Expected quant.job-request/v1")
    plan = validate_payload(request["payload"])
    if artifact_identity(plan["snapshot"]) not in [artifact_identity(r) for r in request["inputs"]]:
        raise ValueError("Job inputs must include the requested snapshot")
    run_id = request["run_id"]
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("Nonempty run_id required")
    if (
        context.get("schema_version") != "quant.attempt-context/v1"
        or any(
            not isinstance(context.get(key), str) or not context[key].strip()
            for key in ("job_id", "attempt_id", "worker_id", "project_id", "run_id")
        )
        or type(context.get("fencing_token")) is not int
        or context["fencing_token"] <= 0
        or context["project_id"] != request["project_id"]
        or context["run_id"] != run_id
    ):
        raise ValueError("Attempt context must bind this job request")
    store = ArtifactStore(Path(os.environ["QUANT_RESEARCH_STORE_ROOT"]), context=context)
    cache = FactorCache(Path(os.environ["QUANT_RESEARCH_CACHE_DB"]))
    registry_path = os.environ.get("QUANT_RESEARCH_REGISTRY_DB")
    registry = TrialRegistry(Path(registry_path)) if registry_path else None
    owner = f"job/{context['job_id']}/attempt/{context['attempt_id']}/outputs"
    try:
        result = execute_research(
            plan,
            adapter=QdkResearchAdapter(store, owner=owner),
            cache=cache,
            run_id=run_id,
            registry=registry,
            execution_context=context,
        )
    except Exception:
        _LOG.exception(
            canonical(
                {
                    **context,
                    "event": "research_failed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )
        raise
    _LOG.info(
        canonical(
            {
                **context,
                "event": "research_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "research_id": result["result"]["research_id"],
            }
        )
    )
    return result


def worker_handler(request: dict, workspace: Path) -> dict:
    """Use runtime-owned context.json; never infer an attempt from task payload."""
    context = json.loads((Path(workspace) / "context.json").read_text(encoding="utf-8"))
    return _configured_execution(request, context)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Execute a fixed rolling-research job JSON")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    # Standalone execution is explicitly separate from runtime-leased jobs.
    # A fresh attempt directory is required; never replace an existing context.
    context = {
        "schema_version": "quant.attempt-context/v1",
        "job_id": "cli-" + uuid4().hex,
        "attempt_id": uuid4().hex,
        "fencing_token": 1,
        "worker_id": "standalone-cli",
        "project_id": request["project_id"],
        "run_id": request["run_id"],
    }
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace / "context.json").open("x", encoding="utf-8") as handle:
        handle.write(canonical(context))
    result = worker_handler(request, args.workspace)
    print(canonical(result))


if __name__ == "__main__":
    main()
