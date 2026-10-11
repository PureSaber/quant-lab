"""Optional three-repository integration: all local acceptance runs use real QDK."""

import json
import os
import sqlite3
import subprocess
import sys
from copy import deepcopy

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("quant_factors.cache", reason="rolling research needs the optional factors SDK")
pytest.importorskip("quant_data_kit.shared_snapshots", reason="rolling research needs QDK CAS")

from quant_data_kit.artifacts import ArtifactRef, ArtifactStore
from quant_data_kit.exceptions import ValidationError
from quant_data_kit.shared_snapshots import SnapshotStore
from quant_factors.cache import FactorCache
from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec

from quant_lab.rolling_research import (
    QdkResearchAdapter,
    ResolvedMarket,
    compute_research,
    execute_research,
    register_research,
    summarize_factors,
    validate_payload,
    worker_handler,
)
from quant_lab.trials import TrialRegistry


@pytest.fixture
def setup(tmp_path):
    rows = []
    for symbol in ("A", "B"):
        for i in range(60):
            stamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i)
            rows.append(
                {
                    "instrument_id": symbol,
                    "event_time": stamp,
                    "received_at": stamp,
                    "available_at": stamp,
                    "sequence": i,
                    "event_id": f"{symbol}-{i}",
                    "close": float(100 + i + i % 5),
                }
            )
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    store = ArtifactStore(tmp_path / "cas", context={"run_id": "test"})
    snapshots = SnapshotStore(store)
    manifest = snapshots.publish(
        {"prices": source},
        dataset="synthetic",
        version="v1",
        temporal={"calendar": "synthetic"},
        owner="fixture",
        date_column="event_time",
        symbol_column="instrument_id",
        available_at_column="available_at",
    )
    freq = FrequencySpec(
        "daily", "session_bar", "365", "synthetic", "v1", None, "trading_day", None, None, None
    )
    factor = FactorSpec(
        "momentum",
        "1.0.0",
        "momentum",
        "bar",
        [FactorDependency("market", "close", "available_at")],
        5,
        "float64",
        False,
        "null",
    )
    payload = {
        "schema_version": "quant.rolling-research/v1",
        "snapshot": manifest.artifact.to_dict(),
        "frequency": freq.to_contract(),
        "factors": [factor.to_contract()],
        "as_of": AsOfSpec("source_available_at", None).to_contract(),
        "universe": ["A", "B"],
        "diagnostics": {"hac_lags": 2},
    }
    return store, payload, source, FactorCache(tmp_path / "cache.db")


def request(payload):
    return {
        "schema_version": "quant.job-request/v1",
        "kind": "rolling_research",
        "project_id": "synthetic",
        "run_id": "run-1",
        "payload": payload,
        "inputs": [payload["snapshot"]],
        "timeout_seconds": 60,
        "max_attempts": 2,
    }


def test_real_snapshot_repeat_outputs_and_cross_parameter_input_dedup(setup):
    store, payload, source, cache = setup
    adapter = QdkResearchAdapter(store, owner="research:test")
    first = execute_research(payload, adapter=adapter, cache=cache, run_id="first")
    source.write_bytes(b"source changed after immutable snapshot")
    repeated = execute_research(payload, adapter=adapter, cache=cache, run_id="second")
    assert first["outputs"] == repeated["outputs"]
    assert first["result"]["research_id"] == repeated["result"]["research_id"]
    assert repeated["result"]["cache"]["execution_mode"] == "cache_hit"
    assert repeated["result"]["cache"]["engine_rows"] == 0
    assert first["result"]["summary"]["momentum"]["hac"] is not None
    for window in (10, 20):
        variant = deepcopy(payload)
        variant["factors"][0]["window_periods"] = window
        result = execute_research(variant, adapter=adapter, cache=cache, run_id=f"window-{window}")
        assert result["result"]["research_id"] != first["result"]["research_id"]
    with sqlite3.connect(cache.path) as db:
        assert db.execute("SELECT count(*) FROM factor_cache_v1").fetchone()[0] == 3
        assert db.execute("SELECT count(*) FROM factor_cache_inputs_v1").fetchone()[0] == 1
    for ref in first["outputs"]:
        store.verify(ArtifactRef.from_dict(ref))
        with pytest.raises(ValidationError):
            store.delete(ArtifactRef.from_dict(ref))
    report = json.loads(store.read_bytes(ArtifactRef.from_dict(first["outputs"][1])))
    assert report["factor_identity"]["inputs"][0]["sha256"] in {
        payload["snapshot"]["sha256"],
        SnapshotStore(store)
        .load(ArtifactRef.from_dict(payload["snapshot"]))
        .document["members"][0]["artifact"]["sha256"],
    }
    table = pa.ipc.open_stream(
        store.read_bytes(ArtifactRef.from_dict(first["outputs"][0]))
    ).read_all()
    assert table.num_rows == 120


def test_existing_registry_records_failed_and_retried_attempts(setup, tmp_path):
    store, payload, _, cache = setup
    registry = TrialRegistry(tmp_path / "trials.db")
    register_research(registry, "study", hypothesis="Synthetic test", payloads=[payload])
    payload["study_id"] = "study"

    class InterruptedAdapter(QdkResearchAdapter):
        def publish(self, value, *, media_type):
            raise OSError("injected output failure")

    with pytest.raises(OSError, match="injected"):
        execute_research(
            payload,
            adapter=InterruptedAdapter(store, owner="attempt-failed"),
            cache=cache,
            run_id="failed",
            registry=registry,
        )
    result = execute_research(
        payload,
        adapter=QdkResearchAdapter(store, owner="attempt-success"),
        cache=cache,
        run_id="retry",
        registry=registry,
    )
    assert [e["status"] for e in registry.history("study")] == [
        "running",
        "failed",
        "running",
        "completed",
    ]
    assert result["result"]["trial_attempt_id"]
    assert result["result"]["cache"]["execution_mode"] == "cache_hit"
    altered = deepcopy(payload)
    altered["factors"][0]["window_periods"] = 20
    with pytest.raises(ValueError, match="preregistered"):
        execute_research(
            altered,
            adapter=QdkResearchAdapter(store, owner="bad"),
            cache=cache,
            run_id="bad",
            registry=registry,
        )


def test_real_cli_process_and_environment_only_paths(setup, tmp_path, monkeypatch):
    store, payload, _, cache = setup
    job_path = tmp_path / "request.json"
    job_path.write_text(json.dumps(request(payload)), encoding="utf-8")
    monkeypatch.setenv("QUANT_RESEARCH_STORE_ROOT", str(store.root))
    monkeypatch.setenv("QUANT_RESEARCH_CACHE_DB", str(cache.path))
    monkeypatch.delenv("QUANT_RESEARCH_REGISTRY_DB", raising=False)
    command = [
        sys.executable,
        "-m",
        "quant_lab.rolling_research",
        "--request",
        str(job_path),
        "--workspace",
        str(tmp_path / "first-attempt"),
    ]
    first = json.loads(
        subprocess.run(
            command, check=True, capture_output=True, text=True, env=os.environ.copy()
        ).stdout
    )
    command[-1] = str(tmp_path / "second-attempt")
    second = json.loads(
        subprocess.run(
            command, check=True, capture_output=True, text=True, env=os.environ.copy()
        ).stdout
    )
    assert first["outputs"] == second["outputs"]
    assert second["result"]["cache"]["execution_mode"] == "cache_hit"
    invalid = deepcopy(payload)
    invalid["store_root"] = "C:/arbitrary"
    with pytest.raises(ValueError, match="fields"):
        worker_handler(request(invalid), tmp_path / "first-attempt")
    no_input = request(payload)
    no_input["inputs"] = []
    with pytest.raises(ValueError, match="Job inputs"):
        worker_handler(no_input, tmp_path / "first-attempt")


def test_worker_context_and_pin_owner_are_bound_to_runtime_attempt(setup, tmp_path, monkeypatch):
    store, payload, _, cache = setup
    registry = TrialRegistry(tmp_path / "trials.db")
    register_research(registry, "study", hypothesis="fixture", payloads=[payload])
    payload["study_id"] = "study"
    context = {
        "schema_version": "quant.attempt-context/v1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "worker_id": "worker-1",
        "fencing_token": 2,
        "project_id": "synthetic",
        "run_id": "run-1",
    }
    (tmp_path / "context.json").write_text(json.dumps(context), encoding="utf-8")
    monkeypatch.setenv("QUANT_RESEARCH_STORE_ROOT", str(store.root))
    monkeypatch.setenv("QUANT_RESEARCH_CACHE_DB", str(cache.path))
    monkeypatch.setenv("QUANT_RESEARCH_REGISTRY_DB", str(registry.path))
    result = worker_handler(request(payload), tmp_path)
    assert registry.history("study")[0]["payload"]["context"] == context
    with sqlite3.connect(store.root / "index.sqlite3") as db:
        protected = {
            r[0]
            for r in db.execute(
                "SELECT sha FROM pins WHERE owner=?", ("job/job-1/attempt/attempt-1/outputs",)
            )
        }
    assert all(ref["sha256"] in protected for ref in result["outputs"])
    context["run_id"] = "wrong"
    (tmp_path / "context.json").write_text(json.dumps(context), encoding="utf-8")
    with pytest.raises(ValueError, match="Attempt context"):
        worker_handler(request(payload), tmp_path)


def test_injected_adapter_must_bind_snapshot_and_real_source_tampering_rejected(setup):
    store, payload, _, cache = setup

    class BadResolver(QdkResearchAdapter):
        def resolve(self, *args, **kwargs):
            resolved = super().resolve(*args, **kwargs)
            return ResolvedMarket(resolved.table, [])

    with pytest.raises(ValueError, match="bind the requested"):
        execute_research(
            payload, adapter=BadResolver(store, owner="bad"), cache=cache, run_id="bad"
        )
    bad = deepcopy(payload)
    bad["snapshot"]["uri"] = "file:///tmp/secret"
    with pytest.raises(ValidationError):
        execute_research(
            bad, adapter=QdkResearchAdapter(store, owner="bad"), cache=cache, run_id="bad"
        )


def test_composable_steps_training_summary_and_insufficient_observations(setup):
    store, payload, _, cache = setup
    plan = validate_payload(payload)
    plan["training"] = {
        "start_ns": str(pd.Timestamp("2024-02-10", tz="UTC").value),
        "end_ns": str(pd.Timestamp("2024-02-12", tz="UTC").value),
        "fold_id": "fold",
        "state_sha256": None,
    }
    plan["as_of"] = {"mode": "fixed", "fixed_at_ns": plan["training"]["end_ns"]}
    plan = validate_payload(plan)
    computed = compute_research(
        plan, adapter=QdkResearchAdapter(store, owner="train"), cache=cache, run_id="train"
    )
    summary = summarize_factors(computed.table, plan)
    assert summary["momentum"]["total_rows"] == 6
    assert summary["momentum"]["observation_count"] == 3
    plan["diagnostics"]["hac_lags"] = 3
    summary = summarize_factors(computed.table, plan)
    assert summary["momentum"]["status"] == "insufficient_observations"
    assert summary["momentum"]["hac"] is None


@pytest.mark.parametrize(
    "change", ["field", "schema", "factors", "universe", "diagnostics", "study", "unsupported"]
)
def test_bad_plans_rejected(setup, change):
    _, payload, _, _ = setup
    if change == "field":
        payload["path"] = "/anything"
    elif change == "schema":
        payload["schema_version"] = "invalid"
    elif change == "factors":
        payload["factors"] = []
    elif change == "universe":
        payload["universe"] = ["A", "A"]
    elif change == "diagnostics":
        payload["diagnostics"] = {"hac_lags": True}
    elif change == "study":
        payload["study_id"] = ""
    else:
        payload["factors"][0]["algorithm_id"] = "last_value"
    with pytest.raises(ValueError):
        validate_payload(payload)


def test_registry_configuration_must_be_explicit(setup, tmp_path):
    store, payload, _, cache = setup
    adapter = QdkResearchAdapter(store, owner="test")
    with pytest.raises(ValueError, match="preregistered study_id"):
        execute_research(
            payload,
            adapter=adapter,
            cache=cache,
            run_id="bad",
            registry=TrialRegistry(tmp_path / "trials"),
        )
    payload["study_id"] = "study"
    with pytest.raises(ValueError, match="requires a TrialRegistry"):
        execute_research(payload, adapter=adapter, cache=cache, run_id="bad")
