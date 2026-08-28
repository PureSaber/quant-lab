from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quant_lab.contracts import write_standard_run
from quant_lab.contracts_v2 import (
    ARTIFACT_SCHEMAS_V2,
    BACKTEST_LEDGER_PROFILE,
    RESEARCH_PROFILE,
    load_and_validate_run_v2,
    load_and_validate_standard_run,
    write_standard_run_v2,
)
from quant_lab.scanner import scan_run


def _sample_value(column: str):
    if column == "event_time":
        return pd.Timestamp("2025-01-02T03:04:05Z")
    if column.endswith("_units"):
        return 10_000
    if column.endswith("_scale"):
        return 2
    if column == "event_sequence":
        return 1
    if column in {"gross_return", "net_return", "nav", "value"}:
        return 1.0
    if column in {"currency", "base_currency"}:
        return "USD"
    return f"sample-{column}"


def _frames(names: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.DataFrame(
            [{column: _sample_value(column) for column in ARTIFACT_SCHEMAS_V2[name]}],
            columns=ARTIFACT_SCHEMAS_V2[name],
        )
        for name in names
    }


def _write_v2(
    run: Path,
    *,
    profile: str = RESEARCH_PROFILE,
    frames: dict[str, pd.DataFrame] | None = None,
):
    selected = frames or _frames(("returns", "positions", "valuations", "exposures"))
    lineage = {name: ["dataset:fixture-v1"] for name in ("config", "metrics", *selected)}
    return write_standard_run_v2(
        run,
        project="demo-project",
        run_id=run.name,
        strategy_ids=["demo-strategy"],
        profile=profile,
        frames=selected,
        metrics={"total_return": 0.1},
        config={"seed": 7, "base_currency": "USD"},
        code_version="abc123",
        internal_dependencies={"quant-data-kit": "v0.4.0"},
        random_seed=7,
        dataset_snapshots={"prices": "sha256:fixture-v1"},
        instrument_master_version="instruments-v1",
        execution_model_version="not-applicable",
        base_currency="USD",
        lineage=lineage,
        capabilities=["daily", "research"],
        tags={"environment": "test"},
        created_at="2025-01-02T00:00:00+00:00",
    )


def test_v2_research_run_is_strict_immutable_and_scannable(tmp_path: Path) -> None:
    run = tmp_path / "r1"
    manifest = _write_v2(run)

    loaded = load_and_validate_run_v2(run)
    assert loaded == manifest
    assert loaded.schema_version == "2.0.0"
    assert loaded.profile == RESEARCH_PROFILE
    assert loaded.time_range == {
        "start": "2025-01-02T03:04:05+00:00",
        "end": "2025-01-02T03:04:05+00:00",
    }
    assert {record.name for record in loaded.artifacts} == {
        "config",
        "metrics",
        "returns",
        "positions",
        "valuations",
        "exposures",
    }

    scanned = scan_run(run)
    assert scanned is not None
    assert scanned.project == "demo-project"
    assert scanned.run_type == "standard_v2_research"
    assert Path(scanned.config_path) == run / "standard" / "v2" / "config.json"

    with pytest.raises(FileExistsError, match="immutable"):
        _write_v2(run)


def test_v2_backtest_ledger_profile_requires_and_validates_every_artifact(
    tmp_path: Path,
) -> None:
    frames = _frames(tuple(ARTIFACT_SCHEMAS_V2))
    manifest = _write_v2(tmp_path / "ledger", profile=BACKTEST_LEDGER_PROFILE, frames=frames)
    assert len(manifest.artifacts) == len(ARTIFACT_SCHEMAS_V2) + 2
    assert all(record.required for record in manifest.artifacts)
    assert load_and_validate_run_v2(tmp_path / "ledger").profile == BACKTEST_LEDGER_PROFILE


def test_v2_never_falls_back_to_v1_when_v2_is_mutated(tmp_path: Path) -> None:
    run = tmp_path / "dual-read"
    write_standard_run(
        run,
        project="legacy",
        run_id="dual-read",
        strategy="legacy",
        frames={},
        metrics={},
        config={},
        code_version="v1",
    )
    _write_v2(run)
    (run / "standard" / "v2" / "metrics.json").write_text(
        json.dumps({"mutated": True}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="mutated"):
        load_and_validate_standard_run(run)


def test_v2_rejects_extra_files_and_artifact_path_escape(tmp_path: Path) -> None:
    extra_run = tmp_path / "extra"
    _write_v2(extra_run)
    (extra_run / "standard" / "v2" / "undeclared.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        load_and_validate_run_v2(extra_run)

    escape_run = tmp_path / "escape"
    _write_v2(escape_run)
    manifest_path = escape_run / "standard" / "v2" / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(item for item in payload["artifacts"] if item["name"] == "returns")["path"] = (
        "../returns.parquet"
    )
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (escape_run / "standard" / "v2" / "run_manifest.sha256").write_text(
        checksum + "\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="escapes"):
        load_and_validate_run_v2(escape_run)


def test_v2_rejects_naive_time_wrong_columns_and_non_finite_values(tmp_path: Path) -> None:
    naive = _frames(("returns", "positions", "valuations", "exposures"))
    naive["returns"]["event_time"] = pd.Timestamp("2025-01-02")
    with pytest.raises(ValueError, match="timezone-naive"):
        _write_v2(tmp_path / "naive", frames=naive)

    wrong_columns = _frames(("returns", "positions", "valuations", "exposures"))
    wrong_columns["returns"]["unexpected"] = "x"
    with pytest.raises(ValueError, match="columns must exactly match"):
        _write_v2(tmp_path / "columns", frames=wrong_columns)

    non_finite = _frames(("returns", "positions", "valuations", "exposures"))
    non_finite["returns"]["net_return"] = float("inf")
    with pytest.raises(ValueError, match="NaN or infinity"):
        _write_v2(tmp_path / "infinity", frames=non_finite)


def test_v2_reader_rejects_changed_arrow_physical_type(tmp_path: Path) -> None:
    run = tmp_path / "wrong-arrow"
    _write_v2(run)
    base = run / "standard" / "v2"
    artifact_path = base / "artifacts" / "returns.parquet"
    frame = pq.read_table(artifact_path).to_pandas()
    frame["strategy_id"] = frame["strategy_id"].astype("string")
    frame.to_parquet(artifact_path, engine="pyarrow", index=False)

    manifest_path = base / "run_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in payload["artifacts"] if item["name"] == "returns")
    record["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (base / "run_manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="invalid Arrow field"):
        load_and_validate_run_v2(run)


def test_v2_artifact_hashes_are_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first = _write_v2(tmp_path / "same")
    second = _write_v2(tmp_path / "other")
    first_hashes = {record.name: record.sha256 for record in first.artifacts}
    second_hashes = {record.name: record.sha256 for record in second.artifacts}
    assert first_hashes == second_hashes


def test_unified_loader_falls_back_only_when_v2_is_absent(tmp_path: Path) -> None:
    write_standard_run(
        tmp_path,
        project="legacy",
        run_id="r1",
        strategy="alpha",
        frames={},
        metrics={},
        config={},
        code_version="v1",
    )
    assert load_and_validate_standard_run(tmp_path).schema_version == "1.0"
