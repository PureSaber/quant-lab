"""Strict, immutable ``standard/v2`` research run contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_lab.contracts import RunManifest, load_and_validate_run

SCHEMA_VERSION_V2 = "2.0.0"
RESEARCH_PROFILE = "research"
BACKTEST_LEDGER_PROFILE = "backtest-ledger"

ARTIFACT_SCHEMAS_V2: dict[str, tuple[str, ...]] = {
    "returns": (
        "event_time",
        "strategy_id",
        "gross_return",
        "net_return",
        "nav",
        "base_currency",
    ),
    "positions": (
        "event_time",
        "account_id",
        "strategy_id",
        "instrument_id",
        "quantity_units",
        "quantity_scale",
        "mark_price_units",
        "mark_price_scale",
        "market_value_units",
        "market_value_scale",
        "currency",
    ),
    "valuations": (
        "event_time",
        "account_id",
        "nav_units",
        "nav_scale",
        "cash_value_units",
        "cash_value_scale",
        "unrealized_pnl_units",
        "unrealized_pnl_scale",
        "realized_pnl_units",
        "realized_pnl_scale",
        "base_currency",
    ),
    "exposures": (
        "event_time",
        "account_id",
        "strategy_id",
        "exposure_type",
        "name",
        "value",
        "unit",
    ),
    "orders": (
        "event_time",
        "order_id",
        "account_id",
        "strategy_id",
        "instrument_id",
        "side",
        "quantity_units",
        "quantity_scale",
        "order_type",
        "status",
    ),
    "order_events": (
        "event_time",
        "order_id",
        "event_sequence",
        "status",
        "reason",
    ),
    "fills": (
        "event_time",
        "fill_id",
        "order_id",
        "instrument_id",
        "quantity_units",
        "quantity_scale",
        "price_units",
        "price_scale",
        "currency",
    ),
    "costs": (
        "event_time",
        "account_id",
        "strategy_id",
        "instrument_id",
        "cost_type",
        "amount_units",
        "amount_scale",
        "currency",
    ),
    "cash": (
        "event_time",
        "account_id",
        "currency",
        "balance_units",
        "balance_scale",
    ),
    "margin": (
        "event_time",
        "account_id",
        "instrument_id",
        "initial_margin_units",
        "maintenance_margin_units",
        "margin_scale",
        "currency",
    ),
}

PROFILE_ARTIFACTS_V2: dict[str, tuple[str, ...]] = {
    RESEARCH_PROFILE: ("returns", "positions", "valuations", "exposures"),
    BACKTEST_LEDGER_PROFILE: tuple(ARTIFACT_SCHEMAS_V2),
}

_INTEGER_COLUMNS = {
    column
    for columns in ARTIFACT_SCHEMAS_V2.values()
    for column in columns
    if column.endswith("_units") or column.endswith("_scale")
} | {"event_sequence"}
_SCALE_COLUMNS = {column for column in _INTEGER_COLUMNS if column.endswith("_scale")}
_FLOAT_COLUMNS = {"gross_return", "net_return", "nav", "value"}
_CURRENCY_PATTERN = re.compile(r"^[A-Z0-9]{3,12}$")


@dataclass(frozen=True)
class ArtifactRecordV2:
    name: str
    path: str
    schema_id: str
    schema_version: str
    sha256: str
    rows: int
    columns: list[str]
    required: bool
    min_event_time: str | None = None
    max_event_time: str | None = None
    min_available_at: str | None = None
    max_available_at: str | None = None


@dataclass(frozen=True)
class RunManifestV2:
    schema_version: str
    project: str
    run_id: str
    strategy_ids: list[str]
    profile: str
    created_at: str
    status: str
    code_version: str
    internal_dependencies: dict[str, str]
    config_sha256: str
    random_seed: int
    dataset_snapshots: dict[str, str]
    instrument_master_version: str
    execution_model_version: str
    base_currency: str
    capabilities: list[str]
    time_range: dict[str, str | None]
    lineage: dict[str, list[str]]
    artifacts: list[ArtifactRecordV2] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_string_map(value: Mapping[str, str], field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    result: dict[str, str] = {}
    for key, item in value.items():
        result[_require_text(key, f"{field_name} key")] = _require_text(
            item, f"{field_name}[{key!r}]"
        )
    return result


def _validate_currency(value: str, field_name: str = "base_currency") -> str:
    value = _require_text(value, field_name)
    if not _CURRENCY_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be an uppercase ISO currency or stablecoin code")
    return value


def _validate_created_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    try:
        parsed = datetime.fromisoformat(_require_text(value, "created_at"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at must be UTC-aware")
    return parsed.isoformat()


def _validate_lineage(
    lineage: Mapping[str, Sequence[str]], artifact_names: set[str]
) -> dict[str, list[str]]:
    if not isinstance(lineage, Mapping):
        raise ValueError("lineage must be a mapping")
    normalized: dict[str, list[str]] = {}
    for node, sources in lineage.items():
        node = _require_text(node, "lineage node")
        if node not in artifact_names:
            raise ValueError(f"Lineage node is not a declared artifact: {node}")
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
            raise ValueError(f"Lineage sources for {node} must be a sequence")
        clean_sources = [_require_text(source, f"lineage[{node}]") for source in sources]
        if len(clean_sources) != len(set(clean_sources)):
            raise ValueError(f"Lineage sources contain duplicates: {node}")
        normalized[node] = clean_sources
    if set(normalized) != artifact_names:
        raise ValueError(
            "Lineage must declare every artifact: "
            f"missing={sorted(artifact_names - set(normalized))}"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"Lineage graph contains a cycle at {node}")
        if node in visited:
            return
        visiting.add(node)
        for source in normalized.get(node, []):
            if source in artifact_names:
                visit(source)
        visiting.remove(node)
        visited.add(node)

    for node in normalized:
        visit(node)
    return normalized


def _validate_utc_series(series: pd.Series, artifact_name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"Artifact {artifact_name} contains null event_time")
    if series.empty:
        return pd.Series(pd.to_datetime(series, utc=True), index=series.index, name=series.name)
    parsed_values: list[pd.Timestamp] = []
    for value in series:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError(f"Artifact {artifact_name} contains timezone-naive event_time")
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp.to_pydatetime()):
            raise ValueError(f"Artifact {artifact_name} event_time must be stored as UTC")
        parsed_values.append(timestamp.tz_convert("UTC"))
    return pd.Series(parsed_values, index=series.index, name=series.name, dtype="datetime64[ns, UTC]")


def _prepare_frame(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    expected = list(ARTIFACT_SCHEMAS_V2[name])
    actual = list(frame.columns)
    if actual != expected:
        raise ValueError(f"Artifact {name} columns must exactly match {expected}; got {actual}")
    result = frame.copy()
    if result[expected].isna().any().any():
        raise ValueError(f"Artifact {name} contains null required values")
    result["event_time"] = _validate_utc_series(result["event_time"], name)

    for column in expected:
        if column in _INTEGER_COLUMNS:
            numeric = pd.to_numeric(result[column], errors="raise")
            if not numeric.empty and not numeric.map(lambda value: float(value).is_integer()).all():
                raise ValueError(f"Artifact {name}.{column} must contain integers")
            result[column] = numeric.astype("int64")
        elif column in _FLOAT_COLUMNS:
            numeric = pd.to_numeric(result[column], errors="raise").astype("float64")
            if not numeric.map(math.isfinite).all():
                raise ValueError(f"Artifact {name}.{column} contains NaN or infinity")
            result[column] = numeric
        elif column != "event_time":
            if not result[column].map(lambda value: isinstance(value, str) and bool(value)).all():
                raise ValueError(f"Artifact {name}.{column} must contain non-empty strings")
            result[column] = result[column].astype("string")

    for column in _SCALE_COLUMNS.intersection(expected):
        if not result[column].between(0, 18).all():
            raise ValueError(f"Artifact {name}.{column} must be between 0 and 18")
    for column in ("currency", "base_currency"):
        if column in result:
            for value in result[column].unique():
                _validate_currency(str(value), f"{name}.{column}")
    return result


def _arrow_type(column: str) -> pa.DataType:
    if column == "event_time":
        return pa.timestamp("ns", tz="UTC")
    if column in _INTEGER_COLUMNS:
        return pa.int64()
    if column in _FLOAT_COLUMNS:
        return pa.float64()
    return pa.string()


def _arrow_schema(name: str) -> pa.Schema:
    return pa.schema(
        [pa.field(column, _arrow_type(column), nullable=False) for column in ARTIFACT_SCHEMAS_V2[name]],
        metadata={
            b"schema_id": f"puresaber.run.{name}".encode("ascii"),
            b"schema_version": SCHEMA_VERSION_V2.encode("ascii"),
        },
    )


def _parquet_record(
    name: str,
    path: Path,
    frame: pd.DataFrame,
    *,
    required: bool,
) -> ArtifactRecordV2:
    minimum = frame["event_time"].min().isoformat() if not frame.empty else None
    maximum = frame["event_time"].max().isoformat() if not frame.empty else None
    return ArtifactRecordV2(
        name=name,
        path=f"artifacts/{name}.parquet",
        schema_id=f"puresaber.run.{name}",
        schema_version=SCHEMA_VERSION_V2,
        sha256=_file_sha256(path),
        rows=len(frame),
        columns=list(frame.columns),
        required=required,
        min_event_time=minimum,
        max_event_time=maximum,
    )


def _json_record(
    name: str,
    path: Path,
    payload: Mapping[str, Any],
) -> ArtifactRecordV2:
    return ArtifactRecordV2(
        name=name,
        path=f"{name}.json",
        schema_id=f"puresaber.run.{name}",
        schema_version=SCHEMA_VERSION_V2,
        sha256=_file_sha256(path),
        rows=1,
        columns=sorted(payload),
        required=True,
    )


def _manifest_time_range(records: Sequence[ArtifactRecordV2]) -> dict[str, str | None]:
    starts = [record.min_event_time for record in records if record.min_event_time is not None]
    ends = [record.max_event_time for record in records if record.max_event_time is not None]
    return {"start": min(starts) if starts else None, "end": max(ends) if ends else None}


def write_standard_run_v2(
    run_dir: Path,
    *,
    project: str,
    run_id: str,
    strategy_ids: Sequence[str],
    profile: str,
    frames: Mapping[str, pd.DataFrame],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
    code_version: str,
    internal_dependencies: Mapping[str, str],
    random_seed: int,
    dataset_snapshots: Mapping[str, str],
    instrument_master_version: str,
    execution_model_version: str,
    base_currency: str,
    lineage: Mapping[str, Sequence[str]],
    capabilities: Sequence[str] | None = None,
    tags: Mapping[str, str] | None = None,
    created_at: str | None = None,
) -> RunManifestV2:
    """Write one complete ``standard/v2`` directory with atomic publication."""
    if profile not in PROFILE_ARTIFACTS_V2:
        raise ValueError(f"Unsupported v2 profile: {profile}")
    required_frames = set(PROFILE_ARTIFACTS_V2[profile])
    supplied_frames = set(frames)
    unknown_frames = supplied_frames - set(ARTIFACT_SCHEMAS_V2)
    missing_frames = required_frames - supplied_frames
    if unknown_frames or missing_frames:
        raise ValueError(
            f"v2 frames mismatch: missing={sorted(missing_frames)}, extra={sorted(unknown_frames)}"
        )
    if not isinstance(metrics, Mapping) or not isinstance(config, Mapping):
        raise ValueError("metrics and config must be mappings")
    strategy_list = [_require_text(item, "strategy_id") for item in strategy_ids]
    if not strategy_list or len(strategy_list) != len(set(strategy_list)):
        raise ValueError("strategy_ids must be non-empty and unique")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    project = _require_text(project, "project")
    run_id = _require_text(run_id, "run_id")
    code_version = _require_text(code_version, "code_version")
    internal_dependency_map = _validate_string_map(
        internal_dependencies, "internal_dependencies"
    )
    snapshot_map = _validate_string_map(dataset_snapshots, "dataset_snapshots")
    if not snapshot_map:
        raise ValueError("dataset_snapshots must identify at least one immutable snapshot")
    instrument_master_version = _require_text(
        instrument_master_version, "instrument_master_version"
    )
    execution_model_version = _require_text(
        execution_model_version, "execution_model_version"
    )
    base_currency = _validate_currency(base_currency)
    capability_list = sorted(
        {_require_text(item, "capability") for item in (capabilities or [])}
    )
    tag_map = _validate_string_map(tags or {}, "tags")
    created_at_value = _validate_created_at(created_at)

    standard_dir = Path(run_dir) / "standard"
    final_dir = standard_dir / "v2"
    if final_dir.exists():
        raise FileExistsError(f"standard/v2 run is immutable and already exists: {final_dir}")
    standard_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = standard_dir / f".v2-tmp-{uuid4().hex}"
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()
        config_payload = dict(config)
        metrics_payload = dict(metrics)
        config_path = temp_dir / "config.json"
        metrics_path = temp_dir / "metrics.json"
        _write_json(config_path, config_payload)
        _write_json(metrics_path, metrics_payload)
        records = [
            _json_record("config", config_path, config_payload),
            _json_record("metrics", metrics_path, metrics_payload),
        ]

        for name in ARTIFACT_SCHEMAS_V2:
            if name not in frames:
                continue
            prepared = _prepare_frame(name, frames[name])
            path = artifacts_dir / f"{name}.parquet"
            table = pa.Table.from_pandas(
                prepared,
                schema=_arrow_schema(name),
                preserve_index=False,
                safe=True,
            )
            pq.write_table(table, path)
            records.append(
                _parquet_record(name, path, prepared, required=name in required_frames)
            )

        artifact_names = {record.name for record in records}
        lineage_map = _validate_lineage(lineage, artifact_names)
        manifest = RunManifestV2(
            schema_version=SCHEMA_VERSION_V2,
            project=project,
            run_id=run_id,
            strategy_ids=strategy_list,
            profile=profile,
            created_at=created_at_value,
            status="complete",
            code_version=code_version,
            internal_dependencies=internal_dependency_map,
            config_sha256=_json_sha256(config_payload),
            random_seed=random_seed,
            dataset_snapshots=snapshot_map,
            instrument_master_version=instrument_master_version,
            execution_model_version=execution_model_version,
            base_currency=base_currency,
            capabilities=capability_list,
            time_range=_manifest_time_range(records),
            lineage=lineage_map,
            artifacts=records,
            tags=tag_map,
        )
        manifest_path = temp_dir / "run_manifest.json"
        _write_json(manifest_path, asdict(manifest))
        (temp_dir / "run_manifest.sha256").write_text(
            _file_sha256(manifest_path) + "\n", encoding="ascii"
        )
        temp_dir.rename(final_dir)
        return manifest
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def _parse_manifest(path: Path) -> RunManifestV2:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifacts"] = [
            ArtifactRecordV2(**record) for record in payload.get("artifacts", [])
        ]
        return RunManifestV2(**payload)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Invalid standard/v2 manifest: {path}") from exc


def _safe_artifact_path(base: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"Invalid artifact path: {relative!r}")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Artifact path escapes standard/v2: {relative}")
    path = base / relative_path
    try:
        path.resolve(strict=False).relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(f"Artifact path escapes standard/v2: {relative}") from exc
    current = path
    while current != base:
        if current.is_symlink():
            raise ValueError(f"Symlinks are forbidden in standard/v2: {relative}")
        current = current.parent
    return path


def _validate_parquet_artifact(base: Path, record: ArtifactRecordV2) -> None:
    path = _safe_artifact_path(base, record.path)
    if not path.is_file() or _file_sha256(path) != record.sha256:
        raise ValueError(f"Artifact missing or mutated: {path}")
    expected_columns = list(ARTIFACT_SCHEMAS_V2[record.name])
    if record.columns != expected_columns:
        raise ValueError(f"Manifest columns invalid for artifact {record.name}")
    table = pq.read_table(path)
    if table.column_names != expected_columns or table.num_rows != record.rows:
        raise ValueError(f"Artifact metadata mismatch: {record.name}")
    expected_schema = _arrow_schema(record.name)
    for arrow_field, expected_field in zip(table.schema, expected_schema, strict=True):
        if arrow_field.type != expected_field.type or arrow_field.nullable:
            raise ValueError(
                f"Artifact {record.name}.{arrow_field.name} has invalid Arrow field "
                f"{arrow_field}"
            )
    metadata = table.schema.metadata or {}
    if metadata.get(b"schema_id") != f"puresaber.run.{record.name}".encode("ascii"):
        raise ValueError(f"Artifact schema metadata mismatch: {record.name}")
    if metadata.get(b"schema_version") != SCHEMA_VERSION_V2.encode("ascii"):
        raise ValueError(f"Artifact schema version metadata mismatch: {record.name}")
    frame = table.to_pandas()
    prepared = _prepare_frame(record.name, frame)
    minimum = prepared["event_time"].min().isoformat() if not prepared.empty else None
    maximum = prepared["event_time"].max().isoformat() if not prepared.empty else None
    if minimum != record.min_event_time or maximum != record.max_event_time:
        raise ValueError(f"Artifact time range mismatch: {record.name}")


def _validate_json_artifact(
    base: Path, record: ArtifactRecordV2
) -> dict[str, Any]:
    path = _safe_artifact_path(base, record.path)
    if not path.is_file() or _file_sha256(path) != record.sha256:
        raise ValueError(f"Artifact missing or mutated: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Artifact is not valid JSON: {record.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact {record.name} must contain a JSON object")
    if record.rows != 1 or record.columns != sorted(payload):
        raise ValueError(f"Artifact metadata mismatch: {record.name}")
    if any(
        value is not None
        for value in (
            record.min_event_time,
            record.max_event_time,
            record.min_available_at,
            record.max_available_at,
        )
    ):
        raise ValueError(f"JSON artifact cannot declare event bounds: {record.name}")
    return payload


def load_and_validate_run_v2(run_dir: Path) -> RunManifestV2:
    """Load ``standard/v2`` and verify its manifest, files, schemas and lineage."""
    base = Path(run_dir) / "standard" / "v2"
    manifest_path = base / "run_manifest.json"
    checksum_path = base / "run_manifest.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"standard/v2 manifest missing: {manifest_path}")
    if not checksum_path.is_file():
        raise ValueError(f"standard/v2 manifest checksum missing: {checksum_path}")
    if checksum_path.read_text(encoding="ascii").strip() != _file_sha256(manifest_path):
        raise ValueError("standard/v2 manifest is mutated")

    manifest = _parse_manifest(manifest_path)
    if manifest.schema_version != SCHEMA_VERSION_V2:
        raise ValueError(
            f"Unsupported run schema {manifest.schema_version}; expected {SCHEMA_VERSION_V2}"
        )
    if manifest.profile not in PROFILE_ARTIFACTS_V2:
        raise ValueError(f"Unsupported v2 profile: {manifest.profile}")
    _require_text(manifest.project, "project")
    _require_text(manifest.run_id, "run_id")
    _require_text(manifest.code_version, "code_version")
    if manifest.status != "complete":
        raise ValueError("Only complete standard/v2 runs are consumable")
    _validate_created_at(manifest.created_at)
    if not manifest.strategy_ids or len(manifest.strategy_ids) != len(set(manifest.strategy_ids)):
        raise ValueError("strategy_ids must be non-empty and unique")
    for strategy_id in manifest.strategy_ids:
        _require_text(strategy_id, "strategy_id")
    if isinstance(manifest.random_seed, bool) or not isinstance(manifest.random_seed, int):
        raise ValueError("random_seed must be an integer")
    _validate_string_map(manifest.internal_dependencies, "internal_dependencies")
    snapshots = _validate_string_map(manifest.dataset_snapshots, "dataset_snapshots")
    if not snapshots:
        raise ValueError("dataset_snapshots must not be empty")
    _require_text(manifest.instrument_master_version, "instrument_master_version")
    _require_text(manifest.execution_model_version, "execution_model_version")
    _validate_currency(manifest.base_currency)
    if manifest.capabilities != sorted(set(manifest.capabilities)):
        raise ValueError("capabilities must be sorted and unique")
    for capability in manifest.capabilities:
        _require_text(capability, "capability")
    _validate_string_map(manifest.tags, "tags")

    records_by_name: dict[str, ArtifactRecordV2] = {}
    paths: set[str] = set()
    for record in manifest.artifacts:
        if record.name in records_by_name:
            raise ValueError(f"Duplicate artifact name: {record.name}")
        if record.path in paths:
            raise ValueError(f"Duplicate artifact path: {record.path}")
        records_by_name[record.name] = record
        paths.add(record.path)

    expected_required = {"config", "metrics", *PROFILE_ARTIFACTS_V2[manifest.profile]}
    allowed = {"config", "metrics", *ARTIFACT_SCHEMAS_V2}
    actual = set(records_by_name)
    if not expected_required.issubset(actual) or not actual.issubset(allowed):
        raise ValueError(
            "Run artifacts mismatch: "
            f"missing={sorted(expected_required - actual)}, extra={sorted(actual - allowed)}"
        )
    for name, record in records_by_name.items():
        if record.schema_id != f"puresaber.run.{name}":
            raise ValueError(f"Artifact schema id mismatch: {name}")
        if record.schema_version != SCHEMA_VERSION_V2:
            raise ValueError(f"Artifact schema version mismatch: {name}")
        if record.required != (name in expected_required):
            raise ValueError(f"Artifact required flag mismatch: {name}")
        if name in ARTIFACT_SCHEMAS_V2:
            _validate_parquet_artifact(base, record)

    config_payload = _validate_json_artifact(base, records_by_name["config"])
    _validate_json_artifact(base, records_by_name["metrics"])
    if _json_sha256(config_payload) != manifest.config_sha256:
        raise ValueError("config_sha256 does not match config.json")

    lineage = _validate_lineage(manifest.lineage, actual)
    if lineage != manifest.lineage:
        raise ValueError("lineage is not normalized")
    expected_time_range = _manifest_time_range(manifest.artifacts)
    if manifest.time_range != expected_time_range:
        raise ValueError("Manifest time range does not match artifacts")

    expected_files = {
        "run_manifest.json",
        "run_manifest.sha256",
        *paths,
    }
    actual_files = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ValueError(
            "standard/v2 file set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return manifest


def load_and_validate_standard_run(run_dir: Path) -> RunManifestV2 | RunManifest:
    """Prefer v2; fall back to v1 only when the v2 directory is absent."""
    run_dir = Path(run_dir)
    if (run_dir / "standard" / "v2").exists():
        return load_and_validate_run_v2(run_dir)
    return load_and_validate_run(run_dir)
