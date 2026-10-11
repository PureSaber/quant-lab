"""Bridge the lightweight declaration to immutable standard/v2 artifacts."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from quant_lab.contracts_v2 import RunManifestV2, load_and_validate_standard_run
from quant_measurement import content_id, validate_measurement, validate_observations


def read_measured_run(run_dir: Path) -> dict:
    """Verify source, declaration and observed grid; no legacy metadata inference."""
    root = Path(run_dir)
    manifest = load_and_validate_standard_run(root)
    if not isinstance(manifest, RunManifestV2):
        raise TypeError("legacy standard/v1 has no certifying measurement")
    base = root / "standard" / "v2"
    records = {record.name: record for record in manifest.artifacts}

    def verified_bytes(name):
        record = records[name]
        raw = (base / record.path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.sha256:
            raise ValueError(f"{name}: source bytes changed after manifest validation")
        return raw

    metrics = json.loads(verified_bytes("metrics"))
    value = validate_measurement(metrics.get("measurement"))
    basis = value["basis"]
    if basis["currency"] != manifest.base_currency:
        raise ValueError("measurement currency differs from native manifest")
    if manifest.code_version not in value["provenance"]["code_identity"].values():
        raise ValueError("measurement code identity omits producer manifest code")
    if value["provenance"]["input_identity"] != manifest.dataset_snapshots:
        raise ValueError("measurement input identity differs from manifest dataset snapshots")
    config = json.loads(verified_bytes("config"))
    policies = config.get("measurement_policies", {})
    for key in ("cost", "valuation"):
        if key not in policies or content_id(policies[key]) != basis[key + "_policy_id"]:
            raise ValueError(f"measurement {key} policy differs from frozen config")
    returns = pd.read_parquet(BytesIO(verified_bytes("returns")))
    if returns.empty or len(returns.strategy_id.unique()) != 1:
        raise ValueError("measured comparison requires one nonempty strategy return stream")
    grid = [
        basis["period_start"],
        *[stamp.isoformat().replace("+00:00", "Z") for stamp in returns.event_time],
    ]
    validate_observations(value, grid)
    return {"manifest": manifest, "metrics": metrics, "measurement": value, "returns": returns}
