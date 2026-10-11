"""Publish a fixed synthetic snapshot and write three Worker job requests.

Run with all three development packages on PYTHONPATH. No market connection.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quant_data_kit.artifacts import ArtifactStore
from quant_data_kit.shared_snapshots import SnapshotStore
from quant_factors.contracts_v2 import AsOfSpec, FactorDependency, FactorSpec, FrequencySpec


def publish_fixture(store: ArtifactStore, output: Path) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol, offset in (("SYNTH-A", 0), ("SYNTH-B", 13)):
        for index in range(120):
            stamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=index)
            rows.append(
                {
                    "instrument_id": symbol,
                    "event_time": stamp,
                    "received_at": stamp,
                    "available_at": stamp,
                    "sequence": index,
                    "event_id": f"{symbol}-{index}",
                    "close": float(100 + offset + index + (index % 7) * 0.25),
                }
            )
    source = output / "synthetic.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    snapshot = SnapshotStore(store).publish(
        {"prices": source},
        dataset="synthetic-rolling-research",
        version="1",
        temporal={
            "calendar_id": "synthetic-daily-v1",
            "timezone": "UTC",
            "availability": "event_time = received_at = available_at; fixed synthetic data",
        },
        owner="fixture:rolling-research-v1",
        date_column="event_time",
        symbol_column="instrument_id",
        available_at_column="available_at",
    )
    frequency = FrequencySpec(
        "daily",
        "session_bar",
        "365",
        "synthetic-daily-v1",
        "v1",
        None,
        "trading_day",
        None,
        None,
        None,
    )
    requests = []
    for window in (5, 10, 20):
        factor = FactorSpec(
            "momentum",
            "1.0.0",
            "momentum",
            "bar",
            [FactorDependency("market", "close", "available_at")],
            window,
            "float64",
            False,
            "null",
        )
        request = {
            "schema_version": "quant.job-request/v1",
            "kind": "rolling_research",
            "project_id": "synthetic-research",
            "run_id": f"window-{window}",
            "payload": {
                "schema_version": "quant.rolling-research/v1",
                "snapshot": snapshot.artifact.to_dict(),
                "frequency": frequency.to_contract(),
                "factors": [factor.to_contract()],
                "as_of": AsOfSpec("source_available_at", None).to_contract(),
                "universe": ["SYNTH-A", "SYNTH-B"],
                "diagnostics": {"hac_lags": 2},
            },
            "inputs": [snapshot.artifact.to_dict()],
            "timeout_seconds": 120,
            "max_attempts": 2,
        }
        (output / f"request-{window}.json").write_text(
            json.dumps(request, indent=2), encoding="utf-8"
        )
        requests.append(request)
    return requests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    requests = publish_fixture(
        ArtifactStore(args.store_root, context={"run_id": "fixture"}), args.output
    )
    print(json.dumps({"snapshot": requests[0]["inputs"][0], "request_count": len(requests)}))
