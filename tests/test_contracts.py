import json

import pandas as pd
import pytest

from quant_lab.contracts import load_and_validate_run, write_standard_run
from quant_lab.scanner import scan_outputs_root, scan_run


def test_standard_run_is_complete_immutable_and_scannable(tmp_path) -> None:
    run = tmp_path / "r1"
    run.mkdir()
    returns = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "strategy": ["demo"],
            "gross_return": [0.01],
            "net_return": [0.009],
            "nav": [1.009],
            "benchmark_return": [0.0],
        }
    )
    manifest = write_standard_run(
        run,
        project="demo-project",
        run_id="r1",
        strategy="demo",
        frames={"returns": returns},
        metrics={"total_return": 0.009},
        config={"seed": 1},
        code_version="abc123",
        dataset_snapshots={"prices": "sha256-demo"},
    )
    assert manifest.schema_version == "1.0"
    assert {record.name for record in manifest.artifacts} == {
        "returns",
        "positions",
        "orders",
        "costs",
        "exposures",
        "metrics",
    }
    assert load_and_validate_run(run).code_version == "abc123"
    scanned = scan_run(run)
    assert scanned is not None
    assert scanned.run_type == "standard_backtest"
    assert scanned.metrics["dataset_snapshots"]["prices"] == "sha256-demo"
    assert [item.run_id for item in scan_outputs_root(run)] == ["r1"]

    with pytest.raises(FileExistsError, match="immutable"):
        write_standard_run(
            run,
            project="demo-project",
            run_id="r1",
            strategy="demo",
            frames={"returns": returns},
            metrics={},
            config={},
            code_version="abc123",
        )


def test_standard_run_detects_artifact_mutation(tmp_path) -> None:
    run = tmp_path / "r2"
    run.mkdir()
    write_standard_run(
        run,
        project="demo",
        run_id="r2",
        strategy="demo",
        frames={},
        metrics={},
        config={},
        code_version="abc",
    )
    (run / "standard" / "metrics.json").write_text(json.dumps({"changed": True}))
    with pytest.raises(ValueError, match="mutated"):
        load_and_validate_run(run)
