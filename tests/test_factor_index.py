import json
from pathlib import Path

from quant_lab.scanner import scan_run
from quant_lab.store import ExperimentStore


def test_scan_run_reads_factor_manifest(tmp_path: Path) -> None:
    run = tmp_path / "factor_smoke"
    run.mkdir()
    (run / "factor_manifest.json").write_text(
        json.dumps(
            {
                "factor_set_hash": "abc123",
                "git_sha": "deadbeef",
                "factors": ["momentum_20d", "reversal_5d"],
            }
        ),
        encoding="utf-8",
    )
    scanned = scan_run(run, project="quant-factors")
    assert scanned is not None
    assert scanned.metrics["factor_set_hash"] == "abc123"
    assert scanned.metrics["factor_count"] == 2

    db = tmp_path / "lab.db"
    store = ExperimentStore(db)
    store.upsert(
        project=scanned.project,
        run_id=scanned.run_id,
        run_path=scanned.run_path,
        run_type=scanned.run_type,
        metrics=scanned.metrics,
    )
    rows = store.list_runs(project="quant-factors")
    assert rows[0].metrics_json
    assert "factor_set_hash" in rows[0].metrics_json
