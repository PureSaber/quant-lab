from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_lab.scanner import scan_run
from quant_lab.store import ExperimentStore


def test_scan_equity_run(tmp_path: Path):
    run = tmp_path / "demo_run"
    run.mkdir()
    pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-02-28"],
            "ols": [10000, 10500],
        }
    ).to_csv(run / "capital_curves.csv", index=False)
    item = scan_run(run, project="a-share-multifactor")
    assert item is not None
    assert item.run_type == "equity_backtest"
    assert item.metrics["total_return"] == 0.05


def test_store_upsert_and_list(tmp_path: Path):
    db = tmp_path / "lab.db"
    store = ExperimentStore(db)
    store.upsert(
        project="demo",
        run_id="r1",
        run_path=str(tmp_path),
        run_type="equity_backtest",
        metrics={"total_return": 0.1},
    )
    rows = store.list_runs("demo")
    assert len(rows) == 1
    assert rows[0].run_id == "r1"
