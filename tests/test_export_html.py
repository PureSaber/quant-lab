from pathlib import Path

from quant_lab.export_html import export_html
from quant_lab.store import ExperimentStore


def test_export_html_writes_file(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    store = ExperimentStore(db)
    store.upsert(
        project="demo",
        run_id="r1",
        run_path=str(tmp_path),
        run_type="equity_backtest",
        metrics={"total_return": 0.1},
    )
    out = export_html(db, tmp_path / "dash.html")
    text = out.read_text(encoding="utf-8")
    assert "demo" in text
    assert "r1" in text
