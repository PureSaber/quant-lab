from pathlib import Path

from quant_lab.export_html import export_html
from quant_lab.store import ExperimentStore


def test_export_html_includes_factor_runs_panel(tmp_path: Path) -> None:
    db = tmp_path / "experiments.db"
    store = ExperimentStore(db)
    store.upsert(
        project="quant-factors",
        run_id="smoke",
        run_path=str(tmp_path),
        run_type="factor_run",
        metrics={"factor_count": 4, "mean_ic": 0.05, "factor_set_hash": "abc"},
    )
    out = export_html(db, tmp_path / "factor_dashboard_smoke.html")
    text = out.read_text(encoding="utf-8")
    assert "Factor Runs" in text
    assert "factor_count" not in text
    assert "factors=4" in text or "4" in text
