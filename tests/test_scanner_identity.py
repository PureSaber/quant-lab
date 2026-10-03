import json

import pandas as pd
import pytest

from quant_lab.contracts import write_standard_run
from quant_lab.research_v2 import write_exploratory_run_v2
from quant_lab.scanner import scan_run
from quant_lab.store import ExperimentStore


def _standard(run, version):
    run.mkdir()
    if version == 1:
        return write_standard_run(
            run,
            project="quant-timing",
            run_id="registered-study",
            strategy="position_only",
            frames={},
            metrics={"total_return": -0.03},
            config={},
            code_version="a" * 40,
        )
    return write_exploratory_run_v2(
        run,
        project="quant-fund",
        run_id="registered-study",
        strategy_id="equal",
        currency="CNY",
        code_version="b" * 40,
        dataset_snapshots={"nav": "sha256-fixture"},
        nav=pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "nav": [100, 97]}),
        comparability="not_historical_universe",
        metrics={"total_return": -0.03},
    )


@pytest.mark.parametrize("version", [1, 2])
def test_standard_identity_comes_from_manifest_not_folder(tmp_path, version):
    run = tmp_path / "local-review-copy"
    manifest = _standard(run, version)

    record = scan_run(run, project="untrusted-folder-hint")
    assert record.run_id == manifest.run_id == "registered-study"
    assert record.project == manifest.project
    assert record.metrics["rankable"] is False
    store = ExperimentStore(tmp_path / "index.db")
    store.upsert(
        project=record.project,
        run_id=record.run_id,
        run_path=record.run_path,
        run_type=record.run_type,
        metrics=record.metrics,
    )
    assert store.get(manifest.project, manifest.run_id).run_path == str(run.resolve())


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "use", "position_scale": 0.3, "research_only": True},
        {"schema_version": "other.decision/v1", "status": "paper_ready"},
    ],
)
def test_other_decision_documents_are_not_paper_cards(tmp_path, payload):
    run = tmp_path / "registered-study"
    _standard(run, 1)
    (run / "decision.json").write_text(json.dumps(payload), encoding="utf-8")

    record = scan_run(run)
    assert record.run_type == "standard_backtest"
    assert record.metrics["total_return"] == -0.03
    assert "decision_status" not in record.metrics


def test_declared_paper_card_preserves_status_and_reasons(tmp_path):
    run = tmp_path / "registered-study"
    run.mkdir()
    payload = {
        "schema_version": "quant.decision/v1",
        "run_id": "registered-study",
        "status": "blocked",
        "reasons": ["missing prices"],
        "validation": {"passed": False},
    }
    (run / "decision.json").write_text(json.dumps(payload), encoding="utf-8")

    record = scan_run(run)
    assert record.run_type == "decision_attempt"
    assert record.metrics["decision_status"] == "blocked"
    assert record.metrics["decision_reasons"] == ["missing prices"]
    assert record.metrics["validation"] == {"passed": False}


@pytest.mark.parametrize(
    "update",
    [{"status": None}, {"status": "approved"}, {"run_id": "another-study"}],
)
def test_invalid_declared_paper_identity_or_status_is_rejected(tmp_path, update):
    run = tmp_path / "local-copy"
    _standard(run, 2)
    card = {
        "schema_version": "quant.decision/v1",
        "run_id": "registered-study",
        "status": "observe",
        **update,
    }
    (run / "decision.json").write_text(json.dumps(card), encoding="utf-8")

    with pytest.raises(ValueError, match="decision identity or status"):
        scan_run(run)
