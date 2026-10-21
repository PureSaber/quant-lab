import json

import pytest

from quant_lab.cli import main
from quant_lab.findings import research_findings
from quant_lab.research import file_hash
from quant_lab.trials import TrialRegistry


def test_findings_include_unstarted_failed_retried_and_tampered_results(tmp_path, capsys):
    path = tmp_path / "trials.db"
    registry = TrialRegistry(path)
    definition = {
        "hypothesis": "not yet proven",
        "parameters": [{"w": 1}, {"w": 2}],
        "code_identity": "abc",
        "selection_rule": "report all",
    }
    registry.register("pending", definition)
    registry.register("retry", definition)
    failed = registry.start("retry", {"w": 1})
    registry.finish(failed, "failed", {"error": "missing data"})
    attempt = registry.start("retry", {"w": 1}, context={"artifact_root": str(tmp_path)})
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "study_id": "retry",
                "attempt_id": attempt,
                "artifacts": {},
                "metrics": {"sharpe": 0.2},
            }
        )
    )
    registry.finish(attempt, "completed", {"result": "result.json", "sha256": file_hash(result)})
    before = path.read_bytes()
    report = research_findings(path)
    assert path.read_bytes() == before
    assert report["studies"][0]["unattempted_candidates"] == 2
    retry = report["studies"][1]
    assert retry["attempt_counts"] == {"failed": 1, "completed": 1}
    assert retry["unattempted_candidates"] == 1
    assert retry["attempts"][-1]["evidence"]["metrics"] == {"sharpe": 0.2}
    assert main(["findings", "--registry", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report
    assert main(["findings", "--registry", str(path)]) == 0
    assert "failed: 1" in capsys.readouterr().out
    result.write_text("{}")
    evidence = research_findings(path)["studies"][1]["attempts"][-1]["evidence"]
    assert evidence["verified"] is False and "metrics" not in evidence


def test_reader_does_not_create_missing_registry(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(FileNotFoundError):
        research_findings(path)
    assert not path.exists()
