from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.cli import main
from quant_lab.compare import format_comparison_table
from quant_lab.scanner import (
    _metrics_from_capital_curves,
    _metrics_from_summary,
    _read_yaml,
    detect_project,
    metrics_to_json,
    scan_outputs_root,
    scan_run,
    scan_workspace,
)


def test_cli_lifecycle_covers_governance_commands(tmp_path: Path, capsys) -> None:
    config = tmp_path / "configs" / "default.yaml"
    assert main(["init", "--config", str(config)]) == 0
    assert main(["init", "--config", str(config)]) == 0
    assert main(["init", "--config", str(config), "--force"]) == 0

    root = tmp_path / "outputs" / "run-1"
    root.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "alpha": [100.0, 110.0]}).to_csv(
        root / "capital_curves.csv", index=False
    )
    db = tmp_path / "state" / "experiments.db"
    assert main(["--db", str(db), "scan", "--root", str(root), "--project", "demo"]) == 0
    assert "Indexed 1 run" in capsys.readouterr().out
    assert main(["--db", str(db), "list"]) == 0
    assert "demo\trun-1" in capsys.readouterr().out
    assert main(["--db", str(db), "list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["run_id"] == "run-1"
    assert main(["--db", str(db), "compare", "--project", "demo", "run-1", "missing"]) == 0
    assert "missing" in capsys.readouterr().out
    assert (
        main(["--db", str(db), "compare", "--project", "demo", "run-1", "missing", "--json"]) == 0
    )
    assert json.loads(capsys.readouterr().out)[1]["error"] == "not found"
    dashboard = tmp_path / "reports" / "dashboard.html"
    assert main(["--db", str(db), "export", "html", "--out", str(dashboard)]) == 0
    assert dashboard.is_file()


def test_scanner_governance_paths_and_fallbacks(tmp_path: Path) -> None:
    assert _read_yaml(tmp_path / "missing.yaml") == {}
    missing_curves = tmp_path / "missing.csv"
    try:
        _metrics_from_capital_curves(missing_curves)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing capital curves must fail explicitly")
    no_date = tmp_path / "no-date.csv"
    pd.DataFrame({"value": [1]}).to_csv(no_date, index=False)
    assert _metrics_from_capital_curves(no_date) == {}
    no_values = tmp_path / "no-values.csv"
    pd.DataFrame({"date": ["2025-01-01"]}).to_csv(no_values, index=False)
    assert _metrics_from_capital_curves(no_values) == {}
    zero_start = tmp_path / "zero.csv"
    pd.DataFrame({"date": ["2025-01-01", "2025-01-02"], "value": [0, 1]}).to_csv(
        zero_start, index=False
    )
    assert _metrics_from_capital_curves(zero_start)["total_return"] == 0.0

    assert _metrics_from_summary(tmp_path / "missing-summary.csv") == {}
    empty_summary = tmp_path / "empty-summary.csv"
    empty_summary.write_text("metric,value\n", encoding="utf-8")
    assert _metrics_from_summary(empty_summary) == {}
    summary = tmp_path / "summary.csv"
    summary.write_text("metric,value,description\nreturn,0.1,ok\n", encoding="utf-8")
    assert _metrics_from_summary(summary)["value"] == 0.1
    assert _metrics_from_summary(summary)["description"] == "ok"

    hinted = tmp_path / "anything"
    hinted.mkdir()
    assert detect_project(hinted, "explicit") == "explicit"
    marker_parent = tmp_path / "a-share-multifactor" / "run"
    marker_parent.mkdir(parents=True)
    assert detect_project(marker_parent) == "a-share-multifactor"
    cap = tmp_path / "cap-run"
    cap.mkdir()
    (cap / "capital_curves.csv").write_text("date,value\n2025-01-01,1\n", encoding="utf-8")
    assert detect_project(cap) == "a-share-multifactor"
    fi = tmp_path / "fi-run"
    fi.mkdir()
    (fi / "feature_importance.csv").write_text("feature,value\na,1\n", encoding="utf-8")
    assert detect_project(fi) == "sklearn-stock-trend"
    spread = tmp_path / "spread-run" / "performance"
    spread.mkdir(parents=True)
    (spread / "summary.csv").write_text("metric,value\na,1\n", encoding="utf-8")
    assert detect_project(spread.parent) == "future_spread"
    unknown = tmp_path / "unknown-run"
    unknown.mkdir()
    assert detect_project(unknown) == tmp_path.name
    assert metrics_to_json({"中文": 1}).startswith("{")

    assert scan_run(tmp_path / "not-a-run") is None
    synth_run = tmp_path / "synth-run"
    synth_run.mkdir()
    pd.DataFrame({"left": [1], "right": [2]}).to_csv(
        synth_run / "synthesis_comparison_summary.csv", index=False
    )
    assert scan_run(synth_run).run_type == "multifactor_compare"
    review_run = tmp_path / "review-run"
    review_run.mkdir()
    (review_run / "review_manifest.json").write_text("{}", encoding="utf-8")
    assert scan_run(review_run).run_type == "agent_review"
    bad_factor = tmp_path / "bad-factor-run"
    bad_factor.mkdir()
    (bad_factor / "factor_manifest.json").write_text("not json", encoding="utf-8")
    assert scan_run(bad_factor).metrics["factor_set_hash"] == ""
    config_run = tmp_path / "config-run"
    config_run.mkdir()
    (config_run / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (config_run / "report.html").write_text("<html></html>", encoding="utf-8")
    assert scan_run(config_run).config_path.endswith("config.yaml")

    assert scan_outputs_root(tmp_path / "missing-root") == []
    parent = tmp_path / "workspace-project" / "outputs"
    child = parent / "known"
    child.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-01-01"], "value": [1]}).to_csv(
        child / "capital_curves.csv", index=False
    )
    (parent / ".hidden").mkdir()
    (parent / "unknown").mkdir()
    runs = scan_outputs_root(parent)
    assert [run.run_id for run in runs] == ["known"]
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "report.html").write_text("", encoding="utf-8")
    assert len(scan_outputs_root(direct)) == 1

    workspace = tmp_path / "workspace.yaml"
    workspace.write_text(
        "projects:\n  - name: demo\n    outputs: " + str(child).replace("\\", "/") + "\n",
        encoding="utf-8",
    )
    assert len(scan_workspace(workspace)) == 1


def test_comparison_table_supports_bilingual_metrics() -> None:
    text = format_comparison_table(
        [
            {
                "project": "demo",
                "run_id": "r1",
                "run_type": "backtest",
                "metrics": {"累计收益率": 0.1, "最大回撤": -0.2, "Sharpe": 1.2},
            },
            {"project": "demo", "run_id": "r2", "error": "not found"},
        ]
    )
    assert "0.1" in text and "-0.2" in text and "1.2" in text and "-" in text
