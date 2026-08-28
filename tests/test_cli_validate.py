from __future__ import annotations

import json

from quant_lab.cli import main
from quant_lab.contracts import write_standard_run
from test_contracts_v2 import _write_v2


def test_cli_validates_standard_run(tmp_path, capsys) -> None:
    write_standard_run(
        tmp_path,
        project="demo",
        run_id="r1",
        strategy="alpha",
        frames={},
        metrics={},
        config={},
        code_version="abc",
    )
    assert main(["validate", "--run-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["artifacts"]["returns"]


def test_cli_prefers_and_validates_standard_v2(tmp_path, capsys) -> None:
    _write_v2(tmp_path)
    assert main(["validate", "--run-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "2.0.0"
    assert payload["artifacts"]["portfolio_snapshots"]
