from __future__ import annotations

import json

from quant_lab.cli import main
from quant_lab.contracts import write_standard_run


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
