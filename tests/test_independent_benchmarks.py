import json
from decimal import Decimal

import pytest

from quant_benchmarks import main, residuals, run_benchmarks
from quant_benchmarks.oracle import expectations


def test_oracle_independent_reconciliations_and_perturbation():
    oracle = expectations()
    assert Decimal(oracle["corporate_actions"]["final_cash"]) == 1790
    assert Decimal(oracle["futures_roll"]["final_cash"]) == 1191
    assert Decimal(oracle["fof_fees"]["fixed_fee_net"]) == Decimal("1200.288")
    assert Decimal(oracle["fx_transfer"]["pnl_base"]) == 69
    assert not residuals({"nav": "1790"}, {"nav": "1791"})["nav"]["pass"]
    for actual in ["nan", "inf"]:
        with pytest.raises(ValueError):
            residuals({"nav": "1"}, {"nav": actual})
    with pytest.raises(ValueError):
        residuals({"nav": "1"}, {})
    with pytest.raises(ValueError):
        residuals({"nav": "1"}, {"nav": 1}, "-1")


def test_missing_dependency_is_unavailable_not_pass_and_reports_are_immutable(tmp_path, capsys):
    result = run_benchmarks({}, tmp_path / "missing", cases=["corporate_actions"])
    assert result["status"] == "unavailable_or_failed"
    assert result["cases"][0]["status"] == "unavailable"
    assert (tmp_path / "missing" / "report.md").is_file()
    with pytest.raises(FileExistsError):
        run_benchmarks({}, tmp_path / "missing")
    source = tmp_path / "sources.json"
    source.write_text("{}")
    assert (
        main(["--sources", str(source), "--out", str(tmp_path / "cli"), "--case", "fof_fees"]) == 2
    )
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable_or_failed"


@pytest.mark.parametrize(
    "case,module",
    [
        ("corporate_actions", "quant_execution"),
        ("futures_roll", "quant_execution"),
        ("fof_fees", "quant_fund"),
        ("fx_transfer", "quant_report_hub"),
    ],
)
def test_optional_real_adapters_against_independent_scalar_oracle(tmp_path, case, module):
    pytest.importorskip(
        module, reason="real cross-repository adapter unavailable in unit environment"
    )
    from quant_benchmarks import adapters

    observed, controls = getattr(adapters, case)(tmp_path)
    checks = residuals(expectations()[case], observed, "1e-9" if case == "fof_fees" else "0")
    assert all(row["pass"] for row in checks.values())
    assert all(value is True for value in controls.values())


def test_harness_rechecks_source_identity_after_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import quant_benchmarks as harness
    from quant_benchmarks import adapters

    source = tmp_path / "source"
    source.mkdir()
    calls = []

    def git(path, *args):
        if args[0] == "rev-parse":
            return "a" * 40
        calls.append(args)
        return "" if len(calls) == 1 else " M economics.py"

    monkeypatch.setattr(harness, "_git", git)
    monkeypatch.setattr(
        harness.importlib,
        "import_module",
        lambda name: SimpleNamespace(__file__=str(source / "__init__.py")),
    )
    monkeypatch.setattr(
        adapters, "fof_fees", lambda path: (expectations()["fof_fees"], {"negative": True})
    )
    report = run_benchmarks(
        {"quant-fund": {"path": str(source), "commit": "a" * 40}},
        tmp_path / "out",
        cases=["fof_fees"],
    )
    assert report["status"] == "unavailable_or_failed"
    assert report["cases"][0]["status"] == "fail"
    assert not report["sources"]["quant-fund"]["verified_after_execution"]
