import json
import subprocess
import sys
from copy import deepcopy

import pytest
from measurement_helpers import measurement, write_measured

from quant_lab.cli import main
from quant_lab.compare import compare_runs
from quant_lab.measurement import read_measured_run
from quant_lab.research import compare_results
from quant_lab.scanner import scan_run
from quant_lab.store import ExperimentStore
from quant_measurement import (
    BASIS_FIELDS,
    compare_measurements,
    content_id,
    validate_measurement,
    validate_observations,
)


def test_stdlib_only_import():
    script = 'import sys; import quant_measurement; assert not any(x in sys.modules for x in ["numpy","pandas","pyarrow","quant_lab"])'
    subprocess.run([sys.executable, "-c", script], check=True)


def test_complete_contract_and_code_variation():
    value = measurement()
    other = deepcopy(value)
    other["provenance"]["code_identity"]["producer"] = "e" * 40
    assert compare_measurements([value, other])["comparable"]
    assert validate_measurement(value) == value
    validate_observations(
        value, ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z", "2025-01-06T00:00:00Z"]
    )


@pytest.mark.parametrize("field", sorted(BASIS_FIELDS))
def test_every_economic_dimension_is_required_and_changed_basis_is_blocked(field):
    original = measurement()
    changed = deepcopy(original)
    changed["basis"].pop(field)
    assert not compare_measurements([original, changed])["comparable"]
    changed = deepcopy(original)
    changed["basis"][field] = {
        "currency": "USD",
        "monetary_unit": "minor_currency",
        "return_unit": "percent",
        "annualization_periods": 252,
        "period_start": "2025-01-01T00:00:00Z",
        "period_end": "2025-01-07T00:00:00Z",
        "initial_capital": "200",
        "risk_free_rate": ".02",
        "risk_free_unit": "per_period_decimal",
        "risk_free_compounding": "effective",
        "annualization_policy": "fixed_periods",
    }.get(field, "different")
    result = compare_measurements([original, changed])
    assert not result["comparable"] and result["reasons"]


@pytest.mark.parametrize(
    "field,value",
    [("sample_kind", "in_sample"), ("evidence_kind", "historical_pit"), ("unsupported_scopes", [])],
)
def test_evidence_types_and_unsupported_scopes_do_not_mix(field, value):
    original = measurement()
    changed = deepcopy(original)
    changed[field] = value
    assert not compare_measurements([original, changed])["comparable"]


def test_input_identity_missing_failed_single_and_grid_controls():
    value = measurement()
    changed = deepcopy(value)
    changed["provenance"]["input_identity"]["fixture"] = "sha256:" + "f" * 64
    assert not compare_measurements([value, changed])["comparable"]
    for values, statuses in [
        ([value], None),
        ([], None),
        ([None, None], None),
        ([value, value], ["completed", "failed"]),
    ]:
        assert not compare_measurements(values, statuses=statuses)["comparable"]
    with pytest.raises(ValueError):
        compare_measurements([value], statuses=[])
    with pytest.raises(ValueError):
        validate_observations(value, [value["basis"]["period_start"], value["basis"]["period_end"]])
    bad = deepcopy(value)
    bad["basis"].update(annualization_policy="fixed_periods", annualization_periods=252)
    with pytest.raises(ValueError, match="event"):
        validate_measurement(bad)
    bad["basis"]["frequency"] = "daily"
    assert validate_measurement(bad)
    bad["basis"]["risk_free_rate"] = "NaN"
    with pytest.raises(ValueError):
        validate_measurement(bad)


def test_real_index_cli_revalidates_and_preserves_history(tmp_path, capsys):
    store = ExperimentStore(tmp_path / "lab.db")
    for name in ["a", "b"]:
        path = tmp_path / name
        write_measured(path)
        run = scan_run(path)
        store.upsert(
            project=run.project,
            run_id=run.run_id,
            run_path=run.run_path,
            run_type=run.run_type,
            metrics=run.metrics,
        )
    args = [
        "--db",
        str(tmp_path / "lab.db"),
        "compare",
        "--project",
        "measured",
        "a",
        "b",
        "--json",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)[0]["comparison"]["comparable"]
    (tmp_path / "a" / "standard" / "v2" / "metrics.json").write_text("{}")
    assert main(args) == 2
    rows = json.loads(capsys.readouterr().out)
    assert not rows[0]["metrics"] and "mutated" in rows[0]["error"]
    assert main([*args, "--view-only"]) == 0
    assert compare_runs(store, "measured", ["missing"])[0]["error"] == "not found"


def test_native_bridge_policy_grid_and_legacy_unknown(tmp_path):
    value = measurement()
    value["basis"]["cost_policy_id"] = content_id({"commission": "different"})
    write_measured(tmp_path / "bad-policy", value)
    with pytest.raises(ValueError, match="cost policy"):
        read_measured_run(tmp_path / "bad-policy")
    value = measurement()
    value["basis"]["observation_schedule_id"] = "sha256:" + "f" * 64
    write_measured(tmp_path / "bad-grid", value)
    with pytest.raises(ValueError, match="grid"):
        read_measured_run(tmp_path / "bad-grid")
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "capital_curves.csv").write_text("date,v\n2025-01-01,100\n2025-01-02,110\n")
    store = ExperimentStore(tmp_path / "lab.db")
    run = scan_run(legacy)
    store.upsert(
        project=run.project,
        run_id=run.run_id,
        run_path=run.run_path,
        run_type=run.run_type,
        metrics=run.metrics,
    )
    rows = compare_runs(store, run.project, [run.run_id, run.run_id])
    assert rows[0]["metrics"]["total_return"] == 0.1 and not rows[0]["comparison"]["comparable"]


def test_research_comparison_never_omits_failed_or_missing_measurement():
    base = {"status": "completed", "candidate": {"name": "base"}, "measurement": measurement()}
    other = {**base, "candidate": {"name": "other"}}
    assert compare_results([base, other])["comparable"]
    failed = {"status": "failed", "candidate": {"name": "failed"}}
    assert not compare_results([base, other, failed])["comparable"]
    assert len(compare_results([base, failed])["requested_results"]) == 2
    assert not compare_results([{k: v for k, v in base.items() if k != "measurement"}, other])[
        "comparable"
    ]


@pytest.mark.parametrize("artifact", ["metrics", "returns"])
def test_replaced_self_consistent_source_cannot_mix_validated_snapshots(
    tmp_path, monkeypatch, artifact
):
    import shutil

    import quant_lab.measurement as bridge
    from quant_lab.contracts_v2 import load_and_validate_standard_run

    old = tmp_path / "old" / "a"
    new = tmp_path / "new" / "a"
    write_measured(old)
    value = measurement()
    if artifact == "metrics":
        value["basis"]["benchmark_id"] = "different"
    write_measured(new, value, rates=(0.2, -0.2) if artifact == "returns" else (0.1, -0.1))
    original = load_and_validate_standard_run

    def replace_after_validation(path):
        manifest = original(path)
        for source in (new / "standard" / "v2").iterdir():
            shutil.copyfile(source, old / "standard" / "v2" / source.name)
        return manifest

    monkeypatch.setattr(bridge, "load_and_validate_standard_run", replace_after_validation)
    with pytest.raises(ValueError, match="source bytes changed"):
        bridge.read_measured_run(old)


@pytest.mark.parametrize(
    "patch",
    [
        {"schema": "future/v99"},
        {"evidence_kind": "unknown"},
        {"sample_kind": "unknown"},
        {"unsupported_scopes": "none"},
        {"unsupported_scopes": ["a", "a"]},
        {"provenance": {"code_identity": {}, "input_identity": {}}},
        {
            "provenance": {
                "code_identity": {"p": "branch"},
                "input_identity": {"f": "sha256:" + "a" * 64},
            }
        },
    ],
)
def test_malformed_metadata_fails_closed(patch):
    value = measurement()
    value.update(patch)
    assert not compare_measurements([value, measurement()])["comparable"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("currency", "usd"),
        ("initial_capital", "0"),
        ("risk_free_rate", "no-number"),
        ("risk_free_rate", "Infinity"),
        ("period_start", "2025-01-01"),
        ("period_end", "2025-01-01T00:00:00Z"),
        ("flow_policy", "unknown"),
        ("annualization_periods", False),
        ("annualization_periods", 0),
        ("annualization_periods", float("inf")),
    ],
)
def test_invalid_economic_values(field, value):
    obj = measurement()
    if field == "annualization_periods":
        obj["basis"].update(annualization_policy="fixed_periods", frequency="daily")
    obj["basis"][field] = value
    assert not compare_measurements([obj, measurement()])["comparable"]


def test_native_currency_code_and_dataset_links(tmp_path, monkeypatch):
    from dataclasses import replace

    import quant_lab.measurement as bridge

    value = measurement()
    write_measured(tmp_path / "a", value)
    source = bridge.load_and_validate_standard_run
    for patch, error in [
        ({"base_currency": "USD"}, "currency"),
        ({"code_version": "f" * 40}, "code identity"),
        ({"dataset_snapshots": {"fixture": "sha256:" + "e" * 64}}, "input identity"),
    ]:
        monkeypatch.setattr(
            bridge,
            "load_and_validate_standard_run",
            lambda path, p=patch: replace(source(path), **p),
        )
        with pytest.raises(ValueError, match=error):
            bridge.read_measured_run(tmp_path / "a")


def test_native_metrics_are_not_overwritten_by_unhashed_legacy_csv(tmp_path):
    write_measured(tmp_path / "a")
    (tmp_path / "a" / "capital_curves.csv").write_text(
        "date,wealth\n2025-01-01,1\n2025-01-02,10000\n"
    )
    assert scan_run(tmp_path / "a").metrics["total_return"] == -0.01


def test_ranking_needs_measured_nonsynthetic_ledger_and_explicit_eligibility(tmp_path):
    from dataclasses import replace

    from quant_lab.ranking import is_rankable

    manifest = replace(write_measured(tmp_path / "a"), profile="backtest-ledger", tags={})
    value = measurement()
    value.update(evidence_kind="historical_pit", sample_kind="out_of_sample")
    assert is_rankable(manifest, value)
    assert not is_rankable(manifest)
    assert not is_rankable(manifest, measurement())
    for tags in [
        {"rankable": "false"},
        {"research_only": "true"},
        {"comparability": "watchlist"},
        {"investable": "false"},
    ]:
        assert not is_rankable(replace(manifest, tags=tags), value)
