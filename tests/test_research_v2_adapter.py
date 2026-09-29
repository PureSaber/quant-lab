from dataclasses import replace

import pandas as pd
import pytest

from quant_lab.ranking import is_rankable
from quant_lab.research import compare_results
from quant_lab.research_v2 import write_exploratory_run_v2
from quant_lab.scanner import scan_run


def test_exploratory_adapter_is_valid_and_not_rankable(tmp_path):
    run = tmp_path / "hk-demo"
    run.mkdir()
    manifest = write_exploratory_run_v2(
        run,
        project="quant-hk-equity",
        run_id="hk-demo",
        strategy_id="momentum",
        currency="HKD",
        code_version="a" * 40,
        dataset_snapshots={"hk": "snapshot"},
        nav=pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-05"],
                "nav": [100.0, 101.0],
                "net_return": [0.0, 0.01],
                "gross_return": [0.0, 0.012],
            }
        ),
        positions=pd.DataFrame(
            {
                "date": ["2026-01-05"],
                "instrument_id": ["00700"],
                "quantity": [100],
                "mark_price": [320.5],
            }
        ),
        comparability="current_watchlist_not_historical_universe",
    )
    assert manifest.profile == "research"
    assert manifest.tags["rankable"] == "false"
    scanned = scan_run(run)
    assert scanned.metrics["rankable"] is False
    assert scanned.metrics["contract"] == "standard/v2"
    assert is_rankable(manifest) is False
    assert is_rankable(replace(manifest, tags={})) is False
    compared = compare_results(
        [
            {
                "status": "completed",
                "candidate": {"name": "a"},
                "rankable": False,
                "contract": "standard/v2",
            },
            {
                "status": "completed",
                "candidate": {"name": "b"},
                "rankable": False,
                "contract": "standard/v1",
            },
        ]
    )
    assert compared["comparable"] is False


def test_git_status_failure_cannot_publish_a_clean_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from quant_lab.research_v2 import clean_git_commit

    (tmp_path / ".git").mkdir()
    results = iter(
        [
            SimpleNamespace(returncode=0, stdout="a" * 40),
            SimpleNamespace(returncode=128, stdout=""),
        ]
    )
    monkeypatch.setattr("quant_lab.research_v2.subprocess.run", lambda *a, **kw: next(results))
    assert clean_git_commit(tmp_path) is None


def test_date_only_summary_does_not_precede_the_us_close():
    from quant_lab.research_v2 import _stamp

    assert _stamp("2026-01-02") == pd.Timestamp("2026-01-02T23:59:59Z")
    assert _stamp("2026-01-02T21:00:00Z") == pd.Timestamp("2026-01-02T21:00:00Z")


@pytest.mark.parametrize(
    "snapshots, nav, message",
    [
        ({}, pd.DataFrame({"date": ["2026-01-02"], "nav": [100]}), "dataset"),
        (
            {"input": "hash"},
            pd.DataFrame([["2026-01-02", 100, 1]], columns=["date", "nav", "nav"]),
            "unique",
        ),
    ],
)
def test_invalid_adapter_inputs_fail_before_writing(tmp_path, snapshots, nav, message):
    with pytest.raises(ValueError, match=message):
        write_exploratory_run_v2(
            tmp_path,
            project="test",
            run_id="test",
            strategy_id="test",
            currency="USD",
            code_version="a" * 40,
            dataset_snapshots=snapshots,
            nav=nav,
            comparability="not_historical_universe",
        )
