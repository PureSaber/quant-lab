from datetime import datetime, timezone

import pytest

from quant_lab.trials import TrialRegistry


def test_registry_keeps_failed_searches_and_locks_future_holdout(tmp_path):
    registry = TrialRegistry(tmp_path / "experiments.db")
    spec = {
        "hypothesis": "momentum persists",
        "parameters": [{"lookback": 20}, {"lookback": 60}],
        "code_identity": "abc",
        "selection_rule": "register all attempts",
        "holdout_start": "2027-01-01",
    }
    clock = datetime(2026, 9, 19, tzinfo=timezone.utc)
    digest = registry.register("momentum", spec, now=clock)
    assert registry.register("momentum", spec, now=clock) == digest
    failed = registry.start("momentum", {"lookback": 20})
    registry.finish(failed, "failed", {"error": "provider unavailable"})
    registry.start("momentum", {"lookback": 60})
    assert [x["status"] for x in registry.history("momentum")] == ["running", "failed", "running"]
    with pytest.raises(ValueError, match="already closed"):
        registry.finish(failed, "completed", {})
    with pytest.raises(ValueError, match="not preregistered"):
        registry.start("momentum", {"lookback": 90})
    with pytest.raises(ValueError, match="changed"):
        registry.register("momentum", {**spec, "selection_rule": "best return"}, now=clock)
    with pytest.raises(ValueError, match="after study"):
        registry.register("leaked", {**spec, "holdout_start": "2025-01-01"}, now=clock)


def test_unknown_and_invalid_studies(tmp_path):
    registry = TrialRegistry(tmp_path / "db.sqlite")
    with pytest.raises(ValueError):
        registry.register("x", {})
    with pytest.raises(ValueError):
        registry.start("x", {})
    with pytest.raises(ValueError):
        registry.finish("absent", "completed", {})
    with pytest.raises(ValueError):
        registry.finish("absent", "success", {})


def test_holdout_is_prospective_frozen_and_evaluated_only_once(tmp_path):
    import sqlite3

    registry = TrialRegistry(tmp_path / "studies.db")
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    definition = {
        "hypothesis": "predeclared",
        "parameters": [{"window": 20}],
        "code_identity": "abc",
        "selection_rule": "fixed",
        "holdout_start": "2026-09-21",
        "holdout_end": "2026-12-31",
    }
    registry.register("holdout", definition, now=now)
    evidence = {
        "start": "2026-09-21",
        "end": "2026-12-31",
        "code_identity": "abc",
        "input_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="not ended"):
        registry.seal_holdout("holdout", evidence, now=now)
    done = datetime(2027, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="original code"):
        registry.seal_holdout("holdout", {**evidence, "code_identity": "new winner"}, now=done)
    registry.seal_holdout("holdout", evidence, now=done)
    with pytest.raises(sqlite3.IntegrityError):
        registry.seal_holdout("holdout", evidence, now=done)
    with registry.connect() as db, pytest.raises(sqlite3.IntegrityError, match="immutable"):
        db.execute("DELETE FROM holdout_evaluations")
