import sqlite3
from datetime import datetime, timezone

import pytest

from quant_lab.trials import TrialRegistry


def test_read_only_registry_preserves_legacy_schema_and_file(tmp_path):
    path = tmp_path / "old # 注册.db"
    # A historical registry has only the schema needed by its original version.
    # A display must not silently add current tables or triggers to it.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE studies (study_id, definition, sha256, registered_at)")
        db.execute("INSERT INTO studies VALUES ('old', '{}', 'digest', '2026-01-01')")
    before = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    reader = TrialRegistry(path, read_only=True)
    assert reader.definition("old")["sha256"] == "digest"
    with reader.connect() as db:
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == [("studies",)]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("CREATE TABLE forbidden (id)")
    assert path.read_bytes() == before
    assert path.stat().st_mtime_ns == stamp
    assert list(tmp_path.iterdir()) == [path]


def test_read_only_missing_registry_does_not_create_directory(tmp_path):
    root = tmp_path / "absent"
    with pytest.raises(FileNotFoundError):
        TrialRegistry(root / "account.db", read_only=True)
    assert not root.exists()


@pytest.mark.parametrize("read_only", [False, True])
def test_connection_context_closes_handle_after_success_and_failure(tmp_path, read_only):
    path = tmp_path / "registry.db"
    TrialRegistry(path)
    registry = TrialRegistry(path, read_only=read_only)
    for fail in (False, True):
        try:
            with registry.connect() as db:
                db.execute("SELECT * FROM studies").fetchall()
                if fail:
                    raise ValueError("consumer failed")
        except ValueError:
            pass
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            db.execute("SELECT 1")
    # Windows refuses this operation while a SQLite handle remains open.
    path.unlink()
    assert not path.exists()


def test_connection_context_keeps_transaction_commit_and_rollback(tmp_path):
    registry = TrialRegistry(tmp_path / "registry.db")
    with registry.connect() as db:
        db.execute("CREATE TABLE consumer_test (value TEXT)")
        db.execute("INSERT INTO consumer_test VALUES ('committed')")
    with pytest.raises(ValueError), registry.connect() as db:
        db.execute("INSERT INTO consumer_test VALUES ('rolled back')")
        raise ValueError("rollback")
    with registry.connect() as db:
        assert db.execute("SELECT value FROM consumer_test").fetchall() == [("committed",)]


def test_read_only_history_and_all_mutators(tmp_path):
    path = tmp_path / "account.db"
    writer = TrialRegistry(path)
    definition = {
        "hypothesis": "fixed",
        "parameters": [{"window": 20}],
        "code_identity": "abc",
        "selection_rule": "fixed",
    }
    writer.register("study", definition)
    attempt = writer.start("study", {"window": 20})
    writer.finish(attempt, "failed", {"error": "missing prices"})
    before = path.read_bytes()
    reader = TrialRegistry(path, read_only=True)
    assert reader.history("study") == writer.history("study")
    calls = [
        lambda: reader.register("study", definition),
        lambda: reader.register_family("family", {}),
        lambda: reader.start("study", {"window": 20}),
        lambda: reader.finish(attempt, "completed", {}),
        lambda: reader.seal_holdout("study", {}),
    ]
    for call in calls:
        with pytest.raises(PermissionError, match="read-only"):
            call()
    assert path.read_bytes() == before


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
