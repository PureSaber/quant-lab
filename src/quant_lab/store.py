"""SQLite experiment index."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ExperimentRecord:
    id: int | None
    project: str
    run_id: str
    run_path: str
    run_type: str
    metrics_json: str
    config_path: str
    scanned_at: str


class ExperimentStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    run_path TEXT NOT NULL,
                    run_type TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    config_path TEXT NOT NULL,
                    scanned_at TEXT NOT NULL,
                    UNIQUE(project, run_id, run_path)
                )
                """
            )

    def upsert(
        self,
        *,
        project: str,
        run_id: str,
        run_path: str,
        run_type: str,
        metrics: dict,
        config_path: str = "",
    ) -> None:
        payload = json.dumps(metrics, ensure_ascii=False)
        scanned_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (project, run_id, run_path, run_type, metrics_json, config_path, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project, run_id, run_path) DO UPDATE SET
                    run_type=excluded.run_type,
                    metrics_json=excluded.metrics_json,
                    config_path=excluded.config_path,
                    scanned_at=excluded.scanned_at
                """,
                (project, run_id, run_path, run_type, payload, config_path, scanned_at),
            )

    def list_runs(self, project: str | None = None) -> list[ExperimentRecord]:
        query = "SELECT * FROM experiments"
        params: tuple = ()
        if project:
            query += " WHERE project = ?"
            params = (project,)
        query += " ORDER BY scanned_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [ExperimentRecord(**dict(row)) for row in rows]

    def get(self, project: str, run_id: str) -> ExperimentRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE project = ? AND run_id = ? ORDER BY scanned_at DESC LIMIT 1",
                (project, run_id),
            ).fetchone()
        return ExperimentRecord(**dict(row)) if row else None
