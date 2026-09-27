"""Append-only study definitions and attempt history, including failed work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def canonical(value: dict) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class TrialRegistry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS studies (
                    study_id TEXT PRIMARY KEY, definition TEXT NOT NULL,
                    sha256 TEXT NOT NULL, registered_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_families (
                    family_id TEXT PRIMARY KEY, definition TEXT NOT NULL,
                    sha256 TEXT NOT NULL, registered_at TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS families_no_update BEFORE UPDATE ON research_families
                    BEGIN SELECT RAISE(ABORT, 'research families are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS families_no_delete BEFORE DELETE ON research_families
                    BEGIN SELECT RAISE(ABORT, 'research families are immutable'); END;
                CREATE TABLE IF NOT EXISTS trial_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    study_id TEXT NOT NULL REFERENCES studies(study_id),
                    attempt_id TEXT NOT NULL, status TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS holdout_evaluations (
                    study_id TEXT PRIMARY KEY REFERENCES studies(study_id),
                    recorded_at TEXT NOT NULL, evidence TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS studies_no_update BEFORE UPDATE ON studies
                    BEGIN SELECT RAISE(ABORT, 'study definitions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studies_no_delete BEFORE DELETE ON studies
                    BEGIN SELECT RAISE(ABORT, 'study definitions are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS trials_no_update BEFORE UPDATE ON trial_events
                    BEGIN SELECT RAISE(ABORT, 'trial history is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS trials_no_delete BEFORE DELETE ON trial_events
                    BEGIN SELECT RAISE(ABORT, 'trial history is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS holdouts_no_update BEFORE UPDATE ON holdout_evaluations
                    BEGIN SELECT RAISE(ABORT, 'holdout evaluation is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS holdouts_no_delete BEFORE DELETE ON holdout_evaluations
                    BEGIN SELECT RAISE(ABORT, 'holdout evaluation is immutable'); END;
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def register(self, study_id: str, definition: dict, *, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None or not study_id.strip():
            raise ValueError("A study ID and timezone-aware registration time are required")
        for field in ("hypothesis", "parameters", "code_identity", "selection_rule"):
            if not definition.get(field):
                raise ValueError(f"Missing study field: {field}")
        if "objective" in definition:
            from quant_lab.objectives import validate_objective

            validate_objective(definition["objective"])
        if definition.get("family_id"):
            family = self.family(definition["family_id"])
            if study_id not in family["definition"]["study_ids"]:
                raise ValueError("Study was not preregistered in this research family")
        payload = canonical(definition)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            existing = db.execute(
                "SELECT sha256 FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("Study definition changed; register a new study ID")
                return digest
            holdout = definition.get("holdout_start")
            if holdout and datetime.fromisoformat(holdout).date() <= now.date():
                raise ValueError("Prospective holdout must begin after study registration")
            if definition.get("holdout_end") and (
                not holdout or definition["holdout_end"] <= holdout
            ):
                raise ValueError("Holdout end must follow holdout start")
            db.execute(
                "INSERT INTO studies VALUES (?,?,?,?)", (study_id, payload, digest, now.isoformat())
            )
        return digest

    def register_family(self, family_id: str, definition: dict, *, now=None) -> str:
        """Declare all member studies before any of them starts; no post-hoc family edits."""
        now = now or datetime.now(timezone.utc)
        studies = definition.get("study_ids", [])
        if (
            not family_id.strip()
            or now.tzinfo is None
            or not studies
            or len(studies) != len(set(studies))
            or any(not isinstance(s, str) or not s.strip() for s in studies)
            or not definition.get("hypothesis")
            or not definition.get("basis")
        ):
            raise ValueError("Family requires hypothesis, common basis and distinct study IDs")
        import pandas as pd

        from quant_lab.selection import audit_family

        audit_family(
            pd.DataFrame(),
            planned=studies,
            statuses={s: "running" for s in studies},
            family_id=family_id,
            basis=definition["basis"],
        )
        payload = canonical(definition)
        digest = hashlib.sha256(payload.encode()).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT sha256 FROM research_families WHERE family_id=?", (family_id,)
            ).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError(
                        "Family changed; use a new ID and disclose the expanded search"
                    )
                return digest
            present = {r[0] for r in db.execute("SELECT study_id FROM studies")}
            if present & set(studies):
                raise ValueError("Register the family before its member studies")
            db.execute(
                "INSERT INTO research_families VALUES (?,?,?,?)",
                (family_id, payload, digest, now.isoformat()),
            )
        return digest

    def family(self, family_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT definition,registered_at,sha256 FROM research_families WHERE family_id=?",
                (family_id,),
            ).fetchone()
        if row is None:
            raise ValueError("Unknown preregistered research family")
        return {"definition": json.loads(row[0]), "registered_at": row[1], "sha256": row[2]}

    def family_inventory(self, family_id: str) -> dict:
        family = self.family(family_id)
        studies = []
        for study_id in family["definition"]["study_ids"]:
            try:
                spec = self.definition(study_id)
            except ValueError:
                studies.append({"study_id": study_id, "status": "not_registered", "attempts": []})
                continue
            if spec["definition"].get("family_id") != family_id:
                raise ValueError("Member study did not declare its family identity")
            events = self.history(study_id)
            latest = {event["attempt_id"]: event for event in events}
            studies.append(
                {
                    "study_id": study_id,
                    "definition": spec,
                    "planned": spec["definition"]["parameters"],
                    "attempts": list(latest.values()),
                    "events": events,
                }
            )
        return {
            "family_id": family_id,
            **family,
            "studies": studies,
            "attempt_count": sum(len(s["attempts"]) for s in studies),
            "policy": "all attempts retained, including failed/interrupted/retried trials",
        }

    def start(self, study_id: str, parameters: dict, *, context: dict | None = None) -> str:
        attempt = uuid.uuid4().hex
        with self.connect() as db:
            row = db.execute(
                "SELECT definition FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
            if row is None or parameters not in json.loads(row[0])["parameters"]:
                raise ValueError("Parameters were not preregistered")
            db.execute(
                "INSERT INTO trial_events(study_id,attempt_id,status,recorded_at,payload) VALUES (?,?,?,?,?)",
                (
                    study_id,
                    attempt,
                    "running",
                    datetime.now(timezone.utc).isoformat(),
                    canonical({"parameters": parameters, "context": context or {}}),
                ),
            )
        return attempt

    def finish(self, attempt_id: str, status: str, payload: dict) -> None:
        if status not in {"completed", "failed", "interrupted", "skipped"}:
            raise ValueError("Invalid terminal trial status")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT study_id,status FROM trial_events WHERE attempt_id=? ORDER BY sequence",
                (attempt_id,),
            ).fetchall()
            if len(rows) != 1 or rows[0][1] != "running":
                raise ValueError("Attempt is absent or already closed")
            db.execute(
                "INSERT INTO trial_events(study_id,attempt_id,status,recorded_at,payload) VALUES (?,?,?,?,?)",
                (
                    rows[0][0],
                    attempt_id,
                    status,
                    datetime.now(timezone.utc).isoformat(),
                    canonical(payload),
                ),
            )

    def history(self, study_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT attempt_id,status,recorded_at,payload FROM trial_events WHERE study_id=? ORDER BY sequence",
                (study_id,),
            ).fetchall()
        return [
            {"attempt_id": a, "status": s, "recorded_at": t, "payload": json.loads(p)}
            for a, s, t, p in rows
        ]

    def definition(self, study_id: str) -> dict:
        with self.connect() as db:
            row = db.execute(
                "SELECT definition,registered_at,sha256 FROM studies WHERE study_id=?", (study_id,)
            ).fetchone()
        if row is None:
            raise ValueError("Unknown study")
        return {"definition": json.loads(row[0]), "registered_at": row[1], "sha256": row[2]}

    def seal_holdout(self, study_id: str, evidence: dict, *, now: datetime | None = None):
        """One terminal evaluation per registered study; no repeated winner selection."""
        now = now or datetime.now(timezone.utc)
        spec = self.definition(study_id)["definition"]
        if not spec.get("holdout_start") or not spec.get("holdout_end"):
            raise ValueError("Study has no complete prospective holdout interval")
        if now.date() <= datetime.fromisoformat(spec["holdout_end"]).date():
            raise ValueError("Holdout has not ended")
        if (
            evidence.get("start") != spec["holdout_start"]
            or evidence.get("end") != spec["holdout_end"]
        ):
            raise ValueError("Evaluation interval differs from preregistration")
        if evidence.get("code_identity") != spec["code_identity"] or not evidence.get(
            "input_sha256"
        ):
            raise ValueError("Holdout requires original code identity and immutable input evidence")
        with self.connect() as db:
            db.execute(
                "INSERT INTO holdout_evaluations VALUES (?,?,?)",
                (study_id, now.isoformat(), canonical(evidence)),
            )
