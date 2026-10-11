"""Compare indexed experiment runs."""

from __future__ import annotations

import json
from pathlib import Path

from quant_lab.measurement import read_measured_run
from quant_lab.scanner import scan_run
from quant_lab.store import ExperimentRecord, ExperimentStore
from quant_measurement import compare_measurements


def _load_metrics(record: ExperimentRecord) -> dict:
    return json.loads(record.metrics_json)


def compare_runs(store: ExperimentStore, project: str, run_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for run_id in run_ids:
        rec = store.get(project, run_id)
        if rec is None:
            rows.append({"project": project, "run_id": run_id, "error": "not found"})
            continue
        row = {
            "project": rec.project,
            "run_id": rec.run_id,
            "run_type": rec.run_type,
            "run_path": rec.run_path,
            "metrics": {},
            "measurement": None,
            "status": "unavailable",
        }
        try:
            run = Path(rec.run_path)
            if (run / "standard").exists():
                try:
                    measured = read_measured_run(run)
                    manifest = measured["manifest"]
                    if manifest.project != project or manifest.run_id != run_id:
                        raise ValueError("index identity differs from source")
                    row["metrics"] = measured["metrics"]
                    row["measurement"] = measured["measurement"]
                    row["status"] = "completed"
                except (ValueError, TypeError, OSError) as exc:
                    row["measurement_error"] = str(exc)
                    # Read-only legacy view may remain available, but never
                    # supply a declaration from a different source snapshot.
                    fresh = scan_run(run)
                    if fresh.project != project or fresh.run_id != run_id:
                        raise ValueError("index identity differs from source")
                    row["metrics"] = fresh.metrics
            elif rec.run_type.startswith("standard") or not run.is_dir():
                raise ValueError("indexed source artifacts are missing")
            else:
                row["metrics"] = _load_metrics(rec)
                row["measurement_error"] = "legacy cache: read-only, not economically comparable"
        except (ValueError, TypeError, OSError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    comparison = compare_measurements(
        [row.get("measurement") for row in rows],
        statuses=[row.get("status", "unavailable") for row in rows],
    )
    for row in rows:
        row["comparison"] = comparison
    return rows


def format_comparison_table(rows: list[dict]) -> str:
    headers = ["project", "run_id", "run_type", "total_return", "max_drawdown", "sharpe"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            " | ".join(
                [
                    str(row.get("project", "")),
                    str(row.get("run_id", "")),
                    str(row.get("run_type", "")),
                    str(metrics.get("total_return", metrics.get("累计收益率", "-"))),
                    str(metrics.get("max_drawdown", metrics.get("最大回撤", "-"))),
                    str(metrics.get("sharpe", metrics.get("Sharpe", "-"))),
                ]
            )
        )
    if rows and "comparison" in rows[0]:
        comparison = rows[0]["comparison"]
        lines.append("\nEconomic compatibility: " + str(comparison["comparable"]))
        lines.extend(
            f"- run {r['index']} {r['field']}: {r['message']}" for r in comparison["reasons"]
        )
        lines.append("Read-only metrics are not certification of market validity or investability.")
    return "\n".join(lines)
