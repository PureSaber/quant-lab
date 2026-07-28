"""Compare indexed experiment runs."""

from __future__ import annotations

import json

from quant_lab.store import ExperimentRecord, ExperimentStore


def _load_metrics(record: ExperimentRecord) -> dict:
    return json.loads(record.metrics_json)


def compare_runs(store: ExperimentStore, project: str, run_ids: list[str]) -> list[dict]:
    rows: list[dict] = []
    for run_id in run_ids:
        rec = store.get(project, run_id)
        if rec is None:
            rows.append({"project": project, "run_id": run_id, "error": "not found"})
            continue
        metrics = _load_metrics(rec)
        rows.append(
            {
                "project": rec.project,
                "run_id": rec.run_id,
                "run_type": rec.run_type,
                "run_path": rec.run_path,
                "metrics": metrics,
            }
        )
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
    return "\n".join(lines)
