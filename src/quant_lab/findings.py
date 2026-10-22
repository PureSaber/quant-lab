"""Read-only research inventory; execution success is not a research verdict."""

from collections import Counter
from pathlib import Path

from quant_lab.research import verify_result
from quant_lab.trials import TrialRegistry, canonical


def research_findings(path: Path) -> dict:
    registry = TrialRegistry(path, read_only=True)
    studies = []
    for study_id in registry.study_ids():
        spec = registry.definition(study_id)
        definition = spec["definition"]
        events = registry.history(study_id)
        starts = {e["attempt_id"]: e for e in events if e["status"] == "running"}
        latest = {e["attempt_id"]: e for e in events}
        attempts = []
        for attempt_id, event in latest.items():
            row = {**event, "parameters": starts[attempt_id]["payload"]["parameters"]}
            payload = event["payload"]
            if event["status"] == "completed" and isinstance(payload.get("result"), str):
                try:
                    root = Path(starts[attempt_id]["payload"]["context"]["artifact_root"]).resolve()
                    target = (root / payload["result"]).resolve()
                    if root not in target.parents:
                        raise ValueError("result escapes its registered artifact root")
                    result = verify_result(target, payload["sha256"])
                    if result.get("study_id") != study_id or result.get("attempt_id") != attempt_id:
                        raise ValueError("result identity mismatch")
                    row["evidence"] = {
                        "verified": True,
                        "metrics": result.get("metrics"),
                        "comparison": result.get("comparison"),
                        "objective_evaluation": result.get("objective_evaluation"),
                    }
                except (KeyError, ValueError, OSError, TypeError) as exc:
                    row["evidence"] = {"verified": False, "reason": str(exc)}
            else:
                row["evidence"] = {"verified": False, "reason": "no file-backed result"}
            attempts.append(row)
        attempted = {canonical(s["payload"]["parameters"]) for s in starts.values()}
        recipe = definition.get("recipe", {})
        studies.append(
            {
                "study_id": study_id,
                "hypothesis": definition["hypothesis"],
                "registered_at": spec["registered_at"],
                "definition_sha256": spec["sha256"],
                "family_id": definition.get("family_id"),
                "interval": recipe.get("interval"),
                "inputs": recipe.get("inputs"),
                "source": recipe.get("source"),
                "selection_rule": definition["selection_rule"],
                "planned_candidates": len(definition["parameters"]),
                "unattempted_candidates": sum(
                    canonical(p) not in attempted for p in definition["parameters"]
                ),
                "attempt_counts": dict(Counter(e["status"] for e in latest.values())),
                "attempts": attempts,
            }
        )
    return {
        "schema": "quant.research-findings/v1",
        "registry": str(Path(path).resolve()),
        "policy": "All attempts retained. Execution status does not imply hypothesis support or promotion.",
        "studies": studies,
    }


def findings_markdown(report: dict) -> str:
    def cell(value):
        return (
            str(value if value is not None else "not recorded")
            .replace("|", "\\|")
            .replace("\n", " ")
        )

    lines = [
        "# Research findings",
        "",
        report["policy"],
        "",
        "| Study | Hypothesis | Family | Planned / unattempted | Attempt outcomes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["studies"]:
        counts = ", ".join(
            f"{key}: {value}" for key, value in sorted(row["attempt_counts"].items())
        )
        lines.append(
            "| "
            + " | ".join(
                map(
                    cell,
                    [
                        row["study_id"],
                        row["hypothesis"],
                        row["family_id"],
                        f"{row['planned_candidates']} / {row['unattempted_candidates']}",
                        counts or "not started",
                    ],
                )
            )
            + " |"
        )
    lines += [
        "",
        (
            "DSR/PBO and human decisions are not inferred from run status. "
            "Use verified family evidence for selection statistics and Studio Research findings for human decisions."
        ),
    ]
    return "\n".join(lines) + "\n"
