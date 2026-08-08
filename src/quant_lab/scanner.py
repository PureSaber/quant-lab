"""Scan research project output directories and extract run metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

KNOWN_MARKERS: dict[str, tuple[str, ...]] = {
    "a-share-multifactor": ("capital_curves.csv", "ic_summary.csv", "report.html"),
    "sklearn-stock-trend": ("feature_importance.csv", "report.html", "proba_signals.parquet"),
    "future_spread": ("performance/summary.csv", "daily/portfolio"),
    "quant-report-hub": ("01_nav_drawdown.png",),
    "quant-agent": ("review_manifest.json",),
}


@dataclass
class ScannedRun:
    project: str
    run_id: str
    run_path: str
    run_type: str
    metrics: dict
    config_path: str


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _metrics_from_capital_curves(path: Path) -> dict:
    df = pd.read_csv(path)
    if df.empty or "date" not in df.columns:
        return {}
    value_cols = [c for c in df.columns if c != "date"]
    if not value_cols:
        return {}
    col = value_cols[0]
    series = df[col].astype(float)
    start = float(series.iloc[0]) if len(series) else 0.0
    end = float(series.iloc[-1]) if len(series) else 0.0
    total_return = (end / start - 1.0) if start else 0.0
    peak = series.cummax()
    dd = (series - peak) / peak.replace(0, pd.NA)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    return {
        "strategy": col,
        "start_value": round(start, 4),
        "end_value": round(end, 4),
        "total_return": round(total_return, 6),
        "max_drawdown": round(max_dd, 6),
    }


def _metrics_from_summary(path: Path) -> dict:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in row.items()}


def detect_project(run_path: Path, project_hint: str = "") -> str:
    if project_hint:
        return project_hint
    parts = {p.name for p in run_path.parents}
    for name in KNOWN_MARKERS:
        if name in parts:
            return name
    if (run_path / "capital_curves.csv").is_file():
        return "a-share-multifactor"
    if (run_path / "feature_importance.csv").is_file():
        return "sklearn-stock-trend"
    if (run_path / "performance" / "summary.csv").is_file():
        return "future_spread"
    return run_path.parent.name or "unknown"


def scan_run(run_path: Path, *, project: str = "") -> ScannedRun | None:
    run_path = Path(run_path)
    if not run_path.is_dir():
        return None

    project_name = detect_project(run_path, project)
    metrics: dict = {"files": [p.name for p in run_path.iterdir() if p.is_file()][:20]}
    run_type = "unknown"

    cap = run_path / "capital_curves.csv"
    if cap.is_file():
        metrics.update(_metrics_from_capital_curves(cap))
        run_type = "equity_backtest"

    summary = run_path / "performance" / "summary.csv"
    if summary.is_file():
        metrics.update(_metrics_from_summary(summary))
        run_type = "spread_backtest"

    synth = run_path / "synthesis_comparison_summary.csv"
    if synth.is_file():
        df = pd.read_csv(synth)
        metrics["synthesis_rows"] = len(df)
        run_type = "multifactor_compare"

    fi = run_path / "feature_importance.csv"
    if fi.is_file():
        metrics["feature_importance_rows"] = len(pd.read_csv(fi))
        run_type = "sklearn_ml"

    review_manifest = run_path / "review_manifest.json"
    if review_manifest.is_file():
        metrics["review_manifest"] = True
        run_type = "agent_review"

    factor_manifest = run_path / "factor_manifest.json"
    if factor_manifest.is_file():
        try:
            manifest = json.loads(factor_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        metrics["factor_set_hash"] = manifest.get("factor_set_hash", "")
        metrics["git_sha"] = manifest.get("git_sha", "")
        metrics["factor_count"] = len(manifest.get("factors") or [])
        run_type = "factor_run"

    config_path = ""
    for candidate in (run_path / "config.yaml", run_path.parent / "configs" / "default.yaml"):
        if candidate.is_file():
            config_path = str(candidate)
            break

    return ScannedRun(
        project=project_name,
        run_id=run_path.name,
        run_path=str(run_path.resolve()),
        run_type=run_type,
        metrics=metrics,
        config_path=config_path,
    )


def scan_outputs_root(root: Path, *, project: str = "") -> list[ScannedRun]:
    root = Path(root)
    if not root.is_dir():
        return []

    runs: list[ScannedRun] = []
    if any((root / marker).exists() for marker in ("capital_curves.csv", "report.html", "performance")):
        item = scan_run(root, project=project)
        if item:
            runs.append(item)
        return runs

    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            item = scan_run(child, project=project)
            if item and item.run_type != "unknown":
                runs.append(item)
    return runs


def load_workspace_config(path: Path) -> dict:
    return _read_yaml(path)


def scan_workspace(config_path: Path) -> list[ScannedRun]:
    cfg = load_workspace_config(config_path)
    projects = cfg.get("projects", [])
    all_runs: list[ScannedRun] = []
    for entry in projects:
        root = Path(entry["outputs"])
        project = entry.get("name", "")
        all_runs.extend(scan_outputs_root(root, project=project))
    return all_runs


def metrics_to_json(metrics: dict) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2)
