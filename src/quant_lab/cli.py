"""quant-lab CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from quant_lab.compare import compare_runs, format_comparison_table
from quant_lab.contracts_v2 import load_and_validate_standard_run
from quant_lab.export_html import export_html
from quant_lab.scanner import scan_outputs_root, scan_workspace
from quant_lab.store import ExperimentStore


def _default_config() -> Path:
    return Path("configs/default.yaml")


def _default_db() -> Path:
    return Path("state/experiments.db")


def cmd_scan(args: argparse.Namespace) -> int:
    if args.workspace:
        runs = scan_workspace(Path(args.workspace))
    else:
        runs = scan_outputs_root(Path(args.root), project=args.project or "")

    store = ExperimentStore(Path(args.db))
    for run in runs:
        store.upsert(
            project=run.project,
            run_id=run.run_id,
            run_path=run.run_path,
            run_type=run.run_type,
            metrics=run.metrics,
            config_path=run.config_path,
        )
    print(f"Indexed {len(runs)} run(s) -> {args.db}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    store = ExperimentStore(Path(args.db))
    rows = store.list_runs(project=args.project or None)
    if args.json:
        payload = [row.__dict__ for row in rows]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    for row in rows:
        print(f"{row.project}\t{row.run_id}\t{row.run_type}\t{row.run_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    store = ExperimentStore(Path(args.db))
    rows = compare_runs(store, args.project, args.run_ids)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(format_comparison_table(rows))
    return 0


def cmd_export_html(args: argparse.Namespace) -> int:
    out = export_html(Path(args.db), Path(args.out))
    print(f"wrote {out}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_and_validate_standard_run(Path(args.run_dir))
    payload = {
        "valid": True,
        "schema_version": manifest.schema_version,
        "project": manifest.project,
        "run_id": manifest.run_id,
        "code_version": manifest.code_version,
        "dataset_snapshots": manifest.dataset_snapshots,
        "artifacts": {record.name: record.sha256 for record in manifest.artifacts},
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"valid standard run: {manifest.project}/{manifest.run_id} "
            f"schema={manifest.schema_version}"
        )
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg_path.is_file() and not args.force:
        print(f"Config already exists: {cfg_path}")
        return 0
    template = {
        "db_path": "state/experiments.db",
        "projects": [
            {"name": "a-share-multifactor", "outputs": "../a-share-multifactor/outputs"},
            {"name": "sklearn-stock-trend", "outputs": "../sklearn-stock-trend/outputs"},
            {"name": "future_spread", "outputs": "../future_spread_analysis-team-framework/output"},
        ],
    }
    cfg_path.write_text(yaml.safe_dump(template, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Wrote {cfg_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant-lab", description="Cross-project experiment indexer")
    p.add_argument("--db", default=str(_default_db()), help="SQLite index path")
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create default workspace config")
    init.add_argument("--config", default=str(_default_config()))
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    scan = sub.add_parser("scan", help="Scan outputs and index runs")
    scan.add_argument("--root", default="", help="Single outputs root directory")
    scan.add_argument("--workspace", default="", help="Workspace YAML with multiple projects")
    scan.add_argument("--project", default="", help="Project name hint")
    scan.set_defaults(func=cmd_scan)

    lst = sub.add_parser("list", help="List indexed runs")
    lst.add_argument("--project", default="")
    lst.add_argument("--json", action="store_true")
    lst.set_defaults(func=cmd_list)

    cmp = sub.add_parser("compare", help="Compare indexed runs")
    cmp.add_argument("--project", required=True)
    cmp.add_argument("run_ids", nargs="+")
    cmp.add_argument("--json", action="store_true")
    cmp.set_defaults(func=cmd_compare)

    export = sub.add_parser("export", help="Export dashboard artifacts")
    export_sub = export.add_subparsers(dest="export_cmd", required=True)
    html_cmd = export_sub.add_parser("html", help="Write static HTML dashboard")
    html_cmd.add_argument("--out", default="reports/dashboard.html")
    html_cmd.set_defaults(func=cmd_export_html)

    validate = sub.add_parser("validate", help="Validate an immutable standard run")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
