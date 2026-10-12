"""Reproducible independent financial benchmarks with explicit unavailable results."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from decimal import Decimal, localcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from quant_benchmarks.oracle import DERIVATIONS, expectations

REPOSITORIES = {
    "quant-execution": "quant_execution",
    "quant-data-kit": "quant_data_kit",
    "quant-fund": "quant_fund",
    "quant-report-hub": "quant_report_hub",
}
CASES = {
    "corporate_actions": ["quant-data-kit", "quant-execution"],
    "futures_roll": ["quant-data-kit", "quant-execution"],
    "fof_fees": ["quant-fund"],
    "fx_transfer": ["quant-report-hub"],
}


def _git(path, *args):
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def residuals(expected, actual, tolerance="0"):
    if set(expected) != set(actual):
        raise ValueError("Expected and observed fields differ")
    tolerance = Decimal(tolerance)
    if not tolerance.is_finite() or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative")
    result = {}
    for key in expected:
        left, right = Decimal(expected[key]), Decimal(str(actual[key]))
        if not left.is_finite() or not right.is_finite():
            raise ValueError("Nonfinite benchmark result")
        error = right - left
        result[key] = {
            "expected": str(left),
            "observed": str(right),
            "residual": str(error),
            "tolerance": str(tolerance),
            "pass": abs(error) <= tolerance,
        }
    return result


def run_benchmarks(sources, output, *, cases=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selected = list(CASES) if cases is None else list(cases)
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(CASES):
        raise ValueError("Select distinct supported cases")
    identities, errors = {}, {}
    # Never pip-install or fetch sources implicitly. Caller supplies pinned checkouts.
    for repo in sorted({r for name in selected for r in CASES[name]}):
        try:
            entry = sources[repo]
            path = Path(entry["path"]).resolve()
            head = _git(path, "rev-parse", "HEAD")
            if head != entry["commit"]:
                raise ValueError("checkout HEAD differs from requested commit")
            dirty = _git(path, "status", "--porcelain")
            if dirty:
                raise ValueError("adapter checkout has uncommitted changes")
            package_root = path / "src" if (path / "src").is_dir() else path
            sys.path.insert(0, str(package_root))
            module = importlib.import_module(REPOSITORIES[repo])
            if not Path(module.__file__).resolve().is_relative_to(path):
                raise ValueError("import resolved outside pinned checkout")
            identities[repo] = {
                "path": str(path),
                "commit": head,
                "module": module.__file__,
                "dirty": False,
            }
        except Exception as exc:  # noqa: BLE001 -- missing adapters remain explicitly unavailable
            errors[repo] = f"{type(exc).__name__}: {exc}"
    with localcontext() as ctx:
        ctx.prec = 50
        expected = expectations()
        # Store oracle BEFORE the adapter invocation; independently auditable.
        (output / "expected.json").write_text(
            json.dumps(expected, indent=2) + "\n", encoding="utf-8"
        )
        from quant_benchmarks import adapters

        rows = []
        for case in selected:
            target = output / case
            target.mkdir()
            row = {
                "case": case,
                "evidence_kind": "synthetic",
                "sample_kind": "deterministic_fixture",
                "derivation": DERIVATIONS[case],
                "status": "unavailable",
            }
            missing = {repo: errors[repo] for repo in CASES[case] if repo in errors}
            if missing:
                row["unavailable_reasons"] = missing
            else:
                try:
                    observed, controls = getattr(adapters, case)(target)
                    tolerance = "0.000000001" if case == "fof_fees" else "0"
                    checks = residuals(expected[case], observed, tolerance)
                    row.update(
                        checks=checks,
                        negative_controls=controls,
                        status="pass"
                        if all(x["pass"] for x in checks.values())
                        and all(v is True for v in controls.values())
                        else "fail",
                    )
                except Exception as exc:  # noqa: BLE001 -- preserve failed real adapter evidence
                    row.update(status="fail", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    # Recheck after every adapter has finished. Never attest the opening pin if
    # another process edited or advanced a checkout during execution.
    for repo, identity in identities.items():
        try:
            path = Path(identity["path"])
            if _git(path, "rev-parse", "HEAD") != identity["commit"] or _git(
                path, "status", "--porcelain"
            ):
                raise ValueError("pinned checkout changed during benchmark execution")
            identity["verified_after_execution"] = True
        except (ValueError, OSError, subprocess.CalledProcessError) as exc:
            errors[repo] = str(exc)
            identity["verified_after_execution"] = False
            for row in rows:
                if repo in CASES[row["case"]]:
                    row.update(status="fail", identity_error=str(exc))
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "quant-lab", "quant-data-kit", "quant-report-hub"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "source checkout without installed metadata"
    report = {
        "schema": "quant.independent-financial-benchmarks/v1",
        "sources": identities,
        "status": "pass" if all(r["status"] == "pass" for r in rows) else "unavailable_or_failed",
        "evidence_kind": "synthetic",
        "cases": rows,
        "source_errors": errors,
        "environment": {"python": sys.version, "executable": sys.executable, "packages": packages},
        "units": {"money": "major_currency", "returns": "decimal"},
        "oracle_sha256": hashlib.sha256((output / "expected.json").read_bytes()).hexdigest(),
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "unsupported_scopes": [
            "live orders",
            "real market validity",
            "unpriced delistings",
            "SPAN margin offsets",
            "FOF investor subscriptions/redemptions and tax",
            "causal attribution of reported FX components",
        ],
        "limitations": [
            "Real repository APIs execute deterministic synthetic fixtures; arithmetic agreement does not certify market effectiveness.",
            "FOF public API uses float; absolute tolerance is 1e-9 major currency units. QExec and FX require exact Decimal equality.",
            "Reported FX component adapter is an accounting bridge, not independent observed market evidence.",
        ],
    }
    report["artifacts"] = {
        str(p.relative_to(output)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*")
        if p.is_file()
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Independent financial benchmarks",
        "",
        "Evidence: synthetic deterministic fixtures; real repository APIs.",
        "",
        f"Status: {report['status']}",
        "",
    ]
    for row in rows:
        lines += [f"## {row['case']}: {row['status']}", "", row["derivation"], ""]
        if "checks" in row:
            lines += [
                "| Quantity | Expected | Observed | Residual | Pass |",
                "|---|---:|---:|---:|---|",
            ]
            lines += [
                f"| {k} | {v['expected']} | {v['observed']} | {v['residual']} | {v['pass']} |"
                for k, v in row["checks"].items()
            ]
            lines += ["", "Negative controls: " + json.dumps(row["negative_controls"]), ""]
        else:
            lines += [json.dumps(row.get("unavailable_reasons", row.get("error"))), ""]
    lines += [
        "## Limits",
        "",
        *["- " + v for v in report["limitations"]],
        "",
        "Unsupported scopes: " + ", ".join(report["unsupported_scopes"]),
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="JSON mapping repo -> {path, commit}; exact clean checkouts",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    args = parser.parse_args(argv)
    report = run_benchmarks(
        json.loads(args.sources.read_text(encoding="utf-8-sig")), args.out, cases=args.case
    )
    print(json.dumps({"status": report["status"], "report": str(args.out / "report.json")}))
    return 0 if report["status"] == "pass" else 2
