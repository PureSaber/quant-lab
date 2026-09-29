"""Publish an exploratory standard/v2 research profile from a daily NAV path.

The profile is a contract adapter. It does not upgrade a watchlist or a
synthetic study into an investable, rankable ledger.
"""

from __future__ import annotations

import subprocess
from decimal import ROUND_HALF_EVEN, Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from quant_lab.contracts_v2 import ARTIFACT_SCHEMAS_V2, RESEARCH_PROFILE, write_standard_run_v2

_SCALE = 4


def clean_git_commit(root: Path) -> str | None:
    root = Path(root)
    if not (root / ".git").exists():
        return None
    try:
        return _clean_git_commit(root)
    except OSError:
        return None


def _clean_git_commit(root: Path) -> str | None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    sha = commit.stdout.strip()
    if commit.returncode != 0 or dirty.returncode != 0 or dirty.stdout.strip() or len(sha) != 40:
        return None
    return sha


def _units(value, scale: int = _SCALE) -> int:
    quantum = Decimal(1).scaleb(-scale)
    amount = Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_EVEN)
    return int(amount.scaleb(scale))


def _stamp(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    if stamp == stamp.normalize():
        # Date-only summaries are available at the end of that UTC day, never
        # before a US close. Callers can supply an exact aware close timestamp.
        stamp = stamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return stamp


def write_exploratory_run_v2(
    run_dir: Path,
    *,
    project: str,
    run_id: str,
    strategy_id: str,
    currency: str,
    code_version: str,
    dataset_snapshots: dict[str, str],
    nav: pd.DataFrame,
    comparability: str,
    positions: pd.DataFrame | None = None,
    internal_dependencies: dict[str, str] | None = None,
    metrics: dict | None = None,
    config: dict | None = None,
) -> object:
    if "date" not in nav.columns or "nav" not in nav.columns:
        raise ValueError("exploratory NAV requires date and nav columns")
    if not dataset_snapshots:
        raise ValueError("exploratory NAV requires dataset snapshots")
    if nav.columns.has_duplicates:
        raise ValueError("exploratory NAV columns must be unique")
    frame = nav.copy()
    frame["event_time"] = frame["date"].map(_stamp)
    frame = frame.sort_values("event_time")
    if frame.empty:
        raise ValueError("exploratory NAV is empty")
    if "net_return" not in frame.columns:
        levels = frame["nav"].astype(float)
        frame["net_return"] = levels.pct_change().fillna(0.0)
    if "gross_return" not in frame.columns:
        frame["gross_return"] = frame["net_return"]
    returns = pd.DataFrame(
        {
            "event_time": frame["event_time"],
            "strategy_id": strategy_id,
            "gross_return": frame["gross_return"].astype(float),
            "net_return": frame["net_return"].astype(float),
            "nav_units": frame["nav"].map(_units),
            "nav_scale": _SCALE,
            "base_currency": currency,
        }
    )
    position_rows = []
    if positions is not None and not positions.empty:
        held = positions.copy()
        held["event_time"] = held["date"].map(_stamp)
        for row in held.itertuples(index=False):
            quantity = Decimal(str(row.quantity))
            if quantity < 0:
                raise ValueError("exploratory adapter requires long-only positions")
            if quantity == 0:
                continue
            mark = Decimal(str(row.mark_price))
            if mark <= 0:
                raise ValueError("position mark price must be positive")
            market = quantity * mark
            position_rows.append(
                {
                    "event_time": row.event_time,
                    "account_id": "research",
                    "strategy_id": strategy_id,
                    "instrument_id": str(row.instrument_id),
                    "quantity_units": _units(quantity),
                    "quantity_scale": _SCALE,
                    "mark_price_units": _units(mark),
                    "mark_price_scale": _SCALE,
                    "market_value_units": _units(market),
                    "market_value_scale": _SCALE,
                    "currency": currency,
                    "fx_rate_units": _units(1),
                    "fx_rate_scale": _SCALE,
                    "fx_snapshot_id": "identity",
                    "base_market_value_units": _units(market),
                    "base_market_value_scale": _SCALE,
                }
            )
    positions_frame = pd.DataFrame(position_rows, columns=list(ARTIFACT_SCHEMAS_V2["positions"]))
    if not positions_frame.empty:
        positions_frame = positions_frame.sort_values(["event_time", "instrument_id"]).reset_index(
            drop=True
        )
    market_by_time: dict = {}
    if not positions_frame.empty:
        for row in positions_frame.itertuples(index=False):
            market_by_time[row.event_time] = market_by_time.get(row.event_time, 0) + int(
                row.market_value_units
            )
    snapshots = []
    exposures = []
    for row in frame.itertuples(index=False):
        nav_units = _units(row.nav)
        market_units = int(market_by_time.get(row.event_time, 0))
        snapshots.append(
            {
                "event_time": row.event_time,
                "account_id": "research",
                "base_currency": currency,
                "nav_units": nav_units,
                "nav_scale": _SCALE,
                "cash_value_units": nav_units - market_units,
                "cash_value_scale": _SCALE,
                "market_value_units": market_units,
                "market_value_scale": _SCALE,
                "unrealized_pnl_units": 0,
                "unrealized_pnl_scale": _SCALE,
                "realized_pnl_units": 0,
                "realized_pnl_scale": _SCALE,
                "margin_used_units": 0,
                "margin_used_scale": _SCALE,
            }
        )
        exposures.append(
            {
                "event_time": row.event_time,
                "account_id": "research",
                "strategy_id": strategy_id,
                "exposure_type": "gross",
                "name": "nav",
                "value": 1.0,
                "unit": "weight",
            }
        )
    dataset_name = next(iter(dataset_snapshots))
    artifacts = {
        "returns": returns,
        "positions": positions_frame,
        "portfolio_snapshots": pd.DataFrame(snapshots),
        "exposures": pd.DataFrame(exposures),
    }
    lineage = {
        "config": [f"dataset:{dataset_name}"],
        "metrics": ["config"],
        "returns": ["config"],
        "positions": ["returns"],
        "portfolio_snapshots": ["returns"],
        "exposures": ["positions"],
    }
    return write_standard_run_v2(
        Path(run_dir),
        project=project,
        run_id=run_id,
        strategy_ids=[strategy_id],
        profile=RESEARCH_PROFILE,
        frames=artifacts,
        metrics=metrics or {"sessions": len(frame), "adapter": "exploratory-research-v2"},
        config=config or {"adapter": "exploratory-research-v2"},
        code_version=code_version,
        internal_dependencies=internal_dependencies or {"quant-lab": _package_version()},
        random_seed=0,
        dataset_snapshots=dataset_snapshots,
        instrument_master_version="exploratory-adapter",
        execution_model_version="exploratory-research-profile",
        base_currency=currency,
        lineage=lineage,
        capabilities=["research-adapter"],
        tags={
            "investable": "false",
            "rankable": "false",
            "comparability": comparability,
            "adapter": "exploratory-research-v2",
        },
    )


def _package_version() -> str:
    try:
        return version("quant-lab")
    except PackageNotFoundError:
        return "source-checkout"
