"""Rank only certified v2 runs whose universe and accounting are comparable."""

from __future__ import annotations

from quant_lab.contracts_v2 import BACKTEST_LEDGER_PROFILE, RunManifestV2

_NOT_COMPARABLE = {
    "watchlist",
    "explicit_watchlist",
    "not_historical_universe",
    "current_watchlist_not_historical_universe",
    "fixed_watchlist_case",
}


def is_rankable(manifest) -> bool:
    """v1 ledgers and tagged watchlist or research-only runs stay out of rankings."""
    if not isinstance(manifest, RunManifestV2):
        return False
    if manifest.profile != BACKTEST_LEDGER_PROFILE:
        return False
    tags = manifest.tags or {}
    if tags.get("rankable") == "false" or tags.get("research_only") == "true":
        return False
    if tags.get("comparability") in _NOT_COMPARABLE:
        return False
    return tags.get("investable") != "false"
