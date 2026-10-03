from copy import deepcopy

import pandas as pd
import pytest

from quant_lab.counterfactuals import attribution, intervention_plan, validate_plan
from quant_lab.research import digest


@pytest.fixture
def definitions():
    base = {
        "factors": {"momentum_20d": 1, "volatility_20d": -1},
        "strategy": {
            "family": "etf_trend",
            "frequency": "weekly",
            "top_n": 2,
            "max_weight": 0.4,
            "cash_buffer": 0.2,
            "trend_window": 60,
        },
        "risk": {"max_single_weight": 0.45, "max_gross_weight": 0.95},
        "risk_model": {"model_kind": "statistical_proxy"},
    }
    benchmarks = {
        "passive": {"mode": "buy_hold", "risk": {}},
        "same_risk_constrained": {"mode": "buy_hold", "risk": base["risk"]},
        "cash": {"mode": "cash", "daily_return": 0},
    }
    return base, benchmarks


@pytest.mark.parametrize("reserve", [0, 0.4])
def test_cash_and_trend_changes_preserve_other_constraints(definitions, reserve):
    base, benchmarks = definitions
    original = deepcopy(base)
    plan = intervention_plan(
        base, {"cash_buffer": reserve, "trend_filter": "rank"}, benchmarks=benchmarks
    )
    assert validate_plan(plan) == plan
    assert base == original
    cash = deepcopy(plan["variants"]["cash_buffer"])
    cash["strategy"]["cash_buffer"] = base["strategy"]["cash_buffer"]
    assert cash == base
    trend = deepcopy(plan["variants"]["trend_filter"])
    trend["strategy"]["family"] = base["strategy"]["family"]
    assert trend == base
    # A valid new hash cannot disguise a simultaneous concentration change.
    plan["variants"]["cash_buffer"]["strategy"]["max_weight"] = 0.5
    plan["sha256"] = digest({k: v for k, v in plan.items() if k != "sha256"})
    with pytest.raises(ValueError, match="only one declared dimension"):
        validate_plan(plan)


@pytest.mark.parametrize("reserve", [True, -0.1, 1, float("nan"), float("inf"), "0"])
def test_invalid_cash_fraction_is_rejected_before_registration(definitions, reserve):
    base, benchmarks = definitions
    with pytest.raises(ValueError, match="finite fractions"):
        intervention_plan(base, {"cash_buffer": reserve}, benchmarks=benchmarks)


def test_trend_intervention_cannot_change_rebalancing_to_buy_hold(definitions):
    base, benchmarks = definitions
    with pytest.raises(ValueError, match="only switch rank and etf_trend"):
        intervention_plan(base, {"trend_filter": "buy_hold"}, benchmarks=benchmarks)
    base["strategy"]["family"] = "rank"
    assert intervention_plan(base, {"trend_filter": "etf_trend"}, benchmarks=benchmarks)
    base["strategy"]["family"] = "buy_hold"
    with pytest.raises(ValueError, match="only switch rank and etf_trend"):
        intervention_plan(base, {"trend_filter": "rank"}, benchmarks=benchmarks)


def test_new_axes_use_same_dates_and_exact_terminal_gap_accounting(definitions):
    base, benchmarks = definitions
    plan = intervention_plan(
        base, {"cash_buffer": 0, "trend_filter": "rank"}, benchmarks=benchmarks
    )
    data = pd.DataFrame(
        {
            "base": [0.1, -0.1, 0, 0],
            "cash_buffer": [0.1, -0.1, 0, 0],
            "trend_filter": [0.05, 0.05, 0, 0],
            "passive": [0.1, 0.1, 0, 0],
            "same_risk_constrained": [0.04, 0.04, 0, 0],
            "cash": [0, 0, 0, 0],
        },
        index=pd.date_range("2026-01-01", periods=4),
    )
    result = attribution(plan, data)
    assert result["one_at_a_time_effects"]["cash_buffer"] == 0
    assert result["one_at_a_time_effects"]["trend_filter"] == pytest.approx(0.1125)
    assert result["return_gap"] == pytest.approx(0.22)
    assert result["unexplained_and_interaction_residual"] == pytest.approx(0.1075)
    data.loc[data.index[0], "cash_buffer"] = float("nan")
    with pytest.raises(ValueError):
        attribution(plan, data)
