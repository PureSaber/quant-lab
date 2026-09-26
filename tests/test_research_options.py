import pytest

from quant_lab.research_options import (
    validate_allocation,
    validate_execution,
    validate_options,
)


def test_allocation_and_history_contracts():
    assert validate_allocation({"mode": "inverse_vol"})["lookback"] == 20
    fields = {name: "state_" + name for name in ("listed", "delisted", "tradable", "limit_up", "limit_down")}
    history = {v: "status" for v in fields.values()}
    history["member"] = "universe"
    config = {"mode": "dynamic", "universe_field": "member", "status_fields": fields}
    assert validate_execution(config, history)["max_retry_sessions"] == 5
    with pytest.raises(ValueError, match="universe"):
        validate_execution(config, {})
    with pytest.raises(ValueError, match="status history"):
        validate_execution({"mode": "fixed", "status_fields": fields}, {})


@pytest.mark.parametrize("config", [
    {"mode": "any"}, {"mode": "equal", "risk_aversion": 0},
    {"mode": "equal", "lookback": 2}, {"mode": "equal", "min_observations": True},
    {"mode": "equal", "covariance_shrinkage": 1.1},
    {"mode": "equal", "turnover_penalty": float("nan")},
    {"mode": "equal", "max_turnover": 3}, {"mode": "equal", "extra": 1},
    {"mode": "equal", "risk_aversion": -1},
])
def test_invalid_allocation(config):
    with pytest.raises(ValueError):
        validate_allocation(config)


@pytest.mark.parametrize("config", [
    {"mode": "dynamic"}, {"mode": "unknown"},
    {"mode": "fixed", "universe_field": "member"},
    {"mode": "fixed", "max_retry_sessions": -1},
    {"mode": "fixed", "status_fields": {}},
])
def test_invalid_execution(config):
    with pytest.raises(ValueError):
        validate_execution(config, {})


@pytest.mark.parametrize("recipe", [
    {"backend": "futures_fixture", "allocation": {}},
    {"backend": "equity", "factor_expressions": {"../bad": "close"}},
    {"backend": "equity", "factor_expressions": {"good": ""}},
    {"backend": "equity", "factor_expressions": []},
])
def test_invalid_extensions(recipe):
    with pytest.raises(ValueError):
        validate_options(recipe)
