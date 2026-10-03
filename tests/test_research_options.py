import pytest

from quant_lab.research_options import (
    validate_allocation,
    validate_execution,
    validate_options,
    validate_risk,
    validate_validation,
)


def test_training_direction_needs_mature_labels_inside_training_window():
    config = {
        "train_sessions": 6,
        "test_sessions": 10,
        "embargo_sessions": 5,
        "direction_policy": "train_ic",
        "direction_horizon": 5,
    }
    with pytest.raises(ValueError, match="mature direction labels"):
        validate_validation(config)
    assert validate_validation({**config, "train_sessions": 7})["train_sessions"] == 7


def test_allocation_and_history_contracts():
    assert validate_allocation({"mode": "inverse_vol"})["lookback"] == 20
    fields = {
        name: "state_" + name
        for name in ("listed", "delisted", "tradable", "limit_up", "limit_down")
    }
    history = {v: "status" for v in fields.values()}
    history["member"] = "universe"
    config = {"mode": "dynamic", "universe_field": "member", "status_fields": fields}
    assert validate_execution(config, history)["max_retry_sessions"] == 5
    with pytest.raises(ValueError, match="universe"):
        validate_execution(config, {})
    with pytest.raises(ValueError, match="status history"):
        validate_execution({"mode": "fixed", "status_fields": fields}, {})


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "any"},
        {"mode": "equal", "risk_aversion": 0},
        {"mode": "equal", "lookback": 2},
        {"mode": "equal", "min_observations": True},
        {"mode": "equal", "covariance_shrinkage": 1.1},
        {"mode": "equal", "turnover_penalty": float("nan")},
        {"mode": "equal", "max_turnover": 3},
        {"mode": "equal", "extra": 1},
        {"mode": "equal", "risk_aversion": -1},
    ],
)
def test_invalid_allocation(config):
    with pytest.raises(ValueError):
        validate_allocation(config)


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "dynamic"},
        {"mode": "unknown"},
        {"mode": "fixed", "universe_field": "member"},
        {"mode": "fixed", "max_retry_sessions": -1},
        {"mode": "fixed", "status_fields": {}},
    ],
)
def test_invalid_execution(config):
    with pytest.raises(ValueError):
        validate_execution(config, {})


@pytest.mark.parametrize(
    "recipe",
    [
        {"backend": "futures_fixture", "allocation": {}},
        {"backend": "equity", "factor_expressions": {"../bad": "close"}},
        {"backend": "equity", "factor_expressions": {"good": ""}},
        {"backend": "equity", "factor_expressions": []},
    ],
)
def test_invalid_extensions(recipe):
    with pytest.raises(ValueError):
        validate_options(recipe)


@pytest.mark.parametrize("mode", ["equal", "cost_aware"])
def test_factor_and_industry_risk_allow_only_joint_constraint_allocators(mode):
    recipe = {
        "backend": "equity",
        "allocation": {"mode": mode},
        "risk_model": {"model_kind": "statistical_proxy", "factor_bounds": {"market": [0, 0.9]}},
        "risk": {"max_industry_weight": 0.4, "industry_field": "industry"},
        "required_history": {"industry": "classification"},
    }
    validate_options(recipe)
    validate_risk(recipe)
    recipe["allocation"] = {"mode": "inverse_vol"}
    with pytest.raises(ValueError, match="risk_model requires equal or cost_aware"):
        validate_options(recipe)
    with pytest.raises(ValueError, match="Industry limits require equal or cost_aware"):
        validate_risk(recipe)


def test_equal_risk_allocation_still_requires_valid_model_and_pit_industry():
    recipe = {"backend": "equity", "allocation": {"mode": "equal"}, "risk_model": {}}
    with pytest.raises(ValueError, match="model_kind"):
        validate_options(recipe)
    recipe["risk"] = {"max_industry_weight": 0.4, "industry_field": "industry"}
    with pytest.raises(ValueError, match="PIT classification"):
        validate_risk(recipe)
