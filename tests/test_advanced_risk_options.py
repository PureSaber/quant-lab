from copy import deepcopy

import pytest

from quant_lab.research_options import validate_risk_model


def recipe():
    return {
        "model_kind": "fundamental_style",
        "lookback": 60,
        "exposure_pipeline": {
            "descriptor_fields": {"size": "log_cap", "value": "book_to_price"},
            "market_cap_field": "market_cap",
            "industry_field": "sector",
            "config": {
                "styles": {"size": {"size": 1}, "value": {"value": 1}},
                "industry_universe": ["a", "b"],
                "orthogonalize": {"value": ["size"]},
            },
        },
        "estimation": {
            "regression_weighting": "sqrt_market_cap",
            "industry_constraint": True,
            "factor_half_life": 30,
            "specific_half_life": 20,
            "risk_horizon": 5,
            "newey_west_lags": 4,
            "specific_newey_west_lags": 4,
            "regime_window": 10,
            "specific_target": "industry",
            "specific_shrinkage_method": "bayesian",
        },
        "factor_bounds": {"industry:a": [0, 0.6], "size": [-1, 1]},
    }


HISTORIES = {
    "log_cap": "fundamentals",
    "book_to_price": "fundamentals",
    "market_cap": "fundamentals",
    "sector": "classification",
}


def test_advanced_contract_and_legacy_defaults():
    value = recipe()
    before = deepcopy(value)
    result = validate_risk_model(value, HISTORIES)
    assert result["estimation"]["newey_west_lags"] == 4
    assert value == before
    legacy = validate_risk_model({"model_kind": "statistical_proxy"}, {})
    assert legacy["estimation"] == {}


@pytest.mark.parametrize(
    "change",
    [
        {"newey_west_lags": 2},
        {"regime_window": 50},
        {"factor_half_life": 0},
        {"specific_half_life": True},
        {"regression_weighting": "inverse_cap"},
        {"industry_constraint": 1},
        {"specific_target": "unknown"},
        {"specific_shrinkage_method": "unknown"},
        {"specific_prior_dof": -1},
        {"risk_horizon": 0},
        {"regime_window": 1},
        {"regime_scale_max": 0.5},
        {"regime_scale_min": 2},
        {"max_condition_number": 1},
        {"unknown": True},
    ],
)
def test_rejects_invalid_estimation_contract(change):
    value = recipe()
    value["estimation"].update(change)
    with pytest.raises(ValueError):
        validate_risk_model(value, HISTORIES)


@pytest.mark.parametrize(
    "change",
    [
        {"styles": {}},
        {"styles": {"market": {"size": 1}}},
        {"styles": {"size": {"size": 0}}},
        {"industry_universe": ["a", "a"]},
        {"market_factor": False},
        {"include_industries": False},
        {"winsor_method": "bad"},
        {"winsor_method": "quantile", "winsor_limit": 1},
        {"max_missing_fraction": 1},
        {"min_assets": True},
        {"max_descriptor_age_days": -1},
        {"orthogonalize": {"value": ["missing"]}},
        {"orthogonalize": {"size": ["value"], "value": ["size"]}},
        {"unknown": True},
    ],
)
def test_rejects_invalid_exposure_contract(change):
    value = recipe()
    value["exposure_pipeline"]["config"].update(change)
    with pytest.raises(ValueError):
        validate_risk_model(value, HISTORIES)


def test_requires_historical_numeric_fields_and_unambiguous_pipeline():
    value = recipe()
    with pytest.raises(ValueError, match="PIT numeric"):
        validate_risk_model(value, {**HISTORIES, "market_cap": "classification"})
    with pytest.raises(ValueError, match="classification history"):
        validate_risk_model(value, {**HISTORIES, "sector": "fundamentals"})
    value["exposure_fields"] = {"size": "log_cap"}
    with pytest.raises(ValueError, match="cannot be combined"):
        validate_risk_model(value, HISTORIES)
    value = {
        "model_kind": "statistical_proxy",
        "estimation": {"regression_weighting": "market_cap"},
    }
    with pytest.raises(ValueError, match="PIT exposure_pipeline"):
        validate_risk_model(value, {})
