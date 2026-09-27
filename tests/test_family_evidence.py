import numpy as np
import pandas as pd
import pytest
from test_research_workbench import recipe as recipe_fixture

from quant_lab.family_evidence import collect_family, evaluate_family
from quant_lab.research import execute_study
from quant_lab.trials import TrialRegistry

recipe = recipe_fixture


def test_cross_study_collection_uses_verified_artifacts_and_all_attempts(recipe, tmp_path):
    registry = TrialRegistry(tmp_path / "shared.db")
    basis = {
        "currency": "USD",
        "benchmark_id": "cash",
        "cost_policy": "net-v1",
        "periods_per_year": 252,
        "sample_kind": "historical",
    }
    registry.register_family("f", {"study_ids": ["s1", "s2"], "hypothesis": "h", "basis": basis})
    data = np.random.default_rng(7).normal(0.001, 0.01, (64, 4))
    dates = pd.date_range("2024-01-01", periods=64)
    call = 0

    def executor(spec, candidate, out):
        nonlocal call
        pd.Series(data[:, call], index=dates).to_csv(out / "returns.csv", header=["net_return"])
        call += 1
        return {
            "comparison": {
                "start": str(dates[0].date()),
                "end": str(dates[-1].date()),
                "currency": "USD",
                "costs": "net-v1",
            }
        }

    for study in ("s1", "s2"):
        recipe.update(study_id=study, family_id="f", measurement_basis=basis, diagnostics={})
        result = execute_study(
            recipe,
            tmp_path / study,
            registry=registry,
            executor=executor,
            identity={"code": "frozen"},
            data_identity={},
        )
        assert not result["failed"]
    values, audit = collect_family(registry, "f")
    assert audit["available"] and values.shape == (64, 4)
    evidence = evaluate_family(
        values,
        audit,
        benchmark=pd.Series(0.0, index=dates),
        blocks=4,
        block_lengths=(3, 5),
        repetitions=100,
    )
    assert all(method["available"] for method in evidence["methods"].values())
    # A failed retry remains part of the search even though the original succeeded.
    parameters = registry.definition("s1")["definition"]["parameters"][0]
    attempt = registry.start("s1", parameters)
    registry.finish(attempt, "failed", {"reason": "a material rerun failed"})
    values, audit = collect_family(registry, "f")
    assert not audit["available"] and len(audit["planned"]) == 5
    assert evaluate_family(values, audit)["methods"] == {}
    next((tmp_path / "s2").glob("attempts/*/returns.csv")).write_text("changed")
    _, audit = collect_family(registry, "f")
    assert audit["collection_errors"]


def test_missing_study_and_attempt_are_not_silently_omitted(tmp_path):
    registry = TrialRegistry(tmp_path / "registry.db")
    basis = {
        "currency": "USD",
        "benchmark_id": "cash",
        "cost_policy": "net",
        "periods_per_year": 12,
        "sample_kind": "historical",
    }
    registry.register_family("f", {"study_ids": ["a", "b"], "hypothesis": "h", "basis": basis})
    registry.register(
        "a",
        {
            "hypothesis": "h",
            "parameters": [{"p": 1}],
            "code_identity": "x",
            "selection_rule": "all",
            "family_id": "f",
        },
    )
    _, audit = collect_family(registry, "f")
    assert not audit["available"] and len(audit["planned"]) == 2
    with pytest.raises(ValueError):
        registry.register(
            "outside",
            {
                "hypothesis": "h",
                "parameters": [{"p": 1}],
                "code_identity": "x",
                "selection_rule": "all",
                "family_id": "f",
            },
        )
