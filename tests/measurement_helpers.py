import json
from copy import deepcopy
from pathlib import Path

import pandas as pd

from quant_lab.contracts_v2 import ARTIFACT_SCHEMAS_V2, write_standard_run_v2
from quant_measurement import content_id


def measurement():
    value = json.loads(
        (Path(__file__).parents[1] / "measurement-example.json").read_text(encoding="utf-8-sig")
    )
    value["basis"].update(
        period_start="2025-01-02T00:00:00Z",
        period_end="2025-01-06T00:00:00Z",
        initial_capital="100",
        cost_policy_id=content_id({"commission": "0"}),
        valuation_policy_id=content_id({"marks": "explicit_close"}),
    )
    value["basis"]["observation_schedule_id"] = content_id(
        ["2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z", "2025-01-06T00:00:00Z"]
    )
    return value


def write_measured(path, value=None, *, rates=(0.1, -0.1)):
    value = deepcopy(measurement() if value is None else value)
    frames = {
        name: pd.DataFrame(columns=ARTIFACT_SCHEMAS_V2[name])
        for name in ("returns", "positions", "portfolio_snapshots", "exposures")
    }
    frames["returns"] = pd.DataFrame(
        [
            {
                "event_time": pd.Timestamp(date),
                "strategy_id": "s",
                "gross_return": rate,
                "net_return": rate,
                "nav_units": nav,
                "nav_scale": 0,
                "base_currency": value["basis"]["currency"],
            }
            for date, rate, nav in zip(
                ["2025-01-03T00:00:00Z", "2025-01-06T00:00:00Z"], rates, [110, 99]
            )
        ]
    )
    return write_standard_run_v2(
        path,
        project="measured",
        run_id=path.name,
        strategy_ids=["s"],
        profile="research",
        frames=frames,
        metrics={"total_return": -0.01, "measurement": value},
        config={
            "measurement_policies": {
                "cost": {"commission": "0"},
                "valuation": {"marks": "explicit_close"},
            }
        },
        code_version="a" * 40,
        internal_dependencies={},
        random_seed=1,
        dataset_snapshots=value["provenance"]["input_identity"],
        instrument_master_version="fixture",
        execution_model_version="fixture",
        base_currency=value["basis"]["currency"],
        lineage={"config": [], **{name: ["dataset:fixture"] for name in ["metrics", *frames]}},
        capabilities=["synthetic"],
        tags={"research_only": "true"},
    )
