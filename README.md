# quant-lab

Cross-project experiment scanner and SQLite index for quant research outputs.

## Install

```bash
cd quant-lab
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start

```bash
quant-lab init
quant-lab scan --workspace configs/default.yaml
quant-lab list
quant-lab compare --project a-share-multifactor run_a run_b
```

## Workspace config

`configs/default.yaml` lists project output roots. Edit paths for your machine.

## Indexed run types

| Type | Markers |
|------|---------|
| `equity_backtest` | `capital_curves.csv` |
| `multifactor_compare` | `synthesis_comparison_summary.csv` |
| `sklearn_ml` | `feature_importance.csv` |
| `spread_backtest` | `performance/summary.csv` |

## Related

- [quant-research-notes](../quant-research-notes)
- [quant-report-hub](../quant-report-hub)
