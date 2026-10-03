# quant-lab

Account and study viewers can open `TrialRegistry(path, read_only=True)` to read an
existing registration without creating directories, tables, or triggers. Missing
databases fail explicitly. Connections use SQLite `mode=ro` and `query_only`, and
registry mutation methods reject calls in this mode. This does not freeze a live
writer or turn an unverified database into authenticated evidence; consumers must
still verify definitions and referenced artifacts.
Registry connection contexts commit or roll back on exit and then close their
SQLite handle, including when the consumer raises. Direct connections remain
available but must be explicitly closed by their caller.

研究可信度升级：接口、使用示例、验收及限制见 [11–20 使用说明](docs/RESEARCH_INTEGRITY_11_20.md)。

Cross-project experiment scanner and SQLite index for quant research outputs.

`quant-lab`同时保留历史`standard/v1`读取能力，并提供严格、不可变的
`standard/v2`Parquet产物契约。统一reader仅在v2目录完全不存在时回退v1；
检测到损坏或不完整的v2会直接失败。

索引中的项目与运行ID来自已验证的标准清单，允许本地输出目录使用不同名称。
只有声明`quant.decision/v1`的`decision.json`才按模拟决策卡读取状态，
并核对运行身份；择时等应用的同名研究文件不会被误认成模拟调仓授权。
已有索引中按目录名保存的标准运行应重新扫描生成新索引；索引升级不改写原研究产物。

## Install

```bash
cd quant-lab
python -m venv .venv
.venv\Scripts\activate
python -m pip install --requirement requirements.lock
python -m pip install --no-deps --no-build-isolation --editable .
python -m pip check
```

## Quick start

```bash
quant-lab init
quant-lab scan --workspace configs/default.yaml
quant-lab validate --run-dir ../a-share-multifactor/outputs/run_001
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
| `standard_v2_research` | validated `standard/v2` research profile |
| `standard_v2_backtest-ledger` | validated `standard/v2` ledger profile |

## standard/v2 API

```python
from quant_lab import load_and_validate_standard_run, write_standard_run_v2

manifest = load_and_validate_standard_run("outputs/run_001")
```

`write_standard_run_v2`要求调用方显式提供代码版本、内部依赖版本、随机种子、
数据快照、标的主数据版本、执行模型版本、基础币种和完整血缘DAG。它先在临时
目录完成schema及hash校验，再原子发布到`standard/v2`，已有目录永不覆盖。

## 契约治理与依赖锁定

本仓库属于`contract`层，公开声明`standard/v2@2.0.0`和
`puresaber.run-manifest@2.0.0`，声明位于`pyproject.toml`的
`[tool.quant-workspace]`，由全栈清单校验。任何契约字段或schema变更必须先更新
黄金样例、兼容测试和迁移说明；`standard/v1`历史产物只读且不可改写。

`requirements.lock`同时覆盖运行时、开发和editable构建依赖，CI先按锁文件安装，再以
`--no-deps --no-build-isolation`安装本仓库。更新锁文件时应在干净分支重新解析并运行完整测试、覆盖率、
`pip check`和跨仓清单校验，锁文件与代码一并提交。若发布验证失败，回滚到上一个
默认分支提交及其锁文件；不得移动旧tag，也不得修改已生成的研究产物。

锁文件使用`pip-compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras`
生成，禁止在锁之外临时解析构建后端。

Equity research recipes may retain `risk_model` and PIT industry caps when
switching allocation between `cost_aware` and `equal`. The consuming allocator
must jointly enforce absolute factor, budget, position and turnover constraints;
this contract change does not waive the execution adapter's tracking-error gate.
Other allocation modes remain rejected for those risk extensions.

## Related

- [quant-research-notes](../quant-research-notes)
- [quant-report-hub](../quant-report-hub)
