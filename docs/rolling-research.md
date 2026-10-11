# 不可变快照上的滚动研究入口

`quant_lab.rolling_research`提供独立、显式启用的研究SDK。安装`.[rolling-research]`会安装提供共享快照与持久缓存的精确QDK/Factors提交；普通Lab功能不导入这两个可选组件。开发锁包含此extra，CI实际执行滚动研究测试。

```bash
python -m pip install '.[rolling-research]'
```

需要工作台和独立Worker的完整固定安装时，使用quant-workspace的`profiles/durable-research`配置；冻结Notebook及前向账户无需迁移。

## 可组合接口

1. `validate_payload(payload)`验证闭合JSON字段、既有因子契约、证券池和诊断参数。
2. `compute_research(plan, adapter=..., cache=..., run_id=...)`解析快照并调用持久因子缓存。
3. `summarize_factors(table, plan)`复用`quant_lab.selection.hac_mean`，计算各事件时刻等权横截面因子均值的HAC诊断。
4. `execute_research(payload, adapter=..., cache=..., run_id=..., registry=None, execution_context=None)`组合上述步骤，发布Arrow表与JSON报告，返回`{"result": {...}, "outputs": [ArtifactRef, ArtifactRef]}`。

诊断对象是因子值，不是策略收益；不做绩效排名或赢家选择。观测不足或lags不小于样本数时返回`insufficient_observations`和null诊断，不伪造统计结果。

`ResearchAdapter`可注入。`resolve(snapshot, columns=..., universe=...)`返回`ResolvedMarket(table,input_identities)`，须包含已验证的请求快照身份；`publish(bytes,media_type=...)`返回真实持久ArtifactRef。内置`QdkResearchAdapter(ArtifactStore(...),owner=...)`使用真实SnapshotStore及CAS完成验证、受控列/证券查询、引用保护和发布，不打开URI路径。首版显式物化完整选中表；不提前按可得性删除事件，以免改变滚动观测数。

## 独立Worker

可信管理员注册`rolling_research=quant_lab.rolling_research:worker_handler`。签名是`handler(request:dict,workspace:Path)`。`request`使用公共`quant.job-request/v1`，`request.inputs`必须包含`payload.snapshot`同一内容身份。

本地环境变量：

```powershell
$env:QUANT_RESEARCH_STORE_ROOT='H:/isolated-engineering/artifacts'
$env:QUANT_RESEARCH_CACHE_DB='H:/isolated-engineering/factors.sqlite3'
# 可选；仅在预登记完成后启用
$env:QUANT_RESEARCH_REGISTRY_DB='H:/isolated-engineering/trials.sqlite3'
```

payload不接受路径、模块、shell或任意SQL。Worker的独占attempt目录必须包含`context.json`，字段为`schema_version=quant.attempt-context/v1`、`job_id`、`attempt_id`、正整数`fencing_token`、`worker_id`、`project_id`、`run_id`。处理器核对project/run，产物发布时原子pin到`job/<job_id>/attempt/<attempt_id>/outputs`。研究不release任何引用；崩溃可能保留pin。运行线仍需执行自己的artifact_guard和fenced finish，任务库是状态权威。

## 可执行合成行情与三组参数

在新隔离环境安装上述extra后，以下命令在quant-lab仓执行：

```powershell
python examples/rolling_research_fixture.py --store-root H:/isolated-engineering/artifacts --output H:/isolated-engineering/fixture
python -m quant_lab.rolling_research --request H:/isolated-engineering/fixture/request-5.json --workspace H:/isolated-engineering/cli-attempt-1
python -m quant_lab.rolling_research --request H:/isolated-engineering/fixture/request-10.json --workspace H:/isolated-engineering/cli-attempt-2
python -m quant_lab.rolling_research --request H:/isolated-engineering/fixture/request-20.json --workspace H:/isolated-engineering/cli-attempt-3
```

生成器发布120天、2只合成证券的同一不可变快照，写出5/10/20窗口的完整JobRequest。CLI创建明确标为standalone-cli的context，不领取运行租约；每次使用新目录，已存在context拒绝覆盖。正式Worker传入自己的context即可。CLI的stdout只输出结果JSON，日志写stderr。

快照角色必须为`date_column=event_time`、`symbol_column=instrument_id`、`available_at_column=available_at`；列还包含`sequence`、`event_id`、`received_at`和因子依赖值。时间必须UTC，满足event_time≤received_at≤available_at。窗口按证券观测数计算，示例的日历及365期年化为显式合成口径。

## TrialRegistry关联与失败语义

调用`register_research(registry,study_id,hypothesis=...,payloads=[全部候选])`预先声明参数集合，然后payload添加study_id。配置registry却缺study_id，或有study_id却未配置registry，均拒绝。每次实际执行（包括缓存命中）使用既有`start/finish`登记新trial attempt，候选参数身份保持同一预登记定义；失败记录保留，重试不会增加新的候选。结果返回`trial_attempt_id`；running事件context包含运行job/attempt/token/worker/project/run。

研究的`completed`只说明研究函数已生成产物，不代表JobStore的`succeeded`；失效Worker可能留下不可变缓存、已pin产物或孤立trial。进程被强杀时registry可遗留running记录，由运行attempt状态解释；本版不自动把它改成成功，不重写TrialRegistry或历史数据库。将运行库状态对账为interrupted属于后续能力。

完整报告绑定因子计算身份、输入快照/成员、源码与运行库版本及诊断参数。相同研究的Arrow及报告字节可复用；run/attempt与执行模式不进入内容身份，保留在调用结果/审计中。缓存输入按内容去重，与参数数量无关；原始数据在QDK CAS共享。

## 验收与范围

`python -m pytest tests/test_rolling_research.py -q --no-cov`实际创建QDK快照、CAS、SQLite缓存和TrialRegistry，并启动命令行子进程。未安装配套可选包时该集成模块会明确skip；交付验收必须配置三仓并做到零skip。完整`python -m pytest -q`验证原研究、家族/holdout、PIT适配和统计。

本版范围不包含真实拟合、辅助源增量、市场多vintage选择、端到端流式、自动清理pin、任务调度器或实盘执行。市场修订及训练边界遵循quant-factors的`docs/persistent-cache.md`。服务机部署验收另行执行，不以源码测试替代。
