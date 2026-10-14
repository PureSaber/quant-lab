# 研究配方与试验登记

`python -m quant_lab.research init --template template.yaml --output my-study.yaml --study-id my-study`

`python -m quant_lab.research plan my-study.yaml`

配方显式指定输入、区间、因子方向、持仓、调仓频率、费用和有限候选。所有候选在计算前统一登记到SQLite，包括单因子、消融、买入持有、成本倍数、延迟、频率和手工参数邻域。最多64个候选，不从结果自动挑选赢家。

`execute_study`接收策略executor；保存每次running/completed/failed/interrupted事件和每个候选的输入身份、代码提交、结果及文件哈希。独占进程锁防止重复写入；重启复用已验证的完成结果，重试失败结果时保留旧尝试。同一study_id不接受修改后的定义。输出目录应位于源代码仓库之外。

默认exploratory。初始化prospective必须提供未来holdout起止日；运行时TrialRegistry再次核验。登记未来区间不等于已经取得留出成绩，留出评价沿用独立seal_holdout接口。历史分段、ICIR和参数扰动不是未触碰的样本外证据。

完整模板、固定依赖环境和运行入口位于quant-workspace的`profiles/research-workbench`。期货/加密模板只接受各自明确的backend_parameters；其费用和会计规则来自隔离的冻结fixture运行环境。

优化器新模板为 `templates/etf-optimizer-v2.yaml`，默认使用成熟、非重叠标签估计的滚动 `ic_vol`，明确预测交易日数。旧模板与冻结账户不会自动迁移。`risk_objective: active` 目前仅供 `cost_aware`，必须提供完整基准；部分现金不会被资金切片归一化隐藏。Barra 估计的 `risk_horizon` 应与预测期限一致。

显式预测使用 `expected_return_model: provided` 和 `inputs.return_forecasts` 指向冻结 CSV。列必须为 `symbol,forecast_at,available_at,horizon_sessions,expected_return,model_version`，时间带时区，收益为对应期限的小数收益。只使用调仓当天、当时已可用的最新修订；缺少候选资产或期限不符会停止候选。输入哈希参与研究身份，原始文件保存在结果中。原始分数只在未声明 v2 的兼容配置中保留。

结果同时记录连续权重、整手目标、实际持仓风险、约束残差、求解器及风险快照版本。无法找到通过账户和风险复核的整手方案时记录 `repair_failed`；求解失败不会自动改成等权。工程比较应固定输入和评价区间，保留所有失败候选，不按最终历史夏普回选参数。
