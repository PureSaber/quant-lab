# 研究配方与试验登记

`python -m quant_lab.research init --template template.yaml --output my-study.yaml --study-id my-study`

`python -m quant_lab.research plan my-study.yaml`

配方显式指定输入、区间、因子方向、持仓、调仓频率、费用和有限候选。所有候选在计算前统一登记到SQLite，包括单因子、消融、买入持有、成本倍数、延迟、频率和手工参数邻域。最多64个候选，不从结果自动挑选赢家。

`execute_study`接收策略executor；保存每次running/completed/failed/interrupted事件和每个候选的输入身份、代码提交、结果及文件哈希。独占进程锁防止重复写入；重启复用已验证的完成结果，重试失败结果时保留旧尝试。同一study_id不接受修改后的定义。输出目录应位于源代码仓库之外。

默认exploratory。初始化prospective必须提供未来holdout起止日；运行时TrialRegistry再次核验。登记未来区间不等于已经取得留出成绩，留出评价沿用独立seal_holdout接口。历史分段、ICIR和参数扰动不是未触碰的样本外证据。

完整模板、固定依赖环境和运行入口位于quant-workspace的`profiles/research-workbench`。期货/加密模板只接受各自明确的backend_parameters；其费用和会计规则来自隔离的冻结fixture运行环境。
