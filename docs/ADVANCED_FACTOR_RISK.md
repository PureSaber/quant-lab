# 高级因子风险配方契约

`risk_model`新增两组可选字段：

- `exposure_pipeline`：完整的`config`、`descriptor_fields`、`market_cap_field`及
  `industry_field`。描述子与市值字段必须映射到`required_history.fundamentals`，
  行业字段必须映射到`classification`。与旧`exposure_fields`互斥。
- `estimation`：回归权重、行业约束、因子/特异风险半衰期、风险期限、Newey–West滞后、
  状态调整窗口/界限、分组特异风险、固定/贝叶斯收缩。旧配方默认空字典。

配方层保持无风险引擎依赖，在执行前验证闭合字段、类型、PIT映射、风格组合、行业全集、
正交化无环图及期限约束。非零HAC滞后必须等于H−1，回看长度必须覆盖滞后及完整状态
调整窗口。加权/行业估计须声明暴露流水线；行业约束须同时包含市场与行业因子。

风险限额可引用流水线中的风格、market及`industry:<name>`列。存在主动风险限额时
仍须提供权重总和为1的基准。`equal`和`cost_aware`配置继续受原有分配规则约束。
此契约不认证来源真实性、不自动校准参数、不改变风险限额。

完整可用配方及历史接入见
[A股研究接入](https://github.com/PureSaber/a-share-multifactor/blob/main/docs/ADVANCED_FACTOR_RISK.md)，
估计公式及独立校准见
[风险模型说明](https://github.com/PureSaber/quant-risk-monitor/blob/main/docs/ADVANCED_FACTOR_RISK.md)。
