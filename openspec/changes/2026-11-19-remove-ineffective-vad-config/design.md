# Design
## Context
历史阈值评审和正式运行器表明nested vad_kwargs不会覆盖FSMN构造选项。现有CPU/NPU模型构造选项来自原始模型配置。服务段级SNR与独立段级能量阈值另有真实用途。
## Goals / Non-Goals
移除四个无效用户参数、Settings正名/别名及自动转发；保留模型有效选项和显式内部vad_kwargs测试钩子。不改变切分规则、过滤功能或资产。
## Decisions
池默认不再构造nested阈值，仅显式内部调用可传vad_kwargs。SNR只用于段级过滤，不再nested转发。复读字段移到实际读取表，数值与此前有效缺省相等。所有当前模板清理，不改历史报告与归档。
## Risks / Trade-offs
外部旧配置仍可能含旧键 → 不提供读回退；文档明确已删除。实际切分变化风险 → 以既有源代码构造审计和模型选项证据确认nested无覆盖；CPU测试验证不自动转发，不执行NPU模型加载。
