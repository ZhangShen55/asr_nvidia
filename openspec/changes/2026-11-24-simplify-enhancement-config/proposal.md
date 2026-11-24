# Proposal

## Why

CAM++与emotion2vec现在同时由功能开关和设备数组表达启用状态，形成重复配置；当前continuous模板还展示两个仅legacy生效的组批参数与两个只能为1的增强batch伪配置。用户已确认清理6项，改用设备数组作为增强能力的唯一启用来源。

## What Changes

- **BREAKING** 删除`features.open_v2_role`和`features.open_v2_emotion`及对应配置读取，CAM++/emo分别由各自devices数组非空决定启用，空数组禁用。
- 启动预加载、路由准入前能力校验、直接增强入口统一读取数组派生状态；保留请求参数决定实际增强执行、已有4015/4016/4018和加载故障分类。
- 根及Docker的continuous模板删除`qwen_asr.batch_wait_timeout_ms`、`qwen_asr.batch_max_duration_ratio`；显式legacy参数与调度支持保留，不删除旧实验能力或重写历史报告。
- 从当前根/Docker模板删除固定为1的`speech_campplus_batch_size`、`emotion2vec_batch_size`显式配置；保留实际计算固定batch1与既有非1值拒绝约束，不扩大为模型批处理改造。迁移当前可执行样例及测试，历史归档原样保留。
- 残留旧增强开关在任何模型或worker启动前给出明确迁移错误；尤其不无声忽略旧false并开启非空数组模型。
- 保留两个prefetch_queue_depth、全部online_repetition_*/repetition_*、retry_count及队列/VAD参数。保留当前用户未提交的VAD阈值改动。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-model-runtime`：设备数组独立决定两增强能力启用、预加载、固定实例与运行入口防护。
- `qwen-asr-v2-adapter`：增强准入和并行执行改为数组驱动，保持错误优先级、段级降级与取消契约。
- `local-model-inference-backends`：后端不改变数组启用来源、预加载与独立容量。
- `service-operations`：精简模板，移除旧增强字段并提供迁移诊断，统一默认配置和文档。

## Impact

涉及core/config.py、core/models.py及能力校验调用方、根/Docker配置、当前可执行测试样例、CPU测试与部署文档。无依赖、权重、精度、实例数或v3字段/错误码数值变更。源码重启后生效；镜像/保护构建后续重新构建纳入，本轮不构建上线。当前阶段只生成规划，不修改运行代码或config.toml。
