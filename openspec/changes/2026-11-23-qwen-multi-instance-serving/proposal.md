# Proposal

## Why

正式分支的continuous仍只支持一个Qwen进程，多实例目前仅存在隔离实验；同六完整音频真实v3实验已取得6/6返回、P50/P95169.55/193.34秒、墙钟194.81秒，但有16段警告，不能代替正式能力或B类验收。用户现明确要求将可配置多实例落入正式代码，再验证正式A6及双开增强的B4。

## What Changes

- 正式continuous/process支持配置N个Qwen owned进程，默认local_instance_count=6、local_device="npu:7"、segment_concurrency=S=16；单HTTP服务instance_count仍为1。
- **BREAKING**：continuous的总音频容量由N派生，全局段窗口由N×S派生；旧audio_concurrency/request_concurrency若与派生值冲突须明确报迁移错误，根与Docker配置不重复维护这些派生值。B子上限默认4，有效值不超过总健康容量；N=6时总6/B4。
- 正常请求动态租用一个健康Qwen owner，整个音频生命周期绑定，逐段独立结束和补位、安全清理后归还供下一请求使用；禁止测试SHA白名单、文件名绑定、实验环境开关或固定六/八档位。
- 正式多实例的预加载、状态、取消、故障隔离、段级恢复/警告、排队和正常关闭纳入契约；帧/token/物理步预算明确为每owner，不把全局预算复制N份。
- 保持B类VAD/有效段裁剪后三路异步分发，Qwen credit先归还、PCM等三分支收尾；CAM++/emo为各自独立共享池，容量独立配置和标定，不随Qwen数复制。B4真实共驻六份Qwen，emo短段skip及声纹聚类规则保留。
- 用改造后的正式源码真实启动服务，执行原a1～a6 A6、固定同源四条A4/B4及增强共驻A6，记录完整/删失P50/P95、墙钟、警告/字段覆盖、各池等待和HBM。沿用用户取消8GiB提前停止的决定，保留336秒、OOM/故障及有界收尾；测试后释放所有owned资源。
- 本次只正式化已验证架构，不混入ITN执行器、供给隔离、prefill调度、KV布局、阈值或引擎重构；旧change任务和历史成绩不改写。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-qwen-asr-inference`：正式多owner、请求租约、每实例段窗口、预加载及状态和故障生命周期。
- `qwen-asr-v2-adapter`：派生总容量、B4分类准入及三分支的独立归还/汇合语义。
- `local-model-runtime`：主协调进程/owned Qwen进程池、独立增强池和健康容量。
- `service-operations`：配置缺省/迁移、实例身份和可观测加载与收尾。
- `offline-asr-service`：continuous默认下的段警告与既有字段/计时契约。
- `resource-pressure-diagnostics`：正式A6/A4/B4及共驻验证、统计与回收证据。

## Impact

预计涉及core/config、qwen client/queue/continuous process及lifespan/status，必要的API请求租约接线、增强池参数配置、根和Docker配置、README/部署说明及CPU/保护/NPU验证工具；保持原HF/BF16、conda asr、NPU7、既有v3接口和错误码，不更新权重或依赖、不实施Docker构建上线。增强池正式默认由B4标定后确定，不能把当前各8或B上限4当成已验证容量。

三个旧change保持open：throughput44/60、continuous20/25、feasibility26/32。新策略明确替代旧throughput按段跨副本均衡与旧continuous单实例的正式continuous范围；未验收旧差量不整体提升为主规格。同步前做独立合并预览和覆盖表，旧结果/任务不被A6/B4代替。当前仅创建规划，不改业务、不启动模型或服务。
