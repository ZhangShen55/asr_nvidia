# Proposal

## Why

当前 v2 只有一个 `audio_concurrency` 音频准入上限，普通转写和同时启用说话人/情绪分析的请求共享同一容量，无法单独限制本地增强模型带来的资源压力。需要把增强请求作为 B 类处理，在总音频容量内增加独立的 spk_emo 容量，同时让服务能力开关与请求参数严格一致，避免请求绕过 `open_v2_role` / `open_v2_emotion` 继续调用已关闭的模型。

## What Changes

- 在 `[qwen_asr]` 新增 `audio_spk_emo_concurrency`，默认值为 `4`；`audio_concurrency` 继续表示 A+B 的总音频并发上限。
- 将 `showSpk`、`showEmotion`、`showRoleIdentify` 全部为 `false` 的请求归为 A；任一参数为 `true` 的请求归为 B，并同时受总容量和 spk_emo 容量限制。`showRoleIdentify` 视为请求说话人能力。
- 使用原子双计数准入：总容量优先判断；总容量有余量但 B 容量已满时返回独立的 spk_emo 队列满错误；B 不占用部分名额后再等待。
- 保持三个增强参数默认值为 `false`。**BREAKING**：请求要求的增强能力若对应 `open_v2_role` 或 `open_v2_emotion` 为 `false`，请求在准入前失败，不再以 `null` 字段正常转写；该请求不进入后续音频处理，也不加载、不调用本地模型。
- 请求的说话人能力关闭时返回 `4015`，请求的情绪能力关闭时返回 `4016`；同时请求两种能力且两种能力都关闭时返回新增组合错误码 `4018`。新增 `4017` 表示 spk_emo 容量已满，`4013` 继续表示总音频容量或共享等待队列已满；即使两类活动容量都满，也优先返回总容量错误 `4013`。
- 在 `queue_wait_timeout_s > 0` 时保留 A+B 共享的等待容量，同类按到达顺序处理；允许 A 绕过因 B 容量已满而等待的 B，有资格的较早请求优先。准入租约从开始处理持有至处理及临时文件清理完成，成功、失败、超时和取消均准确回收等待名额或处理槽位。
- 在状态接口、阶段遥测和日志中暴露 A/B 分类、总音频容量、spk_emo 容量、等待数量和拒绝原因。
- 补充配置契约、路由、准入管理、模型调用屏蔽、槽位生命周期、状态字段和错误码测试，并更新部署文档与 OpenSpec 规格。

## Capabilities

### New Capabilities

无。该变更扩展现有 v2 音频准入能力，不引入独立的长期能力边界。

### Modified Capabilities

- `qwen-asr-v2-adapter`: 修改 v2 请求分类、总音频与 spk_emo 双重并发、等待调度、能力开关校验和错误码契约。
- `local-model-runtime`: 明确关闭 `open_v2_role` / `open_v2_emotion` 后对应模型池不得加载或推理；请求显式要求关闭能力时返回错误。
- `service-operations`: 增加配置项、状态接口字段、A/B 队列可观测性和错误码说明。

## Impact

- 主要涉及 `core/config.py`、`core/qwen_asr_queue.py`、`api/routes/asr_v2.py`、`api/routes/asr_qwen_common.py`、`core/models.py`、`core/status.py`、`core/request_telemetry.py`、配置模板、README 和测试。
- 不改变 `/v2/seacraft_asr` 成功响应结构，不改变 Qwen 段级并发、VAD 分段或本地模型池的设备数组语义。
- **BREAKING**：缺少 `audio_spk_emo_concurrency` 时同样使用默认 `4`，现有配置升级后 B 类音频并发会受此新上限限制；配置修改需重启生效。限制按服务实例进程独立生效。
