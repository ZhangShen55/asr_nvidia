# Proposal

## Why

当前 CAM++ 与 emotion2vec 虽然已经由两个独立的 ModelPool 管理，但请求进入模型池前仍经过共享的 `local_gpu_concurrency` 闸门。该闸门把两个模型的容量绑定在一起，导致设备数组中配置的实例不能直接转化为各自的有效并发，8 路完整音频下本地模型任务在共享闸门前排队，成为端到端耗时的主要来源。

## What Changes

- 移除 CAM++ 与 emotion2vec 共用的模型级 GPU 并发闸门，改为两个模型池各自独立调度。
- 以 `speech_campplus_devices` 数组的有效实例数作为 CAM++ 并发槽位，以 `emotion2vec_devices` 数组的有效实例数作为 emotion2vec 并发槽位；不新增 `cam_slots`、`emo_slots` 或固定比例配置。
- 保持实例比例完全由配置数组决定，支持任意数量组合，例如 3:1、6:2 或其它组合，不在代码和配置默认值中写死比例。
- 保持不同模型池之间不借用实例；一个模型池空闲时不自动扩大另一个模型池的容量。
- 为两个模型池分别提供可观测的 busy、waiting、available、失败实例和等待耗时，能够区分模型池排队与 Qwen/VAD 排队。
- 为模型池等待增加公平调度，避免单个长音频的大量 VAD 段长期占满某一模型池。
- 保留 `segment_concurrency`、`request_concurrency`、音频并发和 VAD worker 池的独立语义；不把这些参数合并为模型槽位配置。
- 在 CAM++ 与 emotion2vec 均部署到 `cuda:0` 的条件下，使用设备数组配置 CAM++ 6 实例、emotion2vec 2 实例进行 8 路完整音频验证；测试完成后关闭 8084，保持 qwen-asr 8806 运行。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-model-runtime`: 模型池的有效并发容量、独立调度、公平等待和状态指标改为由各自设备数组决定。
- `qwen-asr-v2-adapter`: v2 分段的 CAM++/emotion2vec 调用不再受共享模型闸门限制，并增加模型池排队与端到端性能验证契约。

## Impact

- 影响 `api/routes/asr_qwen_common.py` 的本地模型调用闸门和耗时统计。
- 影响 `core/models.py` 的模型池租约、公平排队和状态刷新逻辑。
- 影响 `core/config.py` 对 `local_gpu_concurrency` 的读取与迁移处理；设备数组仍是唯一的模型实例数量来源，不增加新的槽位配置。
- 影响 `/get_status` 的 `localModels` 诊断字段，但不改变 `/v2/seacraft_asr` 成功响应字段。
- 需要在同一张 GPU 上验证不同模型实例组合的显存、GPU 争用、P50/P95、吞吐、错误率和结果一致性；实例数增加可能使单次推理变慢，因此以实测数据选择部署组合。
