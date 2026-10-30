# Proposal

## Why

当前每个服务进程只缓存一份 CAM++ 和 emotion2vec 模型；`local_gpu_concurrency` 只能限制并发调用，不能增加模型副本，因此在 `showSpk=true`、`showEmotion=true` 的完整音频并发场景中，本地模型成为主要吞吐瓶颈。需要把模型实例从完整服务进程数量中解耦，并允许通过设备数组显式选择实例落卡，验证多实例是否能改善 2、4、8、16、32 路音频并发的端到端效率。

## What Changes

- 新增可配置的本地模型池能力，分别管理 CAM++ 与 emotion2vec 的实例数量；默认值保持单实例，兼容通常使用的 `instance_count=1`。
- 将模型池配置统一放入 `[model_device.local_model_pool]`，每个模型以设备数组声明实例；数组长度决定实例数，允许显式跨 GPU 配置，不做自动迁移。
- 让同一模型池内的多个实例参与并发推理，并保证请求结果仍按现有流程完成时间轴映射、角色聚类和情绪输出；不同 CAM++ 实例返回的 embedding 必须可在统一聚类阶段合并使用。
- 保留现有懒加载、失败降级、`showSpk`/`showEmotion` 请求开关和 `open_v2_role`/`open_v2_emotion` 配置开关语义；模型副本按实际启用的能力加载。
- 增加模型池运行状态与可观测字段，至少能区分配置副本数、已加载副本数、设备、忙碌/等待情况及推理失败。
- 增加完整音频并发基准：`showSpk=true`、`showEmotion=true`，并发档位为 2、4、8、16、32；比较单实例 baseline 与多实例方案，记录端到端时延、吞吐、GPU 利用率、显存、排队和错误情况，只有在数据证明改善时才接受推荐副本配置。
- 产出可归档的中文测试报告，说明各并发档位是否提升、同卡实例抢占造成的代价、显存边界和不满足提升目标时的结论。

## Capabilities

### New Capabilities

- `local-model-pool`: 管理 CAM++ 与 emotion2vec 的多设备实例生命周期、调度、状态和容量约束。

### Modified Capabilities

- `local-model-runtime`: 将单份本地模型缓存和实例显存关系扩展为可配置模型池，同时保留服务进程统一承载、懒加载和状态查询契约。
- `qwen-asr-v2-adapter`: 增加模型池配置、按设备数组绑定、并发推理结果一致性和 2/4/8/16/32 完整音频性能验证要求。

## Impact

- 影响配置解析与默认配置，特别是 `[model_device]` 下 CAM++/emotion2vec 的副本数量和设备字段。
- 影响 `core/models.py` 的模型缓存、推理调度和状态统计，以及 v2 role/emotion 调用路径；不改变对外响应字段和 `instance_count` 的完整服务进程语义。
- 影响 `/get_status` 中本地模型状态的表达，需兼容未启用模型和加载失败场景。
- 增加同卡多副本的显存与 GPU 竞争风险；报告和运行时诊断必须显式记录这些资源代价。
- 增加压测脚本、指标采集和中文报告，不新增跨 GPU 调度依赖。
