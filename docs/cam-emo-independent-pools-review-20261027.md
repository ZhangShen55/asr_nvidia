# CAM++/emotion2vec 独立模型池评审与快速验证

## 快速验证

测试条件：8 路完整音频、`showSpk=true`、`showEmotion=true`、CAM++ 4 实例和 emotion2vec 4 实例全部在 `cuda:0`，VAD 固定 8 worker。

仅将：

```toml
local_gpu_concurrency = 8
```

结果：

| 指标 | `local_gpu_concurrency=4` | `local_gpu_concurrency=8` |
| --- | ---: | ---: |
| 成功率 | 8/8 | 8/8 |
| 端到端 P50 | 347.461 秒 | 304.574 秒 |
| 服务阶段中位数 | 316.826 秒 | 266.083 秒 |
| 音频吞吐 | 1.337 路/分钟 | 1.560 路/分钟 |

提升说明共享 GPU 并发闸门确实限制了 8 路处理，但运行中仍观察到 `CAM++ busy=4、waiting=3`，说明 CAM++ 的 4 个实例仍是该模型的容量上限。仅放大共享闸门不能突破每个模型池的实例数限制。

## 需求评审

### 1、CAM++ 与 emotion2vec 独立配置

同意。当前 `local_gpu_concurrency` 同时限制两类模型，无法表达不同模型的独立并发策略。应改为模型池自身提供容量，避免一个全局值把两类模型绑定在一起。

### 2、两类模型独立槽位

同意。CAM++ 和 emotion2vec 应分别维护自己的实例锁和等待队列：

- CAM++ 槽位数 = 已成功加载的 CAM++ 实例数
- emotion2vec 槽位数 = 已成功加载的 emotion2vec 实例数
- 两类模型互不占用对方槽位

当前 `ModelPool` 已经为每个实例维护独立 `asyncio.Lock`，所以底层能力已经存在；需要移除或改造外层共享 `_local_gpu_semaphore`。

### 3、是否可以直接由设备数组长度决定槽位

可以，这是推荐方案：

```toml
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0", "cuda:0"]
```

对应 CAM++ 3 个实例/3 个槽位，emotion2vec 2 个实例/2 个槽位。模型池本身的实例锁就是槽位控制，不再需要额外的 `local_gpu_concurrency` 配置。

需要以“成功加载实例数”作为实际容量，而不是只看数组长度；如果某个实例加载失败，实际槽位应自动减少，并在状态接口中暴露失败数。

### 4、配置表是否改为 `[model_device_pool]`

可以改，结构更直接：

```toml
[model_device_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0", "cuda:0"]
```

顶层 `[model_device]` 中的单设备字段在新配置中可以删除。当前代码中的 `model_device_emotion2vec`、`model_device_speech_campplus` 主要是旧兼容入口，迁移后应改为从设备数组第一个元素返回，短期保留读取兼容，避免外部调用方和旧配置直接失效。

## 实施计划

1. 增加统一的 `[model_device_pool]` 解析入口，优先读取新表；旧的 `[model_device.local_model_pool]` 继续兼容一个迁移周期。
2. 删除业务代码对全局 `_local_gpu_semaphore` 的依赖，直接使用两个 `ModelPool` 的实例锁作为独立槽位。
3. 将状态字段明确为 `configured_instances`、`loaded_instances`、`busy`、`waiting`、`failed_instances`，分别展示 CAM++ 和 emotion2vec。
4. 更新 `config.toml`、`config.docker.toml`、部署文档和配置契约测试；新配置不再要求 `[model_device]` 单设备字段。
5. 增加回归测试：3 个 CAM++ + 2 个 emotion2vec 时，两个池的最大并发分别为 3 和 2，互不抢槽；模拟实例加载失败时容量按成功实例数计算。
6. 做三组验收压测：4 路基线、8 路独立槽位、8 路旧全局槽位，对比 P50/P95、GPU 显存、模型池等待数和失败率。

## 风险与验收边界

移除全局槽位后，同一张 GPU 上 CAM++ 和 emotion2vec 最多可能同时运行“两个设备数组长度之和”的推理任务。若两者都配置 8 个实例，GPU 0 可能承受 16 路推理；需要以显存和 GPU 利用率压测结果决定是否增加可选的设备级总限流，而不是恢复一个绑定两种模型的全局模型槽位。
