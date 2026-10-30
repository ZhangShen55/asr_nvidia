# Design

## Context

当前 `core/models.py` 以模块级单例缓存 CAM++ 与 emotion2vec，`core/config.py` 只解析每个模型的单一 `cuda:N` 设备；`api/routes/asr_qwen_common.py` 通过 `asyncio.to_thread` 调用模型，并用一个本地 GPU semaphore 限制并发。完整服务的 `instance_count` 由 `start.sh` 启动多个 uvicorn 进程，每个进程都会复制模型。详见 `proposal.md` 的 Why，以及本变更的三个 delta spec。

## Goals / Non-Goals

**Goals:**

- 在单个服务进程内为 CAM++ 和 emotion2vec 分别维护可配置的多个模型副本。
- 通过 `[model_device.local_model_pool]` 的 `speech_campplus_devices` / `emotion2vec_devices` 统一配置实例设备，数组长度就是实例数，默认值为 `["cuda:0"]`。
- 让不同 VAD 段可以并行使用不同副本，同时保持 CAM++ embedding 在音频级统一聚类，保持 emotion2vec 的逐段输出契约。
- 在 `/get_status` 暴露池容量、加载、失败、忙碌/等待和显存信息，支持资源竞争和失败定位。
- 通过可重复的完整音频压测选择副本数，证明 2/4/8/16/32 路并发至少一个主要档位有吞吐或端到端效率提升。

**Non-Goals:**

- 不改变 `instance_count` 的完整服务进程语义，不用它作为模型池扩容开关。
- 不实现动态设备迁移、跨进程共享模型权重或独立模型服务进程；跨 GPU 实例仅由静态设备数组显式声明。
- 不对 CAM++ 做批量推理；既有验证表明零填充会改变 embedding，仍保持逐段推理。
- 不改变对外响应结构、角色聚类算法、情绪标签映射或 Qwen 服务协议。

## Decisions

### 1. 配置层级与兼容性

在现有设备配置下增加嵌套表：

```toml
[model_device]
speech_campplus = "cuda:0"
emotion2vec = "cuda:0"

[model_device.local_model_pool]
speech_campplus_devices = ["cuda:1", "cuda:2"]
emotion2vec_devices = ["cuda:1"]
```

`core/config.py` 提供两个设备数组读取属性，缺省为单元素 `cuda:0`，校验数组非空且每个元素为 `cuda:N`。数组长度决定模型池实例数，旧版数量字段仅作为兼容回退。选择嵌套表是为了保留已有 `[model_device]` 单设备配置和回滚路径。

### 2. 池对象与调度

在 `core/models.py` 将单例缓存替换为按模型名索引的池状态。每个池包含配置设备数组、懒加载的模型句柄、每个副本的异步租约锁、失败原因和运行计数。首次启用请求在模型锁内按配置创建副本；创建失败的副本进入隔离状态，不阻塞其它副本。

推理调用先从可用副本中按轮询并结合 busy 计数取得租约，随后在 `asyncio.to_thread` 中执行单段 `generate`，完成后释放租约。每个模型句柄同一时刻只由一个任务使用，以避免第三方模型对象的线程安全问题；现有全局 `local_gpu_concurrency` 继续作为 GPU 总并发上限，因此副本数增加不会绕过已有资源保护。若没有可用副本，调用沿用当前 role/emotion 的降级值 `null`。

选择进程内池和租约调度，而不是继续增加 uvicorn 进程，是因为进程扩容会复制 VAD、模型和服务状态，且无法让同一音频的段在一个统一聚类上下文中收敛；选择轮询/忙碌计数而不是随机分配，是为了在压测中获得稳定的副本利用率和可解释指标。

### 3. 结果归并边界

`api/routes/asr_qwen_common.py` 保持现有并行任务结构：每个 VAD 段同时发起 Qwen、CAM++ 和 emotion2vec 任务。CAM++ 任务只返回 CPU 上的标准化 embedding，所有段完成后仍由现有音频级 `cluster_speakers` 统一聚类和编号；因此不同副本不会形成各自的角色空间。emotion2vec 任务只返回当前段的业务标签，不保存跨段状态。此边界避免把池调度细节暴露到 API，也避免改变分段和时间轴。

### 4. 状态与显存统计

扩展 `local_model_status` 和 `/get_status` 的 `localModels` 字段：保留现有 `loaded`、`device`、`allocated_mb`、`instances`，并增加 `configured_instances`、`available_instances`、`failed_instances`、`busy`、`waiting` 与失败摘要。模型池加载前已加载数为 0；显存按设备和模型池累计登记，不能把同一卡上的其它进程显存误算到池内。状态采集失败只影响诊断字段，不影响推理。

### 5. 验证矩阵与报告

基准固定服务进程 `instance_count=1`、同一份完整音频集、同一 VAD/Qwen 配置和 `showSpk=true`、`showEmotion=true`。先跑单实例 baseline，再跑候选池规模（优先 2 和 4，显存允许时再扩展），每种配置执行并发 2、4、8、16、32，预热请求与正式样本分开统计。采集 P50/P95/max、音频/分钟、段/秒、GPU SM、显存峰值/增量、模型池 busy/waiting、Qwen waiting/KV、错误和质量指标。以同一档位的可重复吞吐提升作为推荐条件，显存 OOM、错误率或角色/情绪一致性退化则淘汰该配置，并将全部结果写入中文报告。

## Risks / Trade-offs

- **[同卡显存不足]** 多副本按权重近似线性增加显存，可能触发 OOM → 默认仍为 1，压测按 1/2/4 逐步扩容，启动和报告记录显存峰值，失败副本隔离。
- **[GPU 争用导致尾延迟上升]** 副本提高并发能力但可能让 SM/显存带宽饱和 → 保留全局 semaphore，报告同时给出吞吐和 P95/max，不以单一 P50 判断。
- **[模型对象线程安全]** 同一句柄并发 `generate` 可能产生不稳定结果 → 每个副本使用独立租约锁，仍允许不同副本并发。
- **[角色标签不一致]** 不同 CAM++ 副本若预处理或权重不一致会影响聚类 → 所有副本使用同一模型目录、精度和参数，并在报告中抽样比较 embedding 与角色结果。
- **[部分副本加载失败]** GPU 资源不足时池可能只有部分可用 → 状态接口暴露失败原因和可用数，推理继续使用健康副本；无健康副本时按既有 `null` 降级。

## Migration Plan

1. 先以未配置 `local_model_pool` 的现有配置启动，确认默认设备数组为 `["cuda:0"]`、响应和状态兼容。
2. 在测试配置中加入设备数组，按 `["cuda:0", "cuda:0"]`、`["cuda:0"]` 等规模逐级执行完整音频压测；需要跨卡时直接在数组中列出目标设备。
3. 选择满足吞吐提升且资源/质量指标可接受的副本数写入部署配置，并归档中文报告。
4. 若出现 OOM、错误率或质量回归，删除嵌套模型池配置并重启，自动回到单副本；无需回滚 `instance_count` 或接口协议。

## Open Questions

无。副本数量的最终生产值由压测报告决定，不影响本变更的配置和调度契约。
