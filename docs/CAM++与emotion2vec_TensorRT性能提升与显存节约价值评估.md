# CAM++ 与 emotion2vec TensorRT 性能提升与显存节约价值评估

## 结论

本次 TensorRT 接入的主要价值是缩短 CAM++、emotion2vec 的推理时间，减少 CAM++ 模型池排队，并提高同一张 GPU 在完整课堂并发场景下的处理吞吐。以当前 RTX 4090、CAM++ 3 实例、emotion2vec 1 实例、8 路完整音频并发的实测结果为依据：

- TensorRT FP16 相对当前 PyTorch CUDA FP16，完整请求 P50 从 407.003 秒降至 97.016 秒，约 **4.20 倍加速**。
- 完整八路批次墙钟从 431.900 秒降至 108.291 秒，下降 **74.93%**，每批八门课节省约 **323.609 秒（5.39 分钟）**。
- 音频吞吐从 56.599 倍实时提高到 225.737 倍实时，约 **3.99 倍**。
- GPU 总显存峰值从 24163 MiB 降至 23713 MiB，表面下降 **450 MiB（1.86%）**。这个数字包含同卡 Qwen、其他进程和 TensorRT 原生分配的影响，不能作为 CAM++ 或 emotion2vec 单独节省 450 MiB 的结论。

因此，当前可以确认 TensorRT 的核心收益是性能和并发处理能力；显存节约收益目前只能评估为有限且未被单模型精确测量。生产上线时应优先按性能收益评估，显存部分使用隔离环境和高频采样重新验收。

## 1. 测试范围

本评估对应 [CAM++ / emotion2vec TensorRT 服务集成验证报告](cam-emo-tensorrt-backends-validation-20261102.md)，测试条件如下：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090，SM89 |
| CUDA / PyTorch / TensorRT | CUDA 12.4 / PyTorch 2.6.0+cu124 / TensorRT 10.7.0.post1 |
| 完整音频 | 8 门不同课程，总时长 24445.131 秒 |
| 请求并发 | 8 路完整音频，`showSpk=true`、`showEmotion=true` |
| CAM++ 模型池 | `speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]` |
| emotion2vec 模型池 | `emotion2vec_devices = ["cuda:0"]` |
| Qwen / 分段 | 同一 Qwen 服务，`segment_concurrency=16` |
| 统计方式 | 每种后端 8 个成功请求；P50/P95 使用线性插值 |

PyTorch 基线使用当前服务的 PyTorch CUDA FP16 路径。FP32 和 FP16 TensorRT 均使用相同的音频、VAD、Qwen 请求集合和模型池配置。

本次每种后端只执行一轮、每轮 N=8。P95 是该批样本的描述值，适合说明本次对比，不代表所有负载下的稳定生产尾延迟。

## 2. 完整请求的性能收益

| 后端 | P50（秒） | P95（秒） | 整批墙钟（秒） | 音频吞吐（倍实时） | 相对 PyTorch P50 | 相对 PyTorch 整批墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch CUDA（原 FP16） | 407.003 | 428.395 | 431.900 | 56.599 | 1.00 倍 | 1.00 倍 |
| TensorRT FP32 | 115.822 | 123.924 | 124.841 | 195.810 | 3.51 倍 | 3.46 倍 |
| TensorRT FP16 | 97.016 | 107.217 | 108.291 | 225.737 | **4.20 倍** | **3.99 倍** |

TensorRT FP16 相对 PyTorch 基线的直接变化为：

- P50 减少 309.987 秒，下降 76.16%。
- P95 减少 321.179 秒，下降 74.97%。
- 整批墙钟减少 323.609 秒，下降 74.93%。
- 吞吐提高 298.83%，即约 3.99 倍。

TensorRT FP32 也有明显收益，但比 FP16 慢：P50 约 3.51 倍、整批墙钟约 3.46 倍。FP32 适合在精度边界更敏感的场景作为保守后端。

### 2.1 单段模型推理收益

单段基准使用 48 个课堂裁剪，每个输入预热后测 3 次，共 144 次/模型/后端；该基准排除了模型池排队，但仍包含特征、拷贝、同步和 CPU 输出读取。

| 模型 | 指标 | PyTorch CUDA FP16 | TensorRT FP16 | 加速倍数 | 延迟下降 |
| --- | --- | ---: | ---: | ---: | ---: |
| CAM++ | P50（毫秒） | 40.543 | 6.801 | 5.96 倍 | 83.23% |
| CAM++ | P95（毫秒） | 126.371 | 8.340 | 15.15 倍 | 93.40% |
| emotion2vec | P50（毫秒） | 16.145 | 1.832 | 8.81 倍 | 88.65% |
| emotion2vec | P95（毫秒） | 28.879 | 3.546 | 8.14 倍 | 87.72% |

完整请求加速倍数低于单段模型加速倍数，是因为完整链路还包含上传、VAD、Qwen 请求、分段调度、模型池等待和结果汇总。不能直接用单段推理倍数承诺端到端倍数。

### 2.2 排队瓶颈变化

八路完整课堂测试中，CAM++ 累计等待时间由 42004.545 秒降至 351.268 秒，下降约 **99.16%**。这说明 TensorRT 解决了当前配置下 CAM++ 模型池的主要排队压力。

emotion2vec 只有一个实例。启用 TensorRT FP16 后，emotion2vec 累计等待时间为 7352.087 秒，高于 PyTorch 的 5860.042 秒，增加约 25.46%。这不是 TensorRT 单段推理变慢，而是 CAM++ 加速后，更多分段更快到达唯一的 emotion2vec 实例，剩余瓶颈集中到了 emotion2vec 模型池。

这意味着后续如果仍需降低端到端 P95，应单独评估 `emotion2vec_devices` 增加实例的收益和显存代价，不能只继续提高 `segment_concurrency`。

## 3. 显存节约价值

### 3.1 当前 GPU 总显存观测

测试使用 `nvidia-smi` 约每 2 秒采样一次，峰值包含 Qwen、CAM++、emotion2vec、TensorRT 原生分配以及同卡其他进程：

| 后端 | GPU 显存峰值（MiB） | 相对 PyTorch | 相对变化 |
| --- | ---: | ---: | ---: |
| PyTorch CUDA（原 FP16） | 24163 | — | — |
| TensorRT FP32 | 24145 | 少 18 MiB | 下降 0.07% |
| TensorRT FP16 | 23713 | 少 450 MiB | 下降 1.86% |

这组结果支持“TensorRT FP16 在本次同卡负载下没有增加总显存，并观测到小幅下降”，但不支持“CAM++ 和 emotion2vec 合计稳定节省 450 MiB”或“TensorRT 一定节省数 GB 显存”。原因有三点：

1. 峰值采样间隔可能漏掉瞬时峰值。
2. Qwen 和其他进程的显存波动会覆盖本地模型的变化。
3. TensorRT 的 engine、context、workspace 和 CUDA 原生分配不能全部由 `torch.cuda.memory_allocated()` 统计。

### 3.2 状态接口中的显存口径

PyTorch 路径可以记录加载时的 PyTorch allocator 增量。本次基线中，CAM++ 3 个实例和 emotion2vec 1 个实例合计约 353.83 MiB：

- CAM++ 3 实例：40.31 MiB。
- emotion2vec 1 实例：313.52 MiB。

TensorRT 路径的 `allocated_mb` 有意显示为 `null`，避免把未统计到的原生分配误报成 0。状态接口另记录可观测的 context 和 engine 文件信息：

| TensorRT 精度 | CAM++ context（单实例） | emotion2vec context | engine 文件大小 |
| --- | ---: | ---: | ---: |
| FP32 | 62.28 MiB | 197.44 MiB | CAM++ 30.07 MiB；emotion2vec 637.64 MiB |
| FP16 | 32.23 MiB | 126.09 MiB | CAM++ 22.83 MiB；emotion2vec 321.02 MiB |

本次配置有 3 个 CAM++ context，因此 FP16 可观测 context 合计约 222.77 MiB，FP32 约 384.27 MiB。engine 文件大小是序列化文件大小，context bytes 是 TensorRT 执行上下文的设备内存提示；二者都不能直接当成完整 GPU 常驻显存，也不能与 PyTorch allocator 增量做严格一对一相减。

### 3.3 显存价值的实际判断

- **当前可确认的价值**：TensorRT FP16 在同卡 Qwen 场景下没有造成显存峰值上升，并观测到 1.86% 的总显存下降。
- **当前不能确认的价值**：单模型精确节省量、增加模型实例后每个实例的显存曲线、跨 GPU 架构的固定节省量。
- **工程上的潜在价值**：如果在隔离 GPU 或更高模型池实例数下，TensorRT 的原生分配继续低于 PyTorch，则可把节省出的空间用于增加 emotion2vec 实例或保留 Qwen KV cache；这需要专门的显存容量实验验证。

## 4. 质量和上线风险

性能收益不能脱离输出一致性判断：

- CAM++ TensorRT FP16 在 8 门完整课程的角色聚类结果与当前 PyTorch FP16 保持一致，8 门课程的角色 ARI 均为 1.0；但没有人工说话人真值，这表示系统一致性，不表示绝对准确率。
- emotion2vec TensorRT FP16 在共同输出段上的业务情绪标签一致率为 99.79684%，有 9 个标签差异；刻意选取的 11 个边界片段只有 2 个标签与基线相同，最大概率绝对差约 3.974 个百分点。
- emotion2vec TensorRT FP32 的共同输出段一致率为 99.93225%，边界样本仍需按业务接受标准复核。
- NPU 只完成设备路由和导入隔离测试，没有完成 torch_npu 实机性能或质量验收。
- engine 与 GPU 架构、TensorRT、CUDA 版本绑定。当前结果针对 RTX 4090 / SM89 / TensorRT 10.7.0.post1 / CUDA 12.4，迁移到其他环境需要重新构建和验证。

## 5. 建议的上线策略

1. **优先验证 CAM++ TensorRT FP16**：端到端收益明显，角色聚类在本次 8 门课中与基线一致，适合先做小流量启用。
2. **emotion2vec 按精度要求选择**：若情绪标签边界稳定性优先，先评估 TensorRT FP32；若业务能接受边界标签变化，再评估 TensorRT FP16。
3. **继续保留四个配置项和默认关闭策略**：

   ```toml
   speech_campplus_use_tensorrt = false
   speech_campplus_tensorrt_precision = "fp16"
   emotion2vec_use_tensorrt = false
   emotion2vec_tensorrt_precision = "fp16"
   ```

   `cuda:N` 且开关为 `true` 时使用 TensorRT；`cuda:N` 且开关为 `false` 时使用 PyTorch CUDA；`npu:N` 固定使用 torch_npu 并忽略 TensorRT 开关；CPU 固定使用 PyTorch CPU。

4. **监控实际后端和排队指标**：上线后观察 `effective_backend`、模型池 `waiting`、P50/P95、4015/4016 错误、GPU 显存峰值和失败实例数。
5. **若要验证显存节约**：在无其他业务的隔离 GPU 上分别只加载 Qwen、只加载本地模型和加载完整链路；使用至少 30 次重复启动/压测，并以高频 `nvidia-smi` 或 CUDA memory snapshot 记录峰值，再比较不同模型实例数。
6. **若要继续降低端到端尾延迟**：先对 `emotion2vec_devices=1/2` 做受控对比，再决定是否调整 `segment_concurrency`；不能将模型已经变快等同于提高所有并发参数都能继续提速。

## 6. 复算依据

完整方法、精确样本、逐请求记录、质量差异、回滚结果和 240 项测试结果见 [CAM++ / emotion2vec TensorRT 服务集成验证报告](cam-emo-tensorrt-backends-validation-20261102.md)。后端选择、模型池、engine manifest 和回滚规则见 [TensorRT 使用说明](TENSORRT.md)。

本文件中的“加速倍数”按基线耗时除以 TensorRT 耗时计算；“下降比例”按（基线值 - TensorRT 值）/ 基线值计算。所有显存结论均保留采样口径，不将总卡显存峰值解释为单模型显存。
