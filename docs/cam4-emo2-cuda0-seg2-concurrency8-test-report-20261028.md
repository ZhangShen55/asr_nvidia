# CAM++ 4 实例、emotion2vec 2 实例、segment_concurrency=2 测试报告

日期：2026-10-28
服务：SeaCraftASR 临时 8084 端口，Qwen-ASR 保持使用 8806 端口
音频：`test/audio_file/教师1.aac`，音频时长约 3303.28 秒
请求：8 路完整音频并发，`showSpk=true`，`showEmotion=true`

## 测试配置

```toml
[qwen_asr]
segment_concurrency = 2
audio_concurrency = 8
base_url = "http://127.0.0.1:8806"

[model_device.local_model_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0", "cuda:0"]
```

模型池启动状态确认：CAM++ 4/4 个实例、emotion2vec 2/2 个实例，均在 `cuda:0`；Qwen 服务为外部 8806 服务。

## 端到端结果

| 指标 | 数值 |
|---|---:|
| 并发请求数 | 8 |
| 成功/失败 | 8 / 0 |
| 单请求 P50 | **308.318 s** |
| 单请求 P95 | **319.389 s** |
| 最大耗时 | 319.389 s |
| 最短耗时 | 290.383 s |
| 每个请求输出段数 | 519 |
| 压测墙钟时间 | 319.444 s |

原始数据：`/tmp/cam4-emo2-seg2-c8-20261028/concurrency_8.json`。

## 阶段统计

服务日志中的阶段数值是**每个请求所有 VAD 段的累计等待/推理时间**，不是单个 VAD 段耗时；8 个请求的阶段累计值按请求计算分位数如下：

| 阶段 | P50 | P95 |
|---|---:|---:|
| Qwen 等待 | 3 ms | 4 ms |
| Qwen 推理累计 | 86.184 s | 94.708 s |
| CAM++ 等待累计 | **338.613 s** | **346.498 s** |
| CAM++ 推理累计 | 125.238 s | 135.534 s |
| emotion2vec 等待累计 | 4.816 s | 6.040 s |
| emotion2vec 推理累计 | 42.638 s | 49.661 s |

模型池状态采样还显示：

- CAM++ 最高 `busy=4`、`waiting=12`，4 个槽位全部用满并持续排队；总等待累计约 2700.2 秒，平均每次调用等待约 703.2 ms。
- emotion2vec 最高 `busy=2`、`waiting=5`；总等待累计约 38.9 秒，平均每次调用等待约 10.1 ms。
- 两个模型池各完成 3840 次调用；每个请求包含 480 次 CAM++ 和 480 次 emotion2vec 调用。

## 结论

把 `segment_concurrency` 从上一轮的 4 降为 2 后，端到端耗时下降到 P50 308.318 秒、P95 319.389 秒。当前配置已经把同一张卡上的竞争压力从 Qwen 侧部分转移到本地模型侧，但主要落后点仍是 CAM++：4 个 CAM++ 实例始终是满载状态，并产生明显 FIFO 等待。emotion2vec 只有少量等待，Qwen 的 HTTP 等待几乎为零，说明这轮测试的排队主因是 CAM++ 计算和其 4 实例容量，而不是 Qwen 请求队列。

与上一轮 `CAM++ 8 + emotion2vec 8 + segment_concurrency=4` 的 P50 362.303 秒相比，本轮 P50 减少约 54.0 秒（约 14.9%）。这说明降低段级并发减少了同卡 CUDA 争用；但 CAM++ 实例数降到 4 后，CAM++ 排队仍然明显，继续降低 `segment_concurrency` 不一定能线性提升，需结合 CAM++ 实例数量和 GPU 利用率一起调优。

注意：阶段累计时间不能直接相加得到端到端耗时，因为 Qwen、CAM++、emotion2vec 在段级流水线中重叠执行；端到端 P50/P95 应以 HTTP 请求耗时为准。
