# CAM++/emotion2vec 独立并发闸门 8 路压测报告

日期：2026-10-28
接口：`POST /v2/seacraft_asr`，服务端口 `8084`
Qwen-ASR：`8806`，测试期间保持运行
测试音频：`test/audio_file/教师1.aac`，完整音频约 3303.28 秒

## 测试配置

```toml
segment_concurrency = 4
audio_concurrency = 8

speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0",
                           "cuda:0", "cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0",
                       "cuda:0", "cuda:0", "cuda:0", "cuda:0"]
```

8 个请求同时提交，`showSpk=true`、`showEmotion=true`。测试完成后 8084 已关闭，8806 仍可访问。

## 端到端结果

| 指标 | 结果 |
|---|---:|
| 成功/失败 | 8/0 |
| P50 | **362.303 s** |
| P95 | **373.513 s** |
| 最大值 | 373.513 s |
| 吞吐 | 1.285 音频/分钟 |
| 每个请求响应段 | 519 |
| 每个请求 role/emotion 填充数 | 519/519 |

## 三路阶段耗时

日志中的阶段值是每个请求内所有 VAD 段的累计等待或推理时间。三路任务对同一段并行发起，因此这些累计值不能直接相加得到端到端时延；它们用于定位排队和计算热点。

| 阶段 | 调用数/请求 | P50 等待 | P95 等待 | P50 推理 | P95 推理 |
|---|---:|---:|---:|---:|---:|
| Qwen-ASR | 535 | 0.003 s | 0.004 s | 94.248 s | 102.465 s |
| CAM++ | 480 | **783.648 s** | **798.227 s** | 284.816 s | 295.708 s |
| emotion2vec | 480 | 0.665 s | 1.721 s | 104.486 s | 108.551 s |

模型池状态采样：CAM++ 最高 `busy=8`、`waiting=24`；emotion2vec 最高 `busy=8`、`waiting=1`。CAM++ 8 个实例全部在 CUDA:0 上运行，实例释放后会继续接收等待段，没有观察到明显空槽间隙。

## 落后原因

主要落后环节是 CAM++。`segment_concurrency=4` 只限制每个音频同时发往 Qwen 的段数；本轮 8 个音频最多有 32 个段级流程在途，而 CAM++ 只有 8 个实例。每个音频约 480 个 CAM++ 段，合计约 3840 个 CAM++ 调用，超出 8 个实例容量的部分进入 CAM++ FIFO。

同一张 CUDA:0 上同时放置 8 个 CAM++ 和 8 个 emotion2vec 还会产生 GPU 算力争用。emotion2vec 本身等待很少，但其推理与 CAM++ 共享 GPU 计算资源；增加实例数不会线性增加吞吐。Qwen 阶段约 94–102 秒，低于端到端 P50，且 Qwen 等待几乎为零，因此本轮不是 Qwen 排队主导。

原始数据：`/tmp/cam8-emo8-seg4-c8-20261028/concurrency_8.json`。
