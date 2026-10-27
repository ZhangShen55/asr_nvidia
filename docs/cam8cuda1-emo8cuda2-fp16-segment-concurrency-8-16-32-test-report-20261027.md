# CAM++/emotion2vec 隔卡 FP16 压测报告

日期：2026-10-27

## 测试条件

- Conda 环境：`asr`
- 测试服务：`127.0.0.1:8084`，测试完成后关闭
- Qwen-ASR：`127.0.0.1:8806`，测试期间保持运行
- GPU：NVIDIA GeForce RTX 4090
- CAM++：8 个 FP16 实例，全部部署在 `cuda:1`
- emotion2vec：8 个 FP16 实例，全部部署在 `cuda:2`
- 请求并发：8
- `showSpk=true`，`showEmotion=true`
- 音频：`test/audio_file/教师1.aac`，完整音频，约 3385.7 秒，单请求 519 个结果段

## 结果

| `segment_concurrency` | 成功 | P50 | P95 | 最大值 | 处理吞吐 |
|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 412.720s | 420.686s | 420.686s | 9.869 段/s |
| 16 | 8/8 | 421.816s | 428.224s | 428.224s | 9.693 段/s |
| 32 | 8/8 | 405.952s | 405.981s | 405.981s | 10.226 段/s |

## 模型池指标（每档 3840 次调用）

| 档位 | CAM++ 平均推理 | CAM++ 平均等待 | emotion2vec 平均推理 | emotion2vec 平均等待 |
|---:|---:|---:|---:|---:|
| 8 | 727.38ms | 4781.61ms | 316.68ms | 21.31ms |
| 16 | 720.24ms | 10224.42ms | 320.54ms | 221.92ms |
| 32 | 686.85ms | 19679.92ms | 305.57ms | 1145.37ms |

所有请求均成功返回，8/8 请求每个均返回 519 个 role 和 519 个 emotion 结果。状态接口确认 CAM++ 的 8 个实例设备均为 `cuda:1`，emotion2vec 的 8 个实例设备均为 `cuda:2`。

## 与同卡 FP16 对比

同样的 8 路完整音频、8 个 CAM++ + 4 个 emotion2vec、全部 CUDA 0 的 FP16 结果为 P50 485.583s（segment 8）、483.198s（segment 16）、501.288s（segment 32）。隔卡后分别变化为：

| `segment_concurrency` | 同卡 FP16 P50 | 隔卡 FP16 P50 | 改善 |
|---:|---:|---:|---:|
| 8 | 485.583s | 412.720s | 72.863s（15.0%） |
| 16 | 483.198s | 421.816s | 61.382s（12.7%） |
| 32 | 501.288s | 405.952s | 95.336s（19.0%） |

隔卡后 CAM++ 平均推理从约 896–945ms 降至约 687–727ms，emotion2vec 平均等待大幅下降；说明同一张卡上的 CAM++/emotion2vec 算力争用是此前延迟的重要来源。`segment_concurrency=32` 在隔卡条件下取得最佳 P50，但 CAM++ 和 emotion2vec 的累计排队仍会上升，继续增加段并发没有线性收益。

## 结论

- 隔卡部署有效改善了 FP16 场景，最佳 P50 从 501.288s 降至 405.952s。
- 当前最佳组合是 `CAM++ 8@cuda:1`、`emotion2vec 8@cuda:2`、`segment_concurrency=32`。
- P50 仍未达到 300s 目标，剩余时间主要来自 CAM++ 池排队、Qwen 阶段和完整音频的段级调度。
- 8084 已关闭，8806 Qwen-ASR 仍在监听。

原始数据：

- `/tmp/cam8cuda1_emo8cuda2_fp16_seg8_20261027/concurrency_8.json`
- `/tmp/cam8cuda1_emo8cuda2_fp16_seg16_20261027/concurrency_8.json`
- `/tmp/cam8cuda1_emo8cuda2_fp16_seg32_20261027/concurrency_8.json`
