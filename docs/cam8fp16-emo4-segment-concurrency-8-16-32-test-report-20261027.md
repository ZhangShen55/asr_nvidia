# CAM++ FP16：8 路完整音频压测报告

日期：2026-10-27

## 测试条件

- Conda 环境：`asr`
- 接口：`POST /v2/seacraft_asr`
- 测试服务：`127.0.0.1:8084`，测试完成后关闭
- Qwen-ASR：`127.0.0.1:8806`，测试期间及测试后保持运行
- GPU：NVIDIA GeForce RTX 4090，所有本地模型均使用 `cuda:0`
- CAM++：8 个实例，全部 FP16
- emotion2vec：4 个实例，全部 FP16
- 请求并发：8
- `showSpk=true`，`showEmotion=true`
- 音频：`test/audio_file/教师1.aac`，完整音频，约 3385.7 秒，单请求 519 个结果段

## 结果

| `segment_concurrency` | 成功 | P50 | P95 | 最大值 | 处理吞吐 |
|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 485.583s | 495.743s | 495.743s | 8.371 段/s |
| 16 | 8/8 | 483.198s | 484.836s | 484.836s | 8.563 段/s |
| 32 | 8/8 | 501.288s | 511.317s | 511.317s | 8.119 段/s |

## 模型池指标（单档 3840 次调用）

| 档位 | CAM++ 平均推理 | CAM++ 平均等待 | emotion2vec 平均推理 | emotion2vec 平均等待 |
|---:|---:|---:|---:|---:|
| 8 | 895.75ms | 6030.85ms | 327.31ms | 107.47ms |
| 16 | 902.77ms | 12701.84ms | 327.09ms | 459.61ms |
| 32 | 944.99ms | 27020.56ms | 353.00ms | 2966.64ms |

所有请求的 CAM++ role 和 emotion2vec emotion 字段均成功返回，8/8 请求每个均为 519 个 role、519 个 emotion 结果。

## 与 FP32 基线比较

此前相同音频、8 路、8 个 CAM++ + 4 个 emotion2vec 的 FP32 基线为：

| `segment_concurrency` | FP32 P50 | FP16 P50 | 变化 |
|---:|---:|---:|---:|
| 8 | 380.380s | 485.583s | +105.203s |
| 16 | 377.770s | 483.198s | +105.428s |
| 32 | 375.095s | 501.288s | +126.193s |

本次 FP16 没有达到提速目标。真实修改 `segment_concurrency` 后，P50 随并发从 485.583s（8）升到 483.198s（16）和 501.288s（32）；CAM++ 平均前向约 896–945ms，CAM++ 平均等待从 6.03s 升到 12.70s、27.02s。emotion2vec 平均等待也从 0.11s 升到 0.46s、2.97s。当前 RTX 4090 上，8 个 CAM++ 与 4 个 emotion2vec 同卡并行时，FP16 不能直接作为性能优化方案；`segment_concurrency=32` 会显著放大模型池排队。

## 结论

- CAM++ FP16 已实现并完成三档完整音频验证，结果正确性通过。
- `segment_concurrency=16` 与 `32` 没有改善，32 反而加重模型池排队；本轮最佳 P50 为 **483.198s**。
- 本轮仍未达到 P50 约 300s 的目标。
- 8084 测试服务已关闭；8806 Qwen-ASR 仍在监听。
- 测试完成后 `config.toml` 已恢复为默认单 CAM++、单 emotion2vec 和 `segment_concurrency=4`。

原始数据：

- `/tmp/cam8fp16_emo4_seg8_20261027/concurrency_8.json`
- `/tmp/cam8fp16_emo4_actual_seg16_20261027/concurrency_8.json`
- `/tmp/cam8fp16_emo4_actual_seg32_20261027/concurrency_8.json`
