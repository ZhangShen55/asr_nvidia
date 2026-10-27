# CAM++ 8 实例、emotion2vec 4 实例：segment_concurrency 压测报告

## 测试条件

- 音频：test/audio_file/教师1.aac，完整音频，时长约 3385.722 秒
- 并发：8 路完整音频
- CAM++：8 个实例，全部部署在 cuda:0
- emotion2vec：4 个实例，全部部署在 cuda:0
- 请求：showSpk=true、showEmotion=true
- Qwen-ASR：8806，测试期间保持运行
- 本地服务：8084，测试完成后关闭

## P50/P95 结果

| segment_concurrency | 成功 | Wall time | P50 | P95 | 吞吐 |
|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 386.813 s | 380.380 s | 386.751 s | 1.241 路/分钟 |
| 16 | 8/8 | 389.009 s | 377.770 s | 388.947 s | 1.234 路/分钟 |
| 32 | 8/8 | 384.077 s | 375.095 s | 384.019 s | 1.250 路/分钟 |

目标 P50 约 300 秒，本次最低 P50 为 375.095 秒，仍高出约 75 秒，约 25.0%。

## 排队与阶段指标

| segment_concurrency | CAM++ 最大等待 | CAM++ 累计等待 | emotion2vec 最大等待 | emotion2vec 累计等待 |
|---:|---:|---:|---:|---:|
| 8 | 56 | 17288.206 s | 47 | 468.591 s |
| 16 | 120 | 37255.871 s | 112 | 1973.794 s |
| 32 | 248 | 75274.350 s | 233 | 8405.544 s |

三个档位都确认 8 个 CAM++ 和 4 个 emotion2vec 实例成功加载。提高 segment_concurrency 没有增加模型实例容量，只是让更多分段提前进入模型池等待队列。32 档的 CAM++ 等待量约为 8 档的 4.4 倍，emotion2vec 等待量约为 18 倍。

## 结论

segment_concurrency=8、16、32 的端到端 P50 都处于 375~380 秒平台，不能单靠提高该参数达到 300 秒。推荐保留 8 或 16 作为后续调优起点；32 档等待队列和 Qwen 阶段波动更大，没有稳定收益。

本轮主要瓶颈是 CAM++ 模型池的实际推理和排队，以及同一张 GPU 上 8 个 CAM++ 与 4 个 emotion2vec 的算力争用。Qwen 队列在 8、16 档等待很低，32 档出现明显阶段波动，但不是唯一主因。若要继续逼近 300 秒，应针对 GPU 0 的模型组合、CAM++ 单次推理耗时和音频分段数量继续优化，而不是继续增大 segment_concurrency。

原始数据：

- /tmp/cam8_emo4_seg8_20261027/concurrency_8.json
- /tmp/cam8_emo4_seg16_20261027/concurrency_8.json
- /tmp/cam8_emo4_seg32_20261027/concurrency_8.json
