# CAM++/emotion2vec 8+8 实例完整音频并发测试报告

## 测试条件

- 日期：2026-10-27
- 服务：`8084`，`instance_count=1`，测试结束后已关闭
- Qwen-ASR：`8806`，测试前后均保持运行
- 音频：`test/audio_file/教师1.aac`，时长 `3303.280354` 秒
- 请求：8 路并发，`showSpk=true`、`showEmotion=true`
- VAD：CPU，固定 `vad_pool_size=8`、`vad_pool_max=8`
- 本地 GPU 并发：`local_gpu_concurrency=8`
- CAM++：8 个实例，全部 `cuda:0`
- emotion2vec：8 个实例，全部 `cuda:0`

## 结果

| 并发 | 成功/总数 | 墙钟时间 | P50 | P95 | 最大值 | 音频吞吐 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8/8 | 778.308 s | 771.457 s | 778.304 s | 778.304 s | 0.617 路/分钟 |

请求均返回 200，均产出 519 个结果段；没有 CUDA OOM、模型加载失败或 Qwen HTTP 错误。
首次请求包含 16 个本地模型副本的懒加载，GPU 0 显存峰值约 `23.6 GB / 49.1 GB`。
最终状态为 CAM++ `8/8`、emotion2vec `8/8` 加载成功，`failed_instances=0`。

## 对比与结论

已有同音频、8 路、4+4 实例且 `local_gpu_concurrency=8` 的结果为 P50 `304.574 s`、
P95 `307.632 s`。本次 8+8 的 P50 增加约 `153%`，没有达到提升目标。

增加同一张 GPU 上的副本没有形成等比例吞吐提升；8+8 会让 CAM++ 与 emotion2vec 在
`cuda:0` 上争用算力和内存带宽，并增加启动加载时间。当前不建议生产使用 8+8；保留
设备数组按实例配置的能力，实例数应由同卡压测和显存余量共同决定。

## 空数组契约

现在允许：

```toml
speech_campplus_devices = []
emotion2vec_devices = []
```

空数组不会拉起对应模型实例。请求开启 `showSpk` 时返回 `4015`（CAM++ 未启用或无可用
实例）；开启 `showEmotion` 时返回 `4016`（emotion2vec 未启用或无可用实例）。未开启
对应字段时不报错，Qwen 转写仍可执行。

原始压测数据：`/tmp/cam_emo_pool_8x8_vad8_gpuconc8_c8_20261027/summary.json`。
