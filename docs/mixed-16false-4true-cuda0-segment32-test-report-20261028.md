# 混合并发压测报告：16 路关闭模型 + 4 路开启模型

日期：2026-10-28

## 测试条件

- 两组请求在同一个 8084 服务内同时发起
- Conda 环境：`asr`
- Qwen-ASR：`127.0.0.1:8806`
- `segment_concurrency=32`
- CAM++：8 个 FP16 实例，全部 `cuda:0`
- emotion2vec：8 个 FP16 实例，全部 `cuda:0`
- 音频：`test/audio_file/教师1.aac`，完整音频，约 3385.7 秒，单请求 519 个结果段

## 分组结果

| 分组 | 请求参数 | 并发 | 成功 | P50 | P95 | 最大值 |
|---|---|---:|---:|---:|---:|---:|
| A | `showSpk=false`，`showEmotion=false` | 16 | 16/16 | **600.288s** | **600.602s** | 600.602s |
| B | `showSpk=true`，`showEmotion=true` | 4 | 4/4 | **583.641s** | **600.422s** | 600.422s |

两组请求同时运行的总墙钟时间为 **600.617s**。A 组每个请求均返回 519 个段，role/emotion 均为空；B 组每个请求均返回 519 个 role 和 519 个 emotion 结果。

## 模型池指标

本轮只有 B 组调用 CAM++ 和 emotion2vec，各 1920 次：

| 模型池 | 实例 | 设备 | 平均推理 | 平均等待 | 最大等待队列 |
|---|---:|---|---:|---:|---:|
| CAM++ | 8 | `cuda:0` | 1858.06ms | 10212.46ms | 106 |
| emotion2vec | 8 | `cuda:0` | 723.68ms | 727.90ms | 73 |

GPU 0 测试后显存约 26396 MiB；GPU 1、GPU 2 未部署本轮模型。8084 关闭前所有请求均已成功返回。

## 原因分析

两组请求合计产生约 20 × 519 个 VAD 段。`segment_concurrency=32` 使大量 Qwen 段请求同时进入服务，16 路关闭模型请求也会占用 Qwen 的全局并发容量；日志显示 Qwen 等待约 10 秒/段。B 组还额外占用同一张卡上的 CAM++ 和 emotion2vec，CAM++ 平均推理约 1.86 秒，明显高于此前 4 路单独开启模型测试的约 0.69 秒，说明同卡 GPU 抢占和 Qwen 排队同时存在。

## 状态

- 8084 已关闭。
- 8806 Qwen-ASR 仍在监听。
- `config.toml` 未修改，本轮使用临时配置 `/tmp/asr_same_cuda0_fp16_seg32_mix.toml`。

原始数据：`/tmp/mixed_16false_4true_cuda0_seg32_20261028/mixed.json`
