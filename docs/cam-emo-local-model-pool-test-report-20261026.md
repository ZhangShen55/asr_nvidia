# CAM++/emotion2vec 同卡模型池完整音频并发测试报告

## 1. 测试范围

- 服务环境：Conda `asr`，本服务端口 `8084`，`instance_count=1`。
- Qwen-asr：继续使用已运行的 `127.0.0.1:8806`，测试过程中未停止或重启。
- GPU：CAM++ 与 emotion2vec 的全部副本固定在 `cuda:0`；GPU 0 为 RTX 4090 49,140 MiB。
- 音频：`test/audio_file/教师1.aac`，约 56 分钟；统一预转换为 `/tmp/cam_emo_teacher1.wav` 后作为完整音频输入。
- 请求：`showSpk=true`、`showEmotion=true`；每档同时提交完整音频副本。
- 基线：CAM++/emotion2vec 各 1 个副本；模型池：各 2 个副本。

## 2. 端到端结果

以下 2/4/8/16 档均使用同一份预转换 WAV，排除了 AAC 重复转码差异。吞吐为成功音频数/分钟。

| 音频并发 | 单实例 wall(s) | 单实例吞吐 | 模型池 wall(s) | 模型池吞吐 | 吞吐变化 | 成功情况 |
|---:|---:|---:|---:|---:|---:|:---:|
| 2 | 114.606 | 1.047 | 100.921 | 1.189 | **+13.6%** | 2/2 vs 2/2 |
| 4 | 165.026 | 1.454 | 138.477 | 1.733 | **+19.2%** | 4/4 vs 4/4 |
| 8 | 307.470 | 1.561 | 262.170 | 1.831 | **+17.3%** | 8/8 vs 8/8 |
| 16 | 672.002 | 1.429 | 521.588 | 1.841 | **+28.8%** | 16/16 vs 16/16 |
| 32 | 未完成 | - | 未完成 | - | 不判定 | 见下方边界说明 |

2/4/8/16 四个主要档位均有可重复吞吐提升，满足“至少一个主要档位提升”的验收条件；当前推荐先使用 CAM++ 2 副本、emotion2vec 2 副本，并继续观察生产流量下的尾延迟。

## 3. 32 路边界

32 路完整音频会同时产生约 17,000 个段级任务，触发服务的长尾资源竞争：

- 单实例预压测中出现 VAD 排队超时；客户端停止时已完成 16 个请求、5 个明确失败，剩余请求未收齐。
- 2 副本模型池压测客户端运行约 20 分钟时已完成 30/32 个请求且失败数为 0，剩余 2 个请求长期占用两个 CAM++ 副本，未在本轮窗口内结束。
- 该档位保留为“高并发保护/长尾边界”，不使用不完整样本计算提升百分比。32 路的主要卡点从单副本 CAM++ 串行，转移为 VAD/音频预处理和 GPU 任务长尾。

## 4. 模型池与资源观测

模型池 2 副本状态接口确认：

```json
{
  "speech_campplus": {
    "device": "cuda:0",
    "instances": 2,
    "configured_instances": 2,
    "allocated_mb": 53.17,
    "failed_instances": 0
  },
  "emotion2vec": {
    "device": "cuda:0",
    "instances": 2,
    "configured_instances": 2,
    "allocated_mb": 628.55,
    "failed_instances": 0
  }
}
```

相对单副本约 340.11 MiB，2 副本约 681.72 MiB；GPU 0 总显存由 Qwen 服务占用约 26–27 GiB，测试未出现 CUDA OOM。模型池运行时可观察到 CAM++ `busy=2`，说明两个副本均参与推理；同卡 GPU 利用率在高并发阶段达到 85–100%，因此继续增加副本可能提高吞吐，也可能放大 P95/max 尾延迟。

## 5. 质量一致性

对同一完整 WAV 分别执行单副本和 2 副本单请求全开比对：

| 指标 | 结果 |
|---|---:|
| 响应成功 | 2/2 |
| 响应分段数 | 518 vs 518 |
| 分段 `bg`/`ed` 完全一致 | 518/518 |
| `role` 完全一致 | 518/518 |
| `emotion` 完全一致 | 518/518 |
| `segment_text` 完全一致 | 506/518 |

文本有 12 段差异，属于 Qwen 上游重复请求的非确定性；分段时间轴、角色聚类标签和情绪标签没有变化。模型池不会把 CAM++ 分段按副本分别聚类，所有 embedding 仍在音频级统一聚类。

## 6. 原始数据与复现

- 单实例 WAV：`/tmp/cam_emo_pool_baseline_wav_low/`，16 档：`/tmp/cam_emo_pool_baseline_wav/concurrency_16.json`。
- 2 副本 WAV：`/tmp/cam_emo_pool_2_wav/`，16 档：`/tmp/cam_emo_pool_2_wav_hi/concurrency_16.json`。
- 压测脚本：[bench_cam_emo_model_pool.py](/home/zhangs/workspace/asr_nvidia/scripts/bench_cam_emo_model_pool.py)。

```bash
conda run -n asr python scripts/bench_cam_emo_model_pool.py \
  --base-url http://127.0.0.1:8084 \
  --audio /tmp/cam_emo_teacher1.wav \
  --concurrency 2,4,8,16,32 \
  --out-dir /tmp/cam_emo_pool_2_wav
```

测试结束后本次 8084 后端已关闭；Qwen-asr 仍保持运行。
