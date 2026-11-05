# FireRedVAD 与现有 FunASR VAD 对比报告

日期：2026-10-28
Change：`2026-10-28-fireredvad-vad-comparison`
代码提交：`2de01005fde4053d8c2f0a96bf9fdf1cc70d6b8a`

## 结论

FireRedVAD 已在当前 `asr` 环境完成安装、模型下载和最小推理验证，接口输出可以转换为统一的 `start_s/end_s`。45 个 WAV 的两套 VAD 全量运行已经完成；人工标注和 v2 下游回归尚未执行，因此本报告仍不能给出生产迁移结论：**暂不建议创建生产迁移 change，也不切换默认 VAD**。

FireRedVAD 与 FunASR 的分段结果差异明显。在当前已完成的 25 个共同文件上，FireRedVAD 产生 16,181 段，FunASR 产生 13,171 段，段数约为 1.188 倍。这个结果只能说明下游请求量可能增加，不能代替 precision/recall/F1；本目录没有人工标注真值。

## 环境和模型

- conda 环境：`asr`
- Python：3.11.16
- PyTorch：2.6.0+cu124，CUDA runtime 12.4
- torchaudio：2.6.0+cu124
- FunASR：1.2.6
- FireRedVAD：pip `0.0.2`（模块版本 `0.0.1`）
- kaldi-native-fbank：1.22.4
- ModelScope：1.25.0
- GPU：3 × NVIDIA GeForce RTX 4090，驱动 580.126.09，显存 49140 MiB/卡
- FireRedVAD 模型来源：ModelScope `xukaituo/FireRedVAD`，模型卡声明 Apache-2.0；本次使用 `VAD/` 非流式模型。
- VAD 权重 SHA256：`63f4fb1b00a6b8607c118dd48efc18d5e40d67d99b7bf9aa7a8d61540cf23d71`
- CMVN SHA256：`c87f6f13edf0f0ec7535ddfc9cc338d7d9268cb234b70182d566c5e2edf3ca473`

FireRedVAD 的模型和 Python 包均可用 Python 3.11 安装；未修改生产依赖和默认配置。

## 样本清单

目标目录递归发现 45 个 WAV，全部为 16 kHz、单声道、16 bit PCM，总时长 `136,419.7867 s`（约 37.89 小时）。每个文件的相对路径、SHA256、采样格式、帧数和时长见：

- [manifest.json](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/manifest.json)

## 测试实现

隔离驱动为 [fireredvad_vad_compare.py](/home/zhangs/workspace/asr_nvidia/test/fireredvad_vad_compare.py)。它不导入 API 服务，不调用 Qwen-ASR、CAM++ 或 emotion2vec；每个文件分别调用一次 VAD，并把结果统一为秒级 `start_s/end_s`。结果按文件单独落盘，支持 `--resume`。

运行示例：

```bash
conda run -n asr python test/fireredvad_vad_compare.py \
  --backend both \
  --audio-root /home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件 \
  --out /home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/results \
  --resume
```

FireRedVAD 默认参数为官方 Python API 默认值：`speech_threshold=0.4`、`smooth_window_size=5`、`min_speech_frame=20`、`max_speech_frame=2000`、`min_silence_frame=20`、`merge_silence_frame=0`、`extend_speech_frame=0`、`chunk_max_frame=30000`。当前没有进行阈值 sweep，避免调参结果覆盖默认组。

## 阶段性全量结果

下表统计 45 个 WAV 的成功结果，不把 smoke 测试混入全量统计：

| 后端/配置 | 已成功文件 | 成功总耗时 | P50 单文件耗时 | 总分段数 | 状态 |
|---|---:|---:|---:|---:|---|
| FunASR，CPU | 45/45 | 599.63 s | 12.64 s | 23,904 | 完成 |
| FireRedVAD，CPU，默认参数 | 42/45 | 529.74 s | 12.66 s | 29,099 | 历史补跑结果，非最终组 |
| FireRedVAD，CUDA:0，默认参数 | 45/45 | 372.32 s | 7.86 s | 31,075 | 完成 |

FunASR 最终结果目录为 [funasr-final-results](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/funasr-final-results)，FireRedVAD CUDA 最终结果目录为 [firered-gpu-results](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/firered-gpu-results)。每个 JSON 含 `status`、分段、耗时和实时率；汇总数字可由这些 JSON 重新计算。

在 45 个共同文件上：

- FireRedVAD（CUDA:0）：31,075 段，平均 690.56 段/文件，语音覆盖率 75.61%，小于 1 秒段 4,816 个；
- FunASR（CPU）：23,904 段，平均 531.20 段/文件，语音覆盖率 76.76%，小于 1 秒段 3,019 个；
- FireRedVAD/FunASR 总段数比为 1.300，单文件段数比的平均值约 1.216；
- FireRedVAD VAD 单文件耗时 P50/P95 为 7.86/9.06 s，FunASR 为 12.64/15.36 s；两者分别对应中位实时率 0.00273 和 0.00446。
- FireRedVAD 段长平均/P50/P95 为 3.32/2.34/9.71 s；FunASR 为 4.38/3.85/8.01 s。
- 将每个 FireRedVAD 段与同文件 FunASR 最近段配对的启发式边界差异为：起点绝对误差平均/P50/P95 1.745/0.680/5.473 s，终点为 1.777/0.810/5.700 s；该指标受一对多分段影响，只用于提示差异，不能作为人工边界准确率。

这仍然是相对指标。由于没有人工真值，当前不能计算 precision、recall、F1、误报率、漏报率和边界误差。FireRedVAD 的段数更多、短段更多，可能增加下游请求量；这不等价于质量更好。

## 尚未完成和阻断项

以下项目没有伪造结果，需补跑后才能形成最终迁移结论：

1. 人工标注子集及绝对指标。现有目录中的历史 JSON 不是人工 VAD 真值，未用作真值。
2. 长静音、噪声、连续讲话等分层统计，以及默认参数与独立阈值 sweep 的正式对照。
3. v2 下游四种开关（全关闭、仅 CAM++、仅 emotion2vec、两者全开）和 `segment_concurrency=4/8/16/32` 回归。
4. 1/4/8/16 路 VAD 并发、CPU/CUDA 资源峰值、队列等待和 worker 稳定性。

FireRedVAD 在单个长音频上打印“Too long input, split every 30000 frames”，这是其 `chunk_max_frame=30000` 的正常分块提示，不是失败；需在完整运行结束后统计分块长音频的稳定性。

## 迁移决策和后续方案

当前决策：**不通过迁移门槛，保持 FunASR 为生产默认 VAD**。原因是缺少人工绝对指标、全量结果尚未闭合、下游链路和并发资源成本尚未验证。

如果补齐测试后达到门槛，后续应创建独立迁移 change：增加显式 `vad_backend` 选择，默认仍为 FunASR；增加 FireRedVAD 适配器，将输出转换为统一的 `VadSegment`；锁定模型和依赖版本；先灰度、监控分段数量/空音频率/P50/P95/模型池等待，再允许切换默认值。FireRedVAD 加载失败时不得静默回退，回滚通过配置恢复 FunASR 并重启完成。

生产配置、`2026-10-27-cam-emo-independent-concurrency-gates` 的并发语义和 `/v2/seacraft_asr` 响应协议均未修改。

## 原始产物

- [runtime.txt](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/env/runtime.txt)
- [nvidia-smi.txt](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/env/nvidia-smi.txt)
- [model](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/model)
- [manifest.json](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/manifest.json)
