# 理论计算机科学导论音频：三套分段结果对比

日期：2026-10-28
音频：`/home/zhangs/workspace/asr_nvidia/test/audio_file/理论计算机科学导论_时间am9.55_191803.wav`
时长：3,306.944 秒（约 55 分 7 秒）
格式：16 kHz、单声道、16 bit PCM

## 一、对比对象和测试方式

本次对同一份 WAV 对比三套分段：

1. **JSON 基线**：读取 `/home/zhangs/workspace/asr_nvidia/test/test_result/理论计算机科学导论_时间am9.55_191803.json` 的 `segments[].bg/ed`。
2. **FunASR 直接基线**：使用当前 FunASR FSMN-VAD 模型直接检测该 WAV，设备为 CPU，配置与当前服务默认 VAD 参数一致。
3. **FireRedVAD**：使用 `/home/zhangs/workspace/asr_nvidia/model/FireRedVAD/VAD`，非流式模型，设备为 CUDA:0，官方默认参数。

原始结果：

- [FunASR 原始结果](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/single-theory-cs/033b49bacdadb1fa.json)
- [FireRedVAD 原始结果](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/single-theory-cs/c4d08b5747ec99c5.json)

说明：提供的 JSON 不是严格合法 JSON，里面包含未加引号的人工注释，例如“细小声音”和“转写错误”。本次没有修改原文件，而是直接提取其中合法的 `bg/ed` 时间区间，共得到 658 个基线段。

## 二、总体结果

| 指标 | JSON 基线 | FunASR 直接基线 | FireRedVAD |
|---|---:|---:|---:|
| 分段数量 | 658 | 664 | 869 |
| 每分钟分段数 | 11.94 | 12.05 | 15.77 |
| 语音覆盖率 | 70.70% | 70.85% | 66.30% |
| 平均段长 | 3.553 秒 | 3.528 秒 | 2.523 秒 |
| 段长 P50 | 3.105 秒 | 3.085 秒 | 2.270 秒 |
| 段长 P95 | 8.010 秒 | 8.010 秒 | 5.140 秒 |
| 小于 1 秒段 | 66（10.03%） | 70（10.54%） | 148（17.03%） |
| 不小于 10 秒段 | 0 | 0 | 2（0.23%） |
| 段间隔均值 | 1.475 秒 | 1.454 秒 | 1.284 秒 |
| 段间隔 P50 | 0.650 秒 | 0.670 秒 | 0.600 秒 |
| 段间隔 P95 | 5.954 秒 | 5.736 秒 | 4.896 秒 |

## 三、结论

### 1. JSON 与直接 FunASR 基本一致

JSON 有 658 段，直接 FunASR 有 664 段，数量只相差 6 段，比例为 100.91%。两者重叠段的起止边界完全一致，最佳 IoU 为 1.0。

因此，在这份音频上，JSON 与当前 FunASR FSMN-VAD 的分段结果基本可以视为同一类基线。之前其他课堂音频出现较大差异，主要应归因于历史 JSON 的生成参数、版本或后处理差异，不能简单归因于低分贝过滤。

### 2. FireRedVAD 在这份音频上切得更碎

FireRedVAD 有 869 段，比 JSON 多 211 段，比直接 FunASR 多 205 段，分段数量约增加 31%。

同时：

- 平均段长从约 3.55 秒降到 2.52 秒；
- 段长 P50 从 3.10 秒降到 2.27 秒；
- 小于 1 秒的段从 66 个增加到 148 个；
- 语音覆盖率从约 70.7% 降到 66.3%。

这说明 FireRedVAD 在该音频上的行为是“切分更细，但检测到的总语音时长更少”。它可能把连续语音拆成更多小段，也可能过滤或漏掉一部分低能量语音，需要结合音频试听和人工标注判断。

### 3. FireRedVAD 的边界变化不是固定偏移

以 JSON 为 A、FireRedVAD 为 B：

| 指标 | 结果 |
|---|---:|
| JSON 段被 FireRedVAD 拆成至少 2 段 | 184 / 658（27.96%） |
| FireRedVAD 段覆盖多个 JSON 段 | 46 / 869（5.29%） |
| JSON 段没有 FireRedVAD 重叠 | 16 |
| FireRedVAD 段没有 JSON 重叠 | 14 |
| 起点差绝对值均值 | 0.671 秒 |
| 起点差绝对值 P50 | 0.230 秒 |
| 起点差绝对值 P95 | 3.248 秒 |
| 终点差绝对值均值 | 0.424 秒 |
| 终点差绝对值 P50 | 0.060 秒 |
| 终点差绝对值 P95 | 2.689 秒 |
| 最佳重叠 IoU 均值 | 0.748 |

FireRedVAD 主要表现为对原有段进行拆分，而不是简单整体前移或后移。

## 四、低分贝过滤是否造成 JSON 与 FunASR 差异

这份音频的结果不支持“低分贝过滤导致差异”这一判断：

- 当前配置 `vad_low_volume_filter_enabled = false`；
- 直接 FunASR 测试没有调用自定义 `filter_low_volume_segments()`；
- JSON 与直接 FunASR 只有 6 段差异，边界完全一致。

所以，本音频中 JSON 与 FunASR 的微小差异更可能是个别结果文件的输出整理或少量段有效性处理造成的，而不是低分贝过滤。

## 五、性能结果

| 后端 | 设备 | VAD 耗时 | 实时率 |
|---|---|---:|---:|
| FunASR | CPU | 12.560 秒 | 0.003798 |
| FireRedVAD | CUDA:0 | 12.853 秒 | 0.003887 |

这次单音频测试中，FunASR CPU 与 FireRedVAD CUDA 的耗时接近。该结果不能作为严格的设备性能结论，因为两者运行设备不同；本次重点是分段差异。

## 六、对下游链路的影响

如果每个 VAD 段都发送一次 Qwen-ASR、CAM++ 和 emotion2vec：

- FireRedVAD 会让该音频的段级任务数增加约 31%；
- CAM++/emotion2vec 的短段任务数量增加，调度次数和模型池调用次数会增加；
- 但单段更短，单次模型输入压力可能下降；
- 语音覆盖率下降 4.40 个百分点，需要先确认是否漏掉了真实语音；
- 不能按旧段索引直接复用 role/emotion 结果，必须基于 FireRedVAD 重新计算。

## 七、最终判断

对这份“理论计算机科学导论”音频：

```text
JSON 基线 ≈ FunASR 直接基线
FireRedVAD 分段数约增加 31%
FireRedVAD 平均段长约减少 29%
FireRedVAD 语音覆盖率约减少 4.4 个百分点
```

当前不能据此认定 FireRedVAD 更好。它在这份音频上的主要变化是切得更碎，同时覆盖到的语音总时长更少。建议优先人工抽查 FireRedVAD 独有的 14 个无基线重叠段，以及 JSON 段被拆分的 184 个样本，确认新增段是有效语音还是噪声/低能量误检。

本测试未修改生产 VAD 配置、服务代码或 Qwen-ASR 服务。
