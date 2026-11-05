# FireRedVAD 与 seaco Paraformer 分段对比报告

日期：2026-10-28
测试范围：`/home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件/`
对比基线：同目录下 45 个 `seaco_paraformer-*.json`
FireRedVAD：`/home/zhangs/workspace/asr_nvidia/model/FireRedVAD/VAD`

## 1. 这份报告怎么看

这次对比把每个 `seaco_paraformer-*.json` 的 `segments[].bg/ed` 当作现有 FSMN-VAD 分段基线，并使用同名 WAV 运行 FireRedVAD。45 个 JSON 与 45 个 WAV 全部一一匹配，没有缺失文件。

需要特别说明：本报告按你的定义把 JSON 分段作为 FSMN-VAD 基线，但从工程数据核验看，JSON 分段数量与此前直接调用当前 FunASR FSMN-VAD 的结果并不总是相同。例如同目录的部分课堂音频出现 `JSON 855 段、直接 FunASR 706 段`、`JSON 770 段、直接 FunASR 457 段`。这说明 JSON 可能经过了另一套 VAD 参数、后处理或不同版本链路。因而本报告的结论严格表示“FireRedVAD 与现有 seaco Paraformer 输出分段的差异”，不能把 JSON 自动当成当前代码实时 FSMN-VAD 的完全复现。

报告先回答“切得多不多、切得碎不碎、边界差多少”，再回答“哪些课程差异最大”。分段数量和时间边界只能说明两个模型的切分行为不同，不能单独证明哪个模型更准确；准确率、召回率和 F1 仍需要人工标注语音区间。

原始逐文件结果和全部指标见：

- [comparison.json](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/paraformer-comparison/comparison.json)
- [compare_firereds_vad_to_paraformer.py](/home/zhangs/workspace/asr_nvidia/test/compare_firereds_vad_to_paraformer.py)

## 2. 总体结论

在这批课堂录音上，FireRedVAD 的整体行为表现为：**分段数量更少、单段更长、短段更少、长段更多、语音覆盖率略低、相邻段之间的静音间隔更长**。

这说明 FireRedVAD 在当前默认参数下整体切分粒度比 seaco Paraformer 基线更粗。它可能把相邻的短语或短停顿保留在同一个段中，也可能过滤掉一部分基线识别到的短语音。FireRedVAD 不是在所有课程上都更粗：不同课程的段数比例变化很大，说明模型差异与音频内容、语言和录音环境有关。

当前不能据此直接切换生产 VAD，因为还没有人工语音区间真值，也没有把两套分段分别接入 Qwen-ASR/CAM++/emotion2vec 做下游回归。

## 3. 全量统计

样本总时长为 136,419.79 秒，约 37.89 小时。

| 指标 | seaco Paraformer JSON | FireRedVAD | FireRedVAD 相对基线 |
|---|---:|---:|---:|
| 文件数 | 45 | 45 | 相同 |
| 分段总数 | 40,490 | 31,075 | -23.25% |
| 每分钟分段数 | 17.81 | 13.67 | -23.25% |
| 语音覆盖率 | 79.39% | 75.61% | -3.78 个百分点 |
| 平均段长 | 2.675 秒 | 3.319 秒 | +24.11% |
| 段长 P50 | 1.98 秒 | 2.34 秒 | +18.18% |
| 段长 P95 | 6.92 秒 | 9.71 秒 | +40.32% |
| 小于 1 秒段 | 9,449（23.34%） | 4,816（15.50%） | -7.84 个百分点 |
| 不小于 10 秒段 | 737（1.82%） | 1,481（4.77%） | +2.95 个百分点 |
| 段间隔均值（逐文件均值的均值） | 0.680 秒 | 1.067 秒 | +0.387 秒 |
| 段间隔 P50（逐文件 P50 的中位数） | 0.120 秒 | 0.330 秒 | +0.210 秒 |
| 段间隔 P95（逐文件 P95 的中位数） | 2.200 秒 | 4.086 秒 | +1.886 秒 |

### 对分段粒度的直接解释

- 每分钟分段数从 17.81 降到 13.67，FireRedVAD 平均每分钟少产生约 4.14 个段。
- FireRedVAD 的段长 P50 增加 0.36 秒，P95 增加 2.79 秒，说明长尾段明显变长。
- 小于 1 秒的段减少约 49%，短促语音和短停顿相关的切分行为明显不同。
- 不小于 10 秒的段约增加 101%，FireRedVAD 更容易保留较长连续段。
- FireRedVAD 的段间隔更长，覆盖率低 3.78 个百分点，表示它检测出的语音总时长更少，或者边界更收缩。

## 4. 逐文件差异

下面的比例是 `FireRedVAD 分段数 / Paraformer 分段数`。比例低于 1 表示 FireRedVAD 切得更粗或检测到的语音更少，比例高于 1 表示 FireRedVAD 产生了更多段。

### 差异最大的低比例样本

| 文件 | Paraformer 段数 | FireRedVAD 段数 | 比例 |
|---|---:|---:|---:|
| 综合岩石学 / 2026-03-24 10:40 / 教师1.wav | 1,069 | 297 | 0.278 |
| 综合岩石学 / 2026-03-17 10:40 / 教师1.wav | 1,109 | 344 | 0.310 |
| 综合岩石学 / 2026-03-19 08:50 / 教师1.wav | 1,035 | 325 | 0.314 |
| 综合岩石学 / 2026-03-19 08:00 / 教师1.wav | 1,088 | 342 | 0.314 |
| 综合岩石学 / 2026-03-17 09:50 / 教师1.wav | 1,044 | 360 | 0.345 |

这些样本说明 FireRedVAD 在综合岩石学课堂录音上明显减少了段数，需要重点抽取音频片段确认它是在合并合理停顿，还是漏掉了短语音。

### 差异最大的高比例样本

| 文件 | Paraformer 段数 | FireRedVAD 段数 | 比例 |
|---|---:|---:|---:|
| 储能学术英语实践 / 教师1.wav | 754 | 1,162 | 1.541 |
| 科学的历程 / 教师1.wav | 1,018 | 1,355 | 1.331 |
| 多模态机器学习 / 09-25-03.wav | 743 | 846 | 1.139 |
| 高等传热学 / 10-50-09.wav | 681 | 769 | 1.129 |
| 苏州大学课程 / 通用学术英语Ⅱ / 教师1.wav | 699 | 759 | 1.086 |

这些样本说明 FireRedVAD 的行为不是固定地“更粗”。在英文或中英混合课堂、不同噪声环境下，它也可能切出更多段，必须结合音频波形和人工标注判断原因。

全体 45 个文件的分段比例中位数为 0.860，P95 为 1.139，说明多数文件 FireRedVAD 段数低于基线，但仍有少数文件反向增加。

## 5. 边界、拆分与合并

边界分析采用重叠诊断：对每个基线段，寻找与其重叠最大的 FireRedVAD 段；同时统计一个基线段与多个 FireRedVAD 段重叠的情况，以及一个 FireRedVAD 段与多个基线段重叠的情况。这些指标用于观察切分形态，不作为准确率。

| 指标 | 结果 |
|---|---:|
| 基线段被 FireRedVAD 拆成至少 2 段 | 11,434 / 40,490（28.24%） |
| FireRedVAD 段覆盖至少 2 个基线段 | 13,582 / 31,075（43.71%） |
| 基线段没有任何 FireRedVAD 重叠 | 268（0.66%） |
| FireRedVAD 段没有任何基线重叠 | 849（2.73%） |
| 起点差绝对值均值（按文件汇总） | 2.267 秒 |
| 起点差绝对值 P50（按文件汇总） | 1.781 秒 |
| 终点差绝对值均值（按文件汇总） | 2.604 秒 |
| 终点差绝对值 P50（按文件汇总） | 2.503 秒 |

结果显示两套边界不是简单的固定偏移，而是存在大量拆分和合并。FireRedVAD 总段数更少，但“一个 FireRedVAD 段覆盖多个基线段”的情况很多，这与其更长的段长和更高的长段比例一致。

## 6. 对下游 Qwen/CAM++/emotion2vec 的影响

按每个 VAD 段发送一次 Qwen、CAM++ 和 emotion2vec 的现有链路语义，FireRedVAD 在这批样本上预计会带来：

- 平均段数下降约 23%，理论上减少每个音频的下游请求数量。
- 单段平均时长增加约 24%，CAM++ 和 emotion2vec 单次输入变长，单次推理耗时可能增加。
- 10 秒以上长段增加约 101%，可能降低段级请求调度次数，但增加单段显存和推理时间压力。
- 覆盖率下降 3.78 个百分点，必须确认是否出现漏检短语音；否则会直接影响 Qwen 转写内容。
- 分段边界变化较大，CAM++ 的说话人嵌入和 emotion2vec 的情绪结果不能直接按段索引与旧结果对齐，应该按时间重叠或重新跑完整链路。

因此，FireRedVAD 不能只用“VAD 本身更快”判断是否适合替换。真正的迁移判断需要继续测试：相同音频分别使用两套分段进入 Qwen-ASR，比较文本召回、时间戳、CAM++/emotion2vec 结果和模型池等待。

## 7. 当前结论和下一步

当前结论：FireRedVAD 在真实课堂音频上表现为“整体更少段、更长段、更少短段”，且不同课程差异明显。它具备作为多语言 VAD 候选的工程可行性，但仅凭本次分段对比不能确认质量优于现有 FSMN-VAD。

建议下一步按以下顺序验证：

1. 从综合岩石学、储能学术英语、科学的历程和多模态机器学习各抽取 3–5 个片段，人工标注真实语音起止。
2. 计算两套 VAD 的 precision、recall、F1、误报率、漏报率和边界误差。
3. 对低比例样本重点检查 FireRedVAD 是否漏掉短语音，对高比例样本检查是否把噪声或英文弱音误检为语音。
4. 选择代表音频接入 Qwen/CAM++/emotion2vec，比较完整响应，而不是只比较 VAD 段数。
5. 在完成上述验证前，保持现有 FSMN-VAD 为生产默认后端。

## 8. 复现命令

FireRedVAD 原始结果已经使用与当前模型目录相同 SHA256 的权重生成。若需要重新运行：

```bash
conda run -n asr python test/fireredvad_vad_compare.py \
  --backend fireredvad \
  --audio-root '/home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件' \
  --firered-model-dir '/home/zhangs/workspace/asr_nvidia/model/FireRedVAD/VAD' \
  --firered-use-gpu \
  --out test/artifacts/fireredvad/firered-gpu-results \
  --resume

conda run -n asr python test/compare_firereds_vad_to_paraformer.py
```

本报告未修改生产服务、默认配置或 Qwen-ASR 服务。
