# 三套 VAD 分段基线对比报告

日期：2026-10-28
样本目录：`/home/zhangs/workspace/asr_nvidia/test/audio_file/音频文件/`
样本数量：45 个 WAV，16 kHz、单声道、16 bit，总时长 136,419.79 秒（约 37.89 小时）

## 1. 三套基线的定义

本次把三类结果完全分开统计：

1. **JSON 基线**：每个 `seaco_paraformer-*.json` 的 `segments[].bg/ed`。
2. **FunASR 基线**：使用当前 FunASR FSMN-VAD 直接对同名 WAV 运行得到的结果。
3. **FireRedVAD 基线**：使用 `/home/zhangs/workspace/asr_nvidia/model/FireRedVAD/VAD`，CUDA 默认参数运行得到的结果。

三套基线全部覆盖相同的 45 个 WAV，没有缺失文件。逐文件原始结果和指标见：

- [comparison.json](/home/zhangs/workspace/asr_nvidia/test/artifacts/fireredvad/three-baseline-comparison/comparison.json)
- [compare_three_vad_baselines.py](/home/zhangs/workspace/asr_nvidia/test/compare_three_vad_baselines.py)

## 2. 总体分段粒度

| 指标 | JSON 基线 | FunASR 直接基线 | FireRedVAD |
|---|---:|---:|---:|
| 分段总数 | 40,490 | 23,904 | 31,075 |
| 每分钟分段数 | 17.81 | 10.51 | 13.67 |
| 语音覆盖率 | 79.39% | 76.76% | 75.61% |
| 平均段长 | 2.675 秒 | 4.381 秒 | 3.319 秒 |
| 段长 P50 | 1.98 秒 | 3.85 秒 | 2.34 秒 |
| 段长 P95 | 6.92 秒 | 8.01 秒 | 9.71 秒 |
| 小于 1 秒段 | 9,449（23.34%） | 3,019（12.63%） | 4,816（15.50%） |
| 不小于 10 秒段 | 737（1.82%） | 0 | 1,481（4.77%） |

从粒度看：

- **JSON 基线最细**：段数最多、每分钟段数最高、平均段长最短。
- **FunASR 直接基线最粗**：段数最少，平均段长和 P50 最大。
- **FireRedVAD 处于两者之间**：比 FunASR 直接基线细，但比 JSON 基线粗；FireRedVAD 的 P95 和长段比例最高，说明长段尾部更明显。

如果按“每个分段都要发送一次 Qwen/CAM++/emotion2vec”估算下游任务量，三套基线的理论任务量排序是：

```text
JSON 40,490  >  FireRedVAD 31,075  >  FunASR 23,904
```

## 3. 三套基线之间的数量差异

| 对比 | 分段数量变化 | 数量比例 |
|---|---:|---:|
| FunASR 相对 JSON | -16,586 | 59.04% |
| FireRedVAD 相对 JSON | -9,415 | 76.75% |
| FireRedVAD 相对 FunASR | +7,171 | 130.00% |

这说明“FireRedVAD 比 JSON 少段”和“FireRedVAD 比直接 FunASR 多段”可以同时成立，原因是 JSON 基线本身比直接 FunASR 产生了更多分段。

## 4. 边界与拆分/合并行为

下表以左侧基线为 A、右侧模型为 B，按时间重叠统计。它用于观察分段形态，不是人工准确率。

| 对比 | A 段被 B 拆成至少 2 段 | B 段覆盖至少 2 个 A 段 | A 无 B 重叠 | B 无 A 重叠 | 起点差均值 | 终点差均值 |
|---|---:|---:|---:|---:|---:|---:|
| JSON → FunASR | 11,304 | 13,664 | 41 | 344 | 2.309 秒 | 2.475 秒 |
| JSON → FireRedVAD | 11,434 | 13,582 | 268 | 849 | 2.267 秒 | 2.604 秒 |
| FunASR → FireRedVAD | 9,520 | 6,553 | 933 | 1,493 | 1.610 秒 | 1.408 秒 |

JSON 与另外两套基线的最佳重叠 IoU 均较低，说明 JSON 分段边界与实际直接 VAD 输出并不是简单的一一对应关系。FunASR 直接结果与 FireRedVAD 的最佳重叠均值相对更高，但仍存在明显拆分和合并。

## 5. 为什么 JSON 与直接 FunASR 数量不同

你标注的“JSON 855 段、直接 FunASR 706 段”**不能直接归因于当前新增的低分贝过滤**。

当前配置为：

```toml
vad_low_volume_filter_enabled = false
```

而且本次直接 FunASR 对比没有调用 `filter_low_volume_segments()` 这层自定义段级低音量过滤。因此，这个差异更可能来自以下因素：

1. JSON 生成时使用了不同的 FunASR VAD 参数，例如结束静音、最大段长、语音噪声阈值。
2. JSON 的 `segments` 可能是“VAD 后经过 ASR 链路整理后的输出段”，不一定是原始 FSMN-VAD 时间戳。
3. JSON 与当前运行使用的模型版本、FunASR 版本或后处理逻辑不同。
4. JSON 生成链路可能做过额外的合并、拆分、短段过滤或时间轴修正。

因此，本报告把 JSON 独立作为历史输出基线，把当前 FunASR 直接结果作为运行时基线，避免混淆。

## 6. 从部署角度如何解读

### JSON 基线

JSON 分段最细，理论上会产生最多的 Qwen、CAM++ 和 emotion2vec 请求。优点是段粒度细，缺点是请求数量和模型池排队压力最大。

### FunASR 直接基线

FunASR 直接结果段数最少、段最长。优点是下游调用量最低，缺点是长段可能增加单段 Qwen/CAM++/emotion2vec 推理时间，也可能把短停顿或说话人切换留在同一段。

### FireRedVAD

FireRedVAD 位于两者之间，分段数量比 FunASR 直接结果多约 30%，但比 JSON 少约 23%。它的长段尾部比另外两套更明显，需要重点检查长段是否跨越多个自然语句或说话人切换。

## 7. 当前结论

仅从分段粒度看：

```text
细粒度：JSON 基线
中等粒度：FireRedVAD
粗粒度：FunASR 直接基线
```

但这不是质量排名。当前还没有人工标注语音区间，因此不能判断哪套 VAD 漏检更多、误检更多，也不能直接决定 FireRedVAD 是否替换 FunASR。

下一步应从三类差异最大的音频中抽样人工检查：

- JSON 段数远高于 FireRedVAD 的综合岩石学样本；
- FireRedVAD 段数高于 JSON 的储能学术英语样本；
- FunASR 与 FireRedVAD 长段比例差异明显的课堂样本。

对这些样本标注真实语音区间后，再计算 precision、recall、F1、误报率、漏报率和边界误差。完成后才能判断 FireRedVAD 是“合并得更合理”，还是“漏掉了更多短语音”。

本报告未修改生产 VAD 配置、服务代码或 Qwen-ASR 服务。
