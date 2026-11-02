# Proposal

## Why

需要验证 `vad_low_volume_rms_dbfs_thres=-41.0` 在真实课堂音频上的实际效果，而不是只依据单个音频的 RMS 中位数判断。目标是对 `test/audio_file/音频文件/` 中的 45 个 WAV 逐一运行 Qwen-ASR，使用同目录的 `seaco_paraformer-*.json` 作为时间和文本对照，确认低音量过滤是否主要去除了无关细小声音，同时确认现有复读后处理是否仍有幻觉重复漏网，并记录 16 路并发下的 P50/P95。

## What Changes

- 建立一套可复核的全量质量验证流程，固定使用 `-41.0 dBFS`、`vad_low_volume_filter_enabled=true`、`segment_concurrency=4`、`showSpk=false`、`showEmotion=false`，服务侧音频并发设置为 16。
- 递归发现 45 个 WAV，并将每个 WAV 与同目录的 `seaco_paraformer-<WAV 文件名>.json` 配对；实际文件前缀为 `seaco_paraformer-`（下划线），测试工具必须明确记录该映射规则并对缺失或多余文件失败。
- 保存每个请求的原始 v2 响应、HTTP 状态、错误、耗时、VAD 段数、Qwen 响应段数和过滤统计，计算全量成功请求的端到端 P50/P95，并报告失败请求与超时。
- 按时间重叠对齐 Qwen 响应段和 Paraformer 基线段，统计基线独有段、Qwen 独有段、重叠段、过滤前后段数变化，以及对应文本差异。
- 对基线独有段生成“低音量无关内容候选”：记录时间、基线 `segment_text`、RMS/时长、相邻上下文和 Qwen 是否在该区间产生文本；候选必须经过规则分类和人工复核，不能仅以段数减少判定为过滤成功。
- 对当前 Qwen 响应执行段内复读、字符级复读、相邻段重复、跨段重复 shingle、异常字符速率和超长文本扫描，并把结果与服务日志中的“复读修复”和“复读检测后保留”事件对账，区分已修复、漏网、误报和无法判定。
- 输出一份包含逐文件明细、总体统计、候选清单、复读漏网清单、性能分位数和后续优化建议的报告。若发现现有规则未覆盖的复读模式，只形成后续后处理优化 change 的具体方案，本 change 不在测试结论产生前直接修改生产算法。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `qwen-asr-v2-adapter`: 增加低音量过滤与历史 Paraformer 基线的可复核质量验证要求，并增加复读后处理的漏网检测、日志对账和后续优化判定要求。

## Impact

- **测试工具**：扩展现有 Qwen v2 并发压测和质量分析工具，支持递归 WAV、`seaco_paraformer-*.json` 配对、逐段时间对齐、低音量候选分类和复读漏网扫描。
- **运行配置**：只生成临时测试配置，不修改默认配置；测试配置覆盖 `vad_low_volume_rms_dbfs_thres`、`vad_low_volume_filter_enabled`、`segment_concurrency` 和 `audio_concurrency`。测试使用 8084，外部 Qwen-ASR 8806 保持运行。
- **报告产物**：在 `docs/` 保存汇总报告，在测试输出目录保存逐文件响应、对齐结果、候选证据、日志快照和机器可读 JSON；不修改原始 WAV 或 Paraformer JSON。
- **服务行为**：本 change 以验证为主，不改变 `/v2/seacraft_asr` 响应契约。只有后续确认存在稳定复读漏网且需要扩大后处理规则时，才另行创建实现 change。
- **准确性边界**：Paraformer JSON 不是人工转写真值，因此报告将“文本准确率”表述为相对基线的一致性和人工复核分类，并明确不能由该数据单独推导绝对 WER/准确率。
