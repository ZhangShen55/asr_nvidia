# Proposal

## Why

`/v2/seacraft_asr` 当前会把部分非常小声的片段送入 Qwen-ASR，产生诸如“嗯”“啊”、短词或疑似外语的误识别，降低结果可用性并增加无效请求。现有响应还存在后处理时序问题：分段被清洗后，`language_ratio`、`speed_info` 和最终文本可能仍按被删除的分段计算。

本变更以 `test/audio_file/理论计算机科学导论_时间am9.55_191803.wav` 和对应人工标注结果作为 VAD 调参基线。该音频约 55 分钟，标注结果包含 46 个“细小声音”分段，标注段时间范围为 9.81~245.84 秒，标注段 RMS 中位数约为 -41.05 dBFS，未标注段 RMS 中位数约为 -13.98 dBFS。人工标注文件包含裸文本标记和少量说明文字，测试使用时应解析为标注数据，不修改原始文件。

## What Changes

- 在 `[qwen_asr]` 中增加 FSMN-VAD 的语音/噪声判定阈值配置，默认保持 FunASR 的 `0.6`，并说明低值更灵敏、高值更容易漏掉小声语音；配置值必须在 worker 初始化时真正传入 VAD。
- 把 `log_path` 从顶层迁移到 `[log].log_path`，同步本地配置、容器配置、配置读取、文档和测试；为已有部署保留明确的迁移策略。
- 将 `[model_device].emotion2vec` 与 `[model_device].speech_campplus` 从 GPU 整数索引改为 `cuda:N` 设备串并进行格式、CUDA 可用性和设备范围校验。VAD 继续使用独立的 `[qwen_asr].vad_device`，不恢复已移除的 `[model_device].vad`。
- 根据有效转写结果的语言占比确定最终语言，并在英文为主时将指定的中文口气词替换为英文口气词。替换集合、映射、英文占比判定和混合语言行为形成明确契约。
- 新增“单句单字语气词分段移除”：整个 `segment_text` 仅为指定的单字语气词（可带外围标点）时，删除整个分段及其关联字段，而不是只清空文本。
- 保证分段删除和文本替换后重新计算 `text`、`language_ratio`、`speed_info`，无有效分段时继续返回现有的 `4008`。
- 增加配置解析、VAD 参数传递、语言后处理、元数据一致性和参考音频回归测试出口；长音频效果测试作为显式集成测试，不强制进入普通单元测试流程。

## Capabilities

### New Capabilities

无。本变更扩展现有 Qwen v2 转写与服务配置能力。

### Modified Capabilities

- `qwen-asr-v2-adapter`: 增加 VAD 灵敏度配置、`cuda:N` 设备串契约、最终语言判定、英文口气词替换、单字语气词分段移除及后处理后元数据一致性要求。
- `service-operations`: 将日志路径配置归入 `[log]`，明确配置迁移和缺省行为。

## Impact

- **配置与读取**：`config.toml`、`config.docker.toml`、`core/config.py`；涉及现有自定义配置的迁移兼容。
- **VAD 运行时**：`core/vad_pool.py`、`core/vad.py`；新增阈值必须随每个 worker 的模型构造参数生效，且重启后生效。
- **模型设备**：`core/models.py`、`api/routes/asr_qwen_common.py`；模型加载和推理上下文由整数索引改为规范设备串。
- **转写后处理**：`utils/qwen_response.py`、`utils/itn.py` 或新增纯文本后处理模块、`api/routes/asr_qwen_common.py`；需要保持时间字段、角色、情绪和分段字段的整体一致性。
- **测试与文档**：现有配置/响应单元测试、OpenSpec 规格、README，以及参考音频和人工标注结果对应的可选集成测试脚本。
- **接口兼容**：`/v2/seacraft_asr` 保持字段结构和现有错误码；新增规则只会减少无效分段或改写展示文本，必须通过回归测试确认不会改变有效语音的时间轴。

