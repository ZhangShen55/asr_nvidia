# Proposal

## Why

当前 VAD 配置与 Qwen-ASR 并发配置混在 `[qwen_asr]` 中，既不利于独立调节 VAD worker，也无法表达“只在完整音频前后指定时间范围内过滤低能量段”的业务需求。现有段级过滤只覆盖 RMS/SNR，无法按 FunASR log 能量帧比例识别课堂录音中的短促、低能量片段，导致无关片段仍可能进入 Qwen-ASR。

## What Changes

- 新增独立 `[vad]` 配置区，集中承载 VAD worker 设备、切分参数、FunASR 全局 VAD 参数和段级过滤参数。
- 将 `vad_device` 从 `[qwen_asr]` 移到 `[vad]`；数组元素继续一一对应固定 FSMN-VAD worker，数组长度决定 VAD 并发容量。
- 将 `min_segment_duration_s`、`vad_max_single_segment_time_ms`、`vad_max_end_silence_time_ms`、`vad_speech_noise_thres`、`vad_decibel_thres`、`vad_snr_thres`、低音量参数迁移到 `[vad]`；`speaker_similarity_threshold` 继续保留在 `[qwen_asr]`。
- **BREAKING**：不再从 `[qwen_asr]` 读取 VAD 配置；旧位置不作为兼容回退来源，启动配置校验和运行时读取统一使用 `[vad]`。
- 保留并实现三个独立段级过滤开关：`vad_decibel_thres_filter_enabled`、`vad_snr_thres_filter_enabled`、`vad_low_volume_filter_enabled`。
- 新增段级 decibel 过滤：按 `vad_decibel_frame_sample_ratio` 抽取段内帧，计算低于 `vad_segment_decibel_thres` 的帧比例，达到 `vad_decibel_low_frame_ratio_thres` 时删除该段。
- 新增 `after_start` 与 `before_end`，控制段级 decibel、SNR、RMS 过滤只作用于完整音频的前后时间范围；不跨 VAD 段合并，不改变 VAD 段时间轴。
- 保留 FunASR 的全局 `vad_decibel_thres` 与 `vad_snr_thres` 语义；段级开关只控制 VAD 完成后的对应段级过滤，二者职责在日志和文档中明确区分。
- 为配置解析、段级过滤、抽帧比例、时间范围边界和过滤统计补充测试与日志，确保被删除的段不会向 Qwen-ASR 发起请求。

## Capabilities

### New Capabilities

无。本 change 扩展既有 VAD 配置和过滤行为，不单独引入新的对外能力边界。

### Modified Capabilities

- `qwen-asr-v2-adapter`：VAD 配置迁移到 `[vad]`，增加段级 decibel/低音量过滤、抽帧比例和前后时间范围语义，并保持每个保留 VAD 段对应一个 Qwen 请求。

## Impact

- 主要影响 `core/config.py`、`core/vad_pool.py`、`core/vad.py`、`core/vad_volume.py`、`config.toml`、相关单元测试和配置/部署文档。
- 过滤发生在 Qwen 请求调度之前，可能减少 Qwen 请求数、段级本地模型调用数和端到端耗时，同时可能删除有效的低声语音；需要通过现有课堂音频回归和前后范围统计验证阈值。
- `vad_decibel_thres=-40.0` 仍是 FunASR 的全局 log 能量阈值（不是 dBFS）；段级 decibel 使用独立的 `vad_segment_decibel_thres=-40.0`，避免把两个单位不同的阈值混用。
- `vad_snr_thres=-18.0` 按当前确认的默认值写入；由于项目段级 SNR 通常为正值，该默认值属于宽松过滤，是否提高到正数应由后续标定数据决定。
