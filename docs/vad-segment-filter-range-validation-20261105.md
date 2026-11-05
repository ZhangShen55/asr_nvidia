# VAD 段级过滤范围验证（2026-11-05）

## 验证目的

验证新段级 decibel、SNR、RMS 过滤在 `after_start=3`、`before_end=3`、`0/0` 和 `-1/-1` 三种范围配置下的候选段数量与删除数量，确认被删除段不会继续进入 Qwen-ASR。

## 数据与方法

- 样本来自 `test/audio_file/音频文件`，选择综合岩石学、苏州大学通用学术英语Ⅱ、高等传热学各 1 个课堂音频。
- 音频采样率均为 16 kHz；候选段来自每个音频现有的 `seaco_paraformer-*.json` 时间段，并按 `min_segment_duration_s=0.3` 过滤。
- 过滤使用当前实现 `core.vad_volume.filter_low_volume_segments()`，对每个候选段单独计算 RMS dBFS、段级 SNR 和低能量帧比例。
- 配置：`vad_segment_decibel_thres=-40`、`vad_decibel_frame_sample_ratio=0.5`、低能量比例阈值 `0.8`、段级 SNR `-18`、RMS `-40 dBFS`，三个 enable 均为 `true`。
- 原始明细保存于 `/tmp/vad_segment_filter_validation_20261105.json`；该文件为临时验证产物，不属于项目运行时依赖。

## 结果

| 音频 | 时长 | 候选段 | 默认前后 3 分钟：范围候选/删除 | 全音频：删除 | 关闭范围限制：删除 |
|---|---:|---:|---:|---:|---:|
| 综合岩石学 | 47.13 分钟 | 967 | 85 / 1（0.10%） | 3（0.31%） | 3（0.31%） |
| 通用学术英语Ⅱ | 47.02 分钟 | 242 | 55 / 1（0.41%） | 22（9.09%） | 22（9.09%） |
| 高等传热学 | 55.12 分钟 | 659 | 17 / 1（0.15%） | 1（0.15%） | 1（0.15%） |

删除原因统计中，命中的主要是 `decibel` 和 `rms_dbfs`；本次 `snr=-18` 没有单独删除段，说明该值对当前正 SNR 样本非常宽松。

## 结论

1. `after_start=3`、`before_end=3` 能把过滤计算限制在完整音频前后范围，三个样本只删除 1 个候选段，明显低于全音频模式；中间段保持原样。
2. `after_start=0`、`before_end=0` 与 `-1/-1` 都会检查整个音频，但语义不同：前者显式全音频生效，后者取消位置限制后仍按 enable 和阈值过滤。
3. 苏州大学样本全音频删除率为 9.09%，说明 RMS/decibel 阈值对不同录音增益较敏感，暂不建议直接把范围改为 `0/0` 作为生产默认。
4. `vad_snr_thres=-18` 适合作为宽松默认，不能承担主要的低能量过滤；后续若需要增强 SNR 过滤，应基于标定样本单独提高正阈值。

本次验证使用已有候选段时间轴评估段级过滤规则，未把这些结果当作 VAD 模型精度结论；上线前仍需在真实 FireRedVAD/FSMN-VAD 输出上观察有效低声语音保留率。

## 回归验证

- `python -m pytest -q --ignore=tests/test_docker_deployment.py`：262 passed，21 warnings。
- 完整 `python -m pytest -q`：266 passed，1 failed；唯一失败是既有 `tests/test_docker_deployment.py::test_build_script_and_docs_use_v3_time_tag_and_v3_route`，它检查 `docker/build.sh` 中旧的 `asr_offline:v3_` 标签。本 change 未修改该脚本，已将其作为范围外基线失败保留。
- `config.toml` 与 `docker/config.toml` 均通过 TOML 解析检查，`[vad].vad_device` 生效且 `[qwen_asr]` 不再包含该键。
