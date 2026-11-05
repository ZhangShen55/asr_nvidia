# Design

## Context

当前 `[qwen_asr]` 同时保存 Qwen 请求并发和 FSMN-VAD 参数。`core/vad_pool.py` 从 `Settings` 读取 VAD kwargs 并启动固定 worker；`core/vad.py` 在解析短段后调用 `core/vad_volume.py` 的段级 RMS/SNR 过滤。现有声音统计已经按 10 ms 帧向量化计算，但没有按低能量帧比例删除段，也没有完整音频前后范围控制。详见 proposal.md 的动机和范围；本设计只覆盖 FSMN-VAD 配置与段级过滤，不引入 FireRedVAD。

## Goals / Non-Goals

**Goals:**

- 让 `[vad]` 成为 VAD 配置的唯一来源，且 `vad_device` 数组继续决定固定 worker 数量。
- 保留 FunASR 全局 log 能量/SNR 参数的原有语义，并增加独立的段级 decibel、SNR、RMS 过滤开关。
- 用确定性抽帧和低能量帧比例过滤短促、低能量候选段，避免所有帧逐个执行 Python 逻辑。
- 让段级规则只作用于完整音频前后配置范围，保留中间范围段和每段原始时间轴。
- 记录每类过滤的候选数、删除数、耗时和阈值，便于后续按课堂音频标定。

**Non-Goals:**

- 不切换 VAD 后端，不实现 FireRedVAD、YAMNet 或环境音分类。
- 不跨 VAD 段合并音频，不修改 `segment_concurrency`、Qwen 请求协议或 CAM++/emotion2vec 模型池容量。
- 不把 dBFS 段级阈值转换成 FunASR log 能量阈值；两者保持独立单位和日志字段。
- 不保留 `[qwen_asr]` 中旧 VAD 字段的运行时回退；部署迁移由配置文件和启动校验完成。

## Decisions

### 1. 以 `[vad]` 为唯一配置边界

在 `Settings` 中新增/迁移 VAD 属性读取，所有 VAD 属性只查 `[vad]`。`vad_device` 规范化为非空的 `cpu`/`cuda:N` 数组；`cuda` 仍可规范化为 `cuda:0`。`[qwen_asr]` 仅保留 Qwen 调度参数和 `speaker_similarity_threshold` 等业务配置。旧位置出现时记录迁移提示，但不覆盖新位置，也不作为缺省回退。

最终默认配置在 `config.toml` 中使用以下结构，并为运维解释单位和生效范围：

```toml
[vad]
# 每个元素对应一个独立 FSMN-VAD worker；数组长度决定完整音频 VAD 并发上限
vad_device = ["cpu", "cpu", "cuda:0", "cuda:0"]
# 小于该时长的候选段直接丢弃，单位：秒
min_segment_duration_s = 0.3
# FunASR 单段上限与结束静音，单位：毫秒
vad_max_single_segment_time_ms = 12000
vad_max_end_silence_time_ms = 800
# 语音/噪声判定阈值，越高越不敏感
vad_speech_noise_thres = 0.8
# FunASR 全局帧级 log 能量阈值，不是 dBFS；-100 表示保持原行为
vad_decibel_thres = -40.0
# FunASR 全局帧级 SNR 阈值；-100 表示保持原行为
vad_snr_thres = -18.0
# VAD 后段级 decibel 过滤：低能量帧比例达到 0.8 即删除
vad_decibel_thres_filter_enabled = true
vad_segment_decibel_thres = -40.0
vad_decibel_frame_sample_ratio = 0.5
vad_decibel_low_frame_ratio_thres = 0.8
# VAD 后段级 SNR 过滤；-18.0 是当前确认的宽松默认值
vad_snr_thres_filter_enabled = true
# VAD 后段级 RMS dBFS 过滤
vad_low_volume_rms_dbfs_thres = -40.0
vad_low_volume_filter_enabled = true
# 段级过滤作用范围：前/后 N 分钟；0=整段，-1=该侧不限制
# 两侧均为 -1 时不按位置限制段级过滤
after_start = 3
before_end = 3
```

`vad_snr_thres` 同时继续传入 FunASR，并作为现有段级 SNR 阈值的默认来源；`vad_snr_thres_filter_enabled` 只控制 VAD 后段级 SNR 删除。这样保留现有调用口径，同时避免用 enable 开关意外改变 FunASR 的切分结果。

### 2. 在 VAD 结果进入下游前统一执行过滤

`detect_segments` 的阶段顺序固定为：解析 FunASR 段 → `min_segment_duration_s` 短段过滤 → 根据完整音频时长判断段是否处于 `after_start/before_end` 生效范围 → 对范围内段一次性计算统计并执行启用规则 → 返回保留段。范围外段直接保留，不重复计算；所有规则都按单个 VAD 段执行，不合并、不重切。

路由正常路径已经拥有解码后的音频张量和采样率，因此以该数据计算完整音频时长、段级 RMS/SNR/dBFS。对没有传入音频张量的离线测试/兼容调用，若启用了段级规则，则通过统一音频读取入口补齐 PCM 数据；读取失败应记录明确原因并返回配置/处理错误，不静默把未计算的段当作通过。

### 3. 复用一次段级统计，组合三个独立规则

扩展 `VolumeStats`/统计函数，使其在同一段内同时返回 RMS dBFS、估计 SNR、10 ms 帧数、抽样帧数和低能量帧比例。`vad_decibel_frame_sample_ratio` 只影响低能量帧比例；RMS/SNR 仍使用现有段级口径。统计使用 NumPy reshape/vectorization，避免每帧 Python 循环。

段级 decibel 规则为：从确定性等间距索引抽取比例为 `r` 的帧（`0 < r <= 1`），计算 `frame_dbfs < vad_segment_decibel_thres` 的比例；比例 `>= vad_decibel_low_frame_ratio_thres` 即命中 `low_decibel`。短于一个完整帧的段至少保留一个可计算帧，避免零分母。RMS 或 SNR 规则命中时分别记录 `rms_dbfs`/`snr` 原因；任意已启用规则命中则删除段。

各 enable 的职责如下：

- `vad_decibel_thres_filter_enabled`：只控制段级低能量帧比例规则。
- `vad_snr_thres_filter_enabled`：只控制段级 SNR 规则。
- `vad_low_volume_filter_enabled`：只控制段级 RMS dBFS 规则。

阈值 `<= -100.0` 视为该数值规则关闭，但 enable 仍保留并在配置/状态日志中显示实际值。这样 `enable=false` 和阈值关闭都具有可预测的短路行为，且可分别做标定。

### 4. 通过统计与日志验证过滤影响

`get_vad_stats()` 增加原始段、短段删除、范围内候选、decibel 删除、SNR 删除、RMS 删除、合计删除和过滤耗时字段。日志按段输出起止时间、命中的原因和关键统计，汇总日志输出有效段数量；不得记录整段音频内容。下游进度总数以过滤后的有效段数为分母，因此被删除的段不会进入 Qwen/CAM++/emotion2vec 调度。

### 5. 配置迁移与测试策略

配置单元测试覆盖 `[vad]` 默认值、类型/范围、`vad_device` 数组顺序、旧 `[qwen_asr]` 字段不回退、`after_start/before_end` 特殊值和三个 enable 的独立性。声音统计测试覆盖 0.5 抽帧、全帧、短段、低能量比例边界、RMS/SNR 任一命中和全部规则关闭。链路测试验证被删除段不创建 Qwen 请求，范围外段保持时间戳和数量。

### 6. 考虑过的替代方案

- **只使用 `-100` 关闭规则**：无法表达“配置存在但本次不启用”的意图，也容易把 FunASR 全局阈值和段级阈值混淆，因此保留显式 enable。
- **每个段跨多个窗口聚合后再判断**：会改变 VAD 段边界并可能把少量有效人声与多数杂音混为一段；本 change 坚持逐段独立判断。
- **全帧 Python 循环**：统计口径直观但在长课堂音频和多路并发下增加 VAD 后处理耗时；采用向量化和可控抽帧比例。

## Risks / Trade-offs

- [低能量规则可能删除有效低声语音] → 默认只作用于前后 3 分钟，保留显式开关和阈值，先通过前后五分钟 RMS/SNR 报告及人工听查标定。
- [抽帧可能漏掉短促边界噪声] → 使用确定性等间距抽样、记录抽样帧数，并允许把比例提高到 `1.0` 做全帧复核。
- [`vad_snr_thres=-18.0` 对正值 SNR 样本很宽松] → 明确记录为当前确认的默认值，报告实际删除率；后续只根据标定数据改为更高正阈值。
- [旧配置直接失效会导致部署遗漏] → 启动时对旧键打印迁移提示，并在 README/默认配置中提供完整 `[vad]` 示例；不在运行时静默回退，避免配置来源不确定。
- [段级统计增加 CPU 时间] → 同一段一次性计算全部特征，使用 NumPy 向量化和抽帧比例；日志记录耗时以便观察。

## Migration Plan

1. 更新 `config.toml`，把 VAD 键移动到 `[vad]`，保留 `[qwen_asr]` 的 Qwen 调度键和 `speaker_similarity_threshold`。
2. 启动服务执行配置校验，确认 `[vad].vad_device` worker 数量、阈值范围和过滤范围；旧位置仅产生迁移提示。
3. 先使用默认前后 3 分钟范围运行课堂音频，比较原始段、过滤段、Qwen 请求数和有效文本保留率。
4. 依据报告调整 `vad_segment_decibel_thres`、`vad_decibel_low_frame_ratio_thres`、RMS/SNR 阈值或把范围改为 `0/0` 全音频验证。
5. 出现有效语音误删或请求异常时，将三个段级 enable 设为 `false`（或阈值设为 `-100.0`），重启即可回到仅 FunASR 全局切分的行为；若配置迁移失败，恢复旧版本配置与代码回滚。

## Open Questions

无。`vad_snr_thres=-18.0` 按用户最后确认的默认值纳入本 change；是否在后续标定中提高为正阈值属于运行参数调优，不改变本 change 的配置结构。
