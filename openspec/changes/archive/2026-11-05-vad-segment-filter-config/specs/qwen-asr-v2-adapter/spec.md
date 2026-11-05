# Spec Delta

## MODIFIED Requirements

### Requirement: 通过配置外置运行参数
Qwen 适配链路的服务地址、模型名、请求并发度、总音频并发度、增强音频并发度、队列容量、超时和 VAD 参数 MUST 来自配置文件；VAD 参数 MUST 只从 `[vad]` 读取，VAD 实例数 MUST 只由 `[vad].vad_device` 数组长度决定，动态 VAD 池参数 MUST 不再作为有效配置。

#### Scenario: 修改并发参数后重启生效
- **WHEN** 运维把 `[vad].vad_device` 改为 8 个元素、`[qwen_asr].segment_concurrency` 改为 16、`[qwen_asr].audio_concurrency` 改为 8、`[qwen_asr].audio_spk_emo_concurrency` 改为 4 并重启服务
- **THEN** 服务启动 8 个固定 VAD worker，并按新的 Qwen、总音频和增强音频并发参数运行

#### Scenario: VAD 旧配置不作为回退
- **WHEN** 配置只在 `[qwen_asr]` 中提供 `vad_device` 或其它 VAD 参数，而 `[vad]` 未提供对应配置
- **THEN** 系统不读取旧位置的 VAD 值，使用 `[vad]` 的默认值或返回明确的配置校验错误，并记录迁移提示

#### Scenario: 配置缺省时使用默认值
- **WHEN** 配置文件缺少相关新配置项
- **THEN** 系统使用 `[vad].vad_device=["cpu", "cpu", "cpu", "cpu", "cuda:0", "cuda:0", "cuda:0", "cuda:0"]`、`segment_concurrency=16`、`request_concurrency=128`、`audio_concurrency=8`、`audio_spk_emo_concurrency=4`，默认满额立即拒绝

### Requirement: VAD worker 数量由设备数组固定决定
系统 MUST 在启动时按 `[vad].vad_device` 数组的元素顺序预热固定数量的 FSMN-VAD worker；数组长度 MUST 等于 VAD 的有效并发数。worker MUST 按 FIFO 接收完整音频，单个 worker 同时只能处理一个完整音频，运行期间 MUST NOT 动态扩容或回收。

#### Scenario: CPU 与 GPU 混合 VAD 配置
- **WHEN** `[vad].vad_device = ["cpu", "cpu", "cuda:0", "cuda:0"]`
- **THEN** 服务启动 4 个 VAD worker，其中 2 个使用 CPU、2 个使用 `cuda:0`，最多同时对 4 个完整音频执行 VAD

#### Scenario: 超过 VAD worker 数量的请求
- **WHEN** 同时提交的完整音频数超过 `[vad].vad_device` 数组长度
- **THEN** 超出的音频按既有准入和 FIFO 排队规则等待或返回容量错误，不创建临时 VAD worker

#### Scenario: VAD 数组元素无效
- **WHEN** `[vad].vad_device` 包含空字符串、未知设备类型或格式不是 `cpu`/`cuda:N` 的元素
- **THEN** 服务启动配置校验失败并指出具体数组下标，不以隐式 CPU 回退

### Requirement: VAD 配置删除动态池参数
系统 MUST 以 `[vad].vad_device` 数组作为 VAD 实例配置来源，并 MUST 不再读取或要求 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`、`vad_call_timeout_s`；VAD 排队由音频并发准入和固定 worker FIFO 调度承担。

#### Scenario: 旧动态池字段残留
- **WHEN** 配置文件仍包含已删除的动态池字段，同时提供合法的 `[vad].vad_device` 数组
- **THEN** 服务不使用这些字段改变 worker 数量或扩缩容行为，并记录迁移提示

#### Scenario: 固定池启动失败
- **WHEN** `[vad].vad_device` 为空或没有任何 worker 成功预热
- **THEN** 服务启动失败并返回明确配置错误，不能以没有 VAD worker 的状态接收转写请求

## ADDED Requirements

### Requirement: 段级低能量过滤在 Qwen 请求前生效
系统 MUST 在短段过滤之后、Qwen 请求调度之前，对每个候选 VAD 段独立执行启用的段级 decibel、SNR 和 RMS 规则；任一已启用规则判定丢弃时，该段 MUST 不发起 Qwen、CAM++ 或 emotion2vec 请求。

#### Scenario: 段级 decibel 低能量比例达到阈值
- **WHEN** 段级 decibel 过滤已启用，抽取帧中低于 `vad_segment_decibel_thres` 的比例大于或等于 `vad_decibel_low_frame_ratio_thres`
- **THEN** 系统删除该 VAD 段并记录段起止时间、低能量帧比例和丢弃原因

#### Scenario: 段级 decibel 比例未达到阈值
- **WHEN** 段级 decibel 过滤已启用，但抽取帧中低于阈值的比例小于 `vad_decibel_low_frame_ratio_thres`
- **THEN** 系统保留该段并继续进入既有 Qwen 请求调度

#### Scenario: 关闭单个过滤规则
- **WHEN** `vad_decibel_thres_filter_enabled`、`vad_snr_thres_filter_enabled` 或 `vad_low_volume_filter_enabled` 中任一项为 `false`
- **THEN** 系统不执行该项对应的段级删除，但其它已启用规则继续独立生效

#### Scenario: 过滤完成后无有效段
- **WHEN** 一个完整音频的候选 VAD 段全部被段级规则删除
- **THEN** 系统不向 Qwen-ASR 发请求，并按现有“未检测到有效人声”响应和统计口径返回

### Requirement: 段级 decibel 抽帧比例可控且统计口径稳定
系统 MUST 使用 `vad_decibel_frame_sample_ratio` 控制段内 decibel 帧的抽样比例，取值范围为 `(0, 1]`；比例为 `0.5` 时约抽取每秒 100 个 10 ms 帧位置中的一半，低能量比例分母只包含实际抽取帧。

#### Scenario: 使用半数帧计算
- **WHEN** `vad_decibel_frame_sample_ratio=0.5` 且候选段包含 100 个 10 ms 帧位置
- **THEN** 系统按确定性规则抽取约 50 帧，只用这 50 帧计算低能量比例，不把未抽取帧计入分母

#### Scenario: 抽帧比例越界
- **WHEN** 配置为 `0`、负数或大于 `1` 的值
- **THEN** 服务配置校验失败并指出 `vad_decibel_frame_sample_ratio`，不得以空样本或全帧隐式替代

#### Scenario: 短段少于一个完整帧
- **WHEN** 候选段长度不足一个 10 ms 分析帧
- **THEN** 系统使用至少一个可计算帧或按既有短段门限删除，不因抽帧分母为零而报错

### Requirement: 段级过滤时间范围按完整音频边界控制
系统 MUST 使用 `[vad].after_start` 和 `[vad].before_end`（单位：分钟）决定段级 decibel、SNR、RMS 规则是否作用于某个 VAD 段；规则按每个 VAD 段独立判断，不跨段合并或重切时间轴。

#### Scenario: 默认只检查前后范围
- **WHEN** `after_start=3`、`before_end=3`
- **THEN** 仅完整音频开始 3 分钟内或结束前 3 分钟内的候选段执行段级过滤，中间范围的段直接保留

#### Scenario: 全音频检查
- **WHEN** `after_start=0` 且 `before_end=0`
- **THEN** 完整音频中的所有候选段都执行已启用的段级过滤

#### Scenario: 关闭范围限制
- **WHEN** `after_start=-1` 且 `before_end=-1`
- **THEN** 段级过滤不因音频位置被限制，按各 enable 开关和阈值处理全部候选段

#### Scenario: 仅启用一侧范围
- **WHEN** `after_start=-1` 且 `before_end=3`，或 `after_start=3` 且 `before_end=-1`
- **THEN** 只对配置为非负数的一侧执行位置过滤，另一侧不设范围限制

#### Scenario: 范围参数非法
- **WHEN** `after_start` 或 `before_end` 小于 `-1`，或不是有限数值
- **THEN** 服务配置校验失败并指出具体参数，不静默转换为 `0`

### Requirement: 全局 FunASR 参数与段级参数职责分离
系统 MUST 将 `vad_decibel_thres` 和 `vad_snr_thres` 继续作为传给 FunASR 的全局帧级 VAD 参数，同时使用独立的 `vad_segment_decibel_thres`、`vad_decibel_low_frame_ratio_thres`、段级 SNR 阈值和 RMS dBFS 阈值执行 VAD 后过滤；全局阈值单位不得按 dBFS 解释。

#### Scenario: 全局阈值保持原有语义
- **WHEN** `vad_decibel_thres=-100.0` 或 `vad_snr_thres=-100.0`
- **THEN** 对应 FunASR 全局限制保持原行为；这两个值不自动关闭或替代段级过滤开关

#### Scenario: 段级 decibel 使用独立门限
- **WHEN** `vad_segment_decibel_thres=-40.0` 且段级 decibel 开关为 `true`
- **THEN** 系统用段内帧的 dBFS 统计与 `-40.0` 比较，不把该值作为 FunASR 的 log 能量参数传入

#### Scenario: 段级 SNR 与 RMS 使用显式开关
- **WHEN** `vad_snr_thres=-18.0` 或 `vad_low_volume_rms_dbfs_thres=-40.0` 且对应 enable 为 `true`
- **THEN** 系统按现有段级 `snr_db`/`rms_dbfs` 口径执行过滤；阈值等于或低于 `-100.0` 时对应规则不丢弃段
