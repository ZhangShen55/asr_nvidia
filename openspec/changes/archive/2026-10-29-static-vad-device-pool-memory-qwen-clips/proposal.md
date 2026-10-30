# Proposal

## Why

当前 v2 链路的每个 VAD 段都会先写入临时 WAV，再由 Qwen-ASR 重新打开上传；在长音频和多路并发下，磁盘 I/O、文件创建和删除会放大裁剪阶段耗时。VAD 池还使用动态扩缩容参数，实际有效并发与配置中的设备数量不够直观，容易让上游误判完整音频的并发能力。

## What Changes

- 将 Qwen-ASR 段请求改为直接上传内存中的标准化 WAV 字节；每个 VAD 段仍对应一个独立 Qwen request，不合并相邻段，不改变 VAD 边界、采样率、声道或 PCM 内容。
- **严格清理缓存**：请求源文件、兼容路径下的段缓存、生成的临时 WAV 及其它服务自有缓存，必须在处理成功、异常、取消和超时后清除；服务重启时只清理可识别的服务自有残留文件，不能误删其它临时文件。
- 控制台隐藏 `httpx`/`httpcore` 成功请求的 `INFO` 访问日志，保留警告和错误，文件日志仍可保留完整 HTTP 记录；在交互式 TTY 中为每个音频显示有效 VAD 段的 `已完成/总段数` tqdm 进度，单段的 Qwen、CAM++、emotion2vec 全部结束（成功或失败）后更新一次并累计失败数，多音频各自独立显示，非 TTY 自动关闭进度控制字符。
- **BREAKING** 将 VAD 实例配置改为 `qwen_asr.vad_device` 设备数组；数组每个元素对应一个启动时固定预热的 FSMN-VAD worker，数组长度就是有效 VAD 并发数，支持 CPU 与多个 CUDA 设备混合部署。
- **BREAKING** 删除 VAD 动态扩缩容和超时配置：`vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`、`vad_call_timeout_s`。固定 worker 按 FIFO 接收完整音频，一个 worker 同时只处理一个完整音频。
- **BREAKING** 删除 `[model_device]` 下的 `emotion2vec` 与 `speech_campplus` 单设备字段，将 `[model_device.local_model_pool]` 重命名为 `[model_device_pool]`；CAM++ 与 emotion2vec 的实例数量和落卡设备只由各自设备数组有效长度与元素决定。
- 保持 `speech_campplus_devices` 与 `emotion2vec_devices` 的非对称配置能力，例如 3:1；空数组表示对应模型不拉起，相关请求返回明确的模型不可用错误。
- 更新默认配置：`segment_concurrency=16`、`audio_concurrency=8`，VAD 默认使用 4 个 CPU worker 和 4 个 `cuda:0` worker；模型池默认示例为 CAM++ 3 个 `cuda:0` 实例、emotion2vec 1 个 `cuda:0` 实例，功能开关关闭时不加载模型池。
- 保留一个 VAD 段对应一个 Qwen request，以及同一 VAD 段内 Qwen、CAM++、emotion2vec 的并行关系。
- 将裁剪累计耗时与裁剪阶段墙钟耗时分开记录；低音量/能量统计优化作为后续独立变更，本次不改变其过滤规则或阈值语义。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `qwen-asr-v2-adapter`: 段音频改用内存字节上传；VAD worker 从动态池改为设备数组固定池；保留逐 VAD 段 Qwen 请求和并行增强能力。
- `local-model-runtime`: 本地模型设备数组和模型池配置统一迁移到 `[model_device_pool]`，模型实例数量由数组长度决定。
- `service-operations`: 更新有效配置项清单、VAD 固定实例启动语义和配置迁移规则。

## Impact

- 影响 `core/config.py`、`core/vad_pool.py`、`core/models.py`、`api/routes/asr_qwen_common.py`、`core/qwen_asr_client.py`、`utils/audio_utils.py` 和默认配置文件。
- 影响 `/get_status` 中 VAD 与本地模型池的设备、实例、busy/waiting 状态字段；需要明确区分配置实例数与成功加载实例数。
- 影响请求资源生命周期和故障恢复：必须登记每个请求创建的缓存路径，在所有退出路径执行幂等清理，并验证并发同名上传和进程异常后的残留处理。
- 影响控制台可观测性：正常 Qwen HTTP 成功行不再刷屏，进度条以完整 VAD 段处理完成为粒度更新，并兼容多音频并发和日志重定向。
- 影响旧配置迁移：旧 `[model_device]` 和 `[model_device.local_model_pool]` 不再作为正式配置结构；启动时应给出迁移提示或明确配置错误，不应悄悄使用旧单设备字段覆盖新数组。
- 不改变 `/v2/seacraft_asr` 响应结构、Qwen 服务协议、VAD 段时间轴、CAM++ 聚类算法、emotion2vec 标签映射和低音量过滤语义。
- 需要用 CPU/GPU 混合 VAD 配置、空模型数组、非对称 CAM++/emotion2vec 数组以及 1/4/8 路完整音频压测验证性能、错误码、显存和结果一致性。
