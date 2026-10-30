# Spec Delta

## ADDED Requirements

### Requirement: VAD worker 数量由设备数组固定决定
系统 MUST 在启动时按 `qwen_asr.vad_device` 数组的元素顺序预热固定数量的 FSMN-VAD worker；数组长度 MUST 等于 VAD 的有效并发数。worker MUST 按 FIFO 接收完整音频，单个 worker 同时只能处理一个完整音频，运行期间 MUST NOT 动态扩容或回收。

#### Scenario: CPU 与 GPU 混合 VAD 配置
- **WHEN** `vad_device = ["cpu", "cpu", "cuda:0", "cuda:0"]`
- **THEN** 服务启动 4 个 VAD worker，其中 2 个使用 CPU、2 个使用 `cuda:0`，最多同时对 4 个完整音频执行 VAD

#### Scenario: 超过 VAD worker 数量的请求
- **WHEN** 同时提交的完整音频数超过 `vad_device` 数组长度
- **THEN** 超出的完整音频进入固定 worker 的 FIFO 等待队列，已运行的 worker 不接收第二个完整音频

#### Scenario: VAD 数组元素无效
- **WHEN** 设备数组包含无法使用的设备
- **THEN** 该 worker 启动失败并记录原因，状态接口区分配置实例数、成功实例数和失败实例数；服务仍可使用其余成功 worker

### Requirement: VAD 配置删除动态池参数
系统 MUST 以 `vad_device` 数组作为 VAD 实例配置来源，并 MUST 不再读取或要求 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`、`vad_call_timeout_s`；VAD 排队由音频并发准入和固定 worker FIFO 调度承担。

#### Scenario: 旧动态池字段残留
- **WHEN** 配置文件仍包含已删除的动态池字段，同时提供合法的 `vad_device` 数组
- **THEN** 服务不使用这些字段改变 worker 数量或扩缩容行为，并记录迁移提示

#### Scenario: 固定池启动失败
- **WHEN** `vad_device` 为空或没有任何 worker 成功预热
- **THEN** 服务启动失败并返回明确配置错误，不能以没有 VAD worker 的状态接收 v2 请求

### Requirement: 每个 VAD 段通过内存音频请求 Qwen
系统 MUST 保持一个 VAD 段对应一个独立 Qwen request，但 MUST 直接上传该段标准化后的内存 WAV 字节；段请求 MUST NOT 依赖为 Qwen 反复创建、打开和删除临时 WAV 文件。上传字节必须保持 VAD 起止样本、16 kHz、单声道和 PCM 内容不变。

#### Scenario: 正常段请求
- **WHEN** VAD 产出一个有效段
- **THEN** Qwen 收到一个独立的 16 kHz 单声道 PCM WAV 请求，响应分段时间轴仍使用该 VAD 段的 `bg` 和 `ed`

#### Scenario: 多段音频
- **WHEN** 一个完整音频包含多个 VAD 段
- **THEN** 每个段分别向 Qwen 发起请求，不合并相邻段，段级并发仍受 `segment_concurrency` 限制

#### Scenario: Qwen 请求失败
- **WHEN** 内存字节请求超时、不可达或返回错误
- **THEN** 按既有重试和错误码契约处理，不能因为取消临时 clip 清理而泄漏文件或改变其它段的结果

### Requirement: 严格清理服务自有缓存文件
系统 MUST 登记每个请求创建的源音频临时文件、兼容路径下的段缓存和其它服务自有缓存，并 MUST 在成功、VAD 异常、裁剪异常、Qwen/本地模型失败、请求取消和超时的所有退出路径执行幂等清理；清理范围 MUST 仅限服务可识别的自有路径。

#### Scenario: 请求成功完成
- **WHEN** v2 请求成功返回或返回无有效人声
- **THEN** 该请求创建的源文件、段缓存和临时 WAV 全部删除，服务自有缓存目录不留下该请求文件

#### Scenario: VAD 或裁剪异常
- **WHEN** VAD worker 失败、排队取消、音频裁剪失败或标准化失败
- **THEN** 已创建的全部请求缓存仍由统一清理路径删除，并保留原有错误码和日志

#### Scenario: Qwen 或本地模型失败
- **WHEN** 任一段的 Qwen 请求失败/超时，或 CAM++、emotion2vec 推理失败导致请求退出
- **THEN** 已完成和未完成段的缓存都被清理，后续请求不受残留文件影响

#### Scenario: 客户端取消或服务超时
- **WHEN** 客户端断开连接、任务被取消或请求达到超时
- **THEN** 取消处理、HTTP 连接关闭和后台协程退出都会触发幂等清理，不留下正在上传或正在写入的缓存文件

#### Scenario: 并发同名上传
- **WHEN** 多个请求使用相同原始文件名同时提交
- **THEN** 每个请求使用隔离的服务自有路径，清理一个请求不会删除另一个请求的文件，全部请求结束后不残留缓存

#### Scenario: 服务重启后存在残留
- **WHEN** 服务启动时发现上一次异常退出留下的服务自有缓存
- **THEN** 只清理带有服务专属标识且不属于当前活动请求的残留文件，不触碰其它应用的临时文件

### Requirement: 显示每个音频的有效段处理进度
系统 MUST 在 VAD 完成、短段和低音量过滤完成并确定有效段总数后，为每个音频建立独立的进度状态；每个有效段的 Qwen、CAM++ 和 emotion2vec（按请求开关启用的能力）全部结束后 MUST 将该音频进度增加 1。成功段和失败段都 MUST 更新进度，并显示或记录失败段数量，避免异常段使进度永久停滞。

#### Scenario: 单音频有效段进度
- **WHEN** 一个音频经过过滤后有 1000 个有效 VAD 段
- **THEN** 交互式控制台显示该音频的 `已完成/1000` 进度，所有段结束时达到 `1000/1000`

#### Scenario: 段处理失败仍推进进度
- **WHEN** 某个段的 Qwen、CAM++ 或 emotion2vec 调用失败、超时或被取消
- **THEN** 该段计入已处理数并增加失败计数，进度条继续推进，同时保留既有错误码和错误日志

#### Scenario: 多音频并发进度隔离
- **WHEN** 多个音频同时处理且有效段数量不同
- **THEN** 每个音频拥有独立进度行和计数，不得把一个音频的完成数或失败数累加到另一个音频

#### Scenario: 非交互输出
- **WHEN** 服务运行在非 TTY、日志重定向或无 tqdm 终端环境
- **THEN** 自动关闭进度条控制字符，业务日志、错误日志和可供审计的计数仍可用，输出不包含 ANSI 清屏/回车控制序列

## MODIFIED Requirements

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时发往 Qwen 服务的 VAD 段请求数量（`segment_concurrency`，默认 16），并 MUST 限制所有音频合计发往 Qwen 服务的请求数量（`request_concurrency`，默认 128），超出部分 MUST 等待可用名额后再发出。两个默认值 MUST 与 Qwen 服务的实际承载能力匹配；扩容收益由 Qwen 服务侧承担。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出超过 16 个 VAD 段
- **THEN** 系统按 `segment_concurrency=16` 分批向 Qwen 服务发起请求，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计发往 Qwen 服务的并发请求数不超过 `request_concurrency=128`

### Requirement: 通过配置外置运行参数
Qwen 适配链路的服务地址、模型名、请求并发度、音频并发度、队列容量、超时和 VAD 参数 MUST 来自配置文件；VAD 实例数 MUST 只由 `vad_device` 数组长度决定，动态 VAD 池参数 MUST 不再作为有效配置。

#### Scenario: 修改并发参数后重启生效
- **WHEN** 运维把 `vad_device` 改为 8 个元素、把 `segment_concurrency` 改为 16、把 `audio_concurrency` 改为 8 并重启服务
- **THEN** 服务启动 8 个固定 VAD worker，并按新的 Qwen 与音频并发参数运行

#### Scenario: 配置缺省时使用默认值
- **WHEN** 配置文件缺少相关新配置项
- **THEN** 服务使用默认 `vad_device=["cpu","cpu","cpu","cpu","cuda:0","cuda:0","cuda:0","cuda:0"]`、`segment_concurrency=16`、`request_concurrency=128`、`audio_concurrency=8`，队列满时按既有拒绝策略处理

### Requirement: 配置 v2 链路模型的落卡
系统 MUST 通过 `[model_device_pool]` 下的 `speech_campplus_devices` 与 `emotion2vec_devices` 数组配置 CAM++ 和 emotion2vec 的实例设备；数组元素数量分别决定两个模型池的配置容量，旧 `[model_device]` 单设备字段和 `[model_device.local_model_pool]` 结构 MUST 不再生效。

#### Scenario: 修改落卡配置
- **WHEN** `speech_campplus_devices=["cuda:0","cuda:0","cuda:0"]` 且 `emotion2vec_devices=["cuda:0"]`
- **THEN** 服务按 3 个 CAM++ 实例和 1 个 emotion2vec 实例建立独立模型池，两个池互不借用实例

#### Scenario: 配置缺省
- **WHEN** 任一模型设备数组为空且请求对应的 `showSpk` 或 `showEmotion` 为 true
- **THEN** 请求返回该模型明确的未启用或无可用实例错误，不伪造标签；未请求该能力时不加载该模型

#### Scenario: 旧单设备字段残留
- **WHEN** 配置同时包含旧 `[model_device]` 字段和新的 `[model_device_pool]` 数组
- **THEN** 服务只使用新数组，并记录配置迁移提示
