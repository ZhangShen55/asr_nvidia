# qwen-asr-v2-adapter Specification

## Purpose

定义基于 Qwen3-ASR 的 `/v2/seacraft_asr` 离线转写能力：本地 VAD 切分与时间轴映射、对 Qwen 服务的并发调用与音频排队、模型自动发现、与 `/v1.1.8` 同构的响应封装，以及后续真实说话人角色与情绪的异步并行计算。

## Requirements

### Requirement: 提供 Qwen 版转写接口
系统 MUST 提供 `POST /v2/seacraft_asr`，其 multipart 表单参数与 `POST /v1.1.8/seacraft_asr` 一致（音频文件、语言、词级时间戳、热词、说话人、情绪、语速、角色识别），且 `/v1.1.8/seacraft_asr` 的行为 MUST NOT 因本能力而改变。

#### Scenario: 提交音频到 v2 接口
- **当** 客户端向 `/v2/seacraft_asr` 上传有效音频
- **则** 系统返回与 `/v1.1.8/seacraft_asr` 同构的成功响应，顶层包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms`

#### Scenario: 未提交可选参数
- **当** 客户端只提交音频文件
- **则** 系统按 `language=auto`、`wordTimestamps=false`、`showSpk=false`、`showEmotion=false`、`showRoleIdentify=false` 处理

#### Scenario: v1.1.8 行为保持不变
- **当** 客户端向 `/v1.1.8/seacraft_asr` 提交请求
- **则** 系统仍按原有 Paraformer/Whisper 分流逻辑处理，响应结构与错误码不变

### Requirement: 以 VAD 段作为转写与时间轴单元
系统 MUST 在服务本地对音频做 VAD 切分，并以「1 个 VAD 段对应 1 次 Qwen 请求、对应 1 个响应分段」的方式组织结果，响应分段的 `bg` 与 `ed` MUST 取该 VAD 段的起止时间；系统 MUST NOT 合并相邻 VAD 段。

#### Scenario: 音频被切分为多段
- **当** 音频包含多段人声
- **则** 系统按 VAD 段逐段请求 Qwen 服务，响应分段的起止时间与对应 VAD 段边界一致

#### Scenario: 音频无人声
- **当** VAD 未检测到任何有效人声
- **则** 系统返回 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`

#### Scenario: 过滤过短语音段
- **当** 某 VAD 段时长小于配置的最小语音段时长
- **则** 系统不向 Qwen 服务发起该段的请求，且该段不产出响应分段

### Requirement: 自动发现 Qwen 服务模型
系统 MUST 在未显式配置模型名时，通过调用 Qwen 服务的 `/v1/models` 接口获取可用模型标识并使用之，且请求 MUST NOT 携带 API Key。

#### Scenario: 未配置模型名
- **当** 配置中的模型名为空
- **则** 系统从 `/v1/models` 返回的列表中选取模型标识用于转写请求

#### Scenario: 显式配置模型名
- **当** 配置中填写了模型名
- **则** 系统直接使用该模型名，不调用模型发现

#### Scenario: 模型发现失败
- **当** `/v1/models` 不可达或返回空列表且未显式配置模型名
- **则** 系统返回 `{"msg": "Qwen 服务不可用或未发现可用模型", "code": 4011}`

### Requirement: 解析 Qwen 转写结果
系统 MUST 解析 Qwen 服务返回的原始文本，剥离 `language <语种><asr_text>` 前缀，取出转写文本与识别语种；返回文本为空或语种为空的段 MUST NOT 产出响应分段。

#### Scenario: 返回带语种前缀的文本
- **当** Qwen 服务返回 `language Chinese<asr_text>识别文本`
- **则** 系统产出文本为 `识别文本`、语种为 `Chinese` 的分段

#### Scenario: 段内无有效语音
- **当** Qwen 服务返回 `language None<asr_text>`
- **则** 该段不产出响应分段

#### Scenario: 服务返回多段拼接文本
- **当** Qwen 服务返回包含多个 `language <语种><asr_text>` 片段的文本
- **则** 系统拼接各片段文本并取第一个可识别的语种

#### Scenario: Qwen 服务调用失败
- **当** Qwen 服务不可达、超时或返回错误状态
- **则** 系统按配置的失败策略重试，重试耗尽后返回 `{"msg": "Qwen 服务调用失败，请稍后再试", "code": 4012}`

### Requirement: 按规则透传语言参数
系统 MUST 保留 `/v2/seacraft_asr` 的 `language` 请求参数（默认 `auto`），但当取值为 `auto`、空串或未提交时 MUST NOT 向 Qwen 服务发送 `language` 字段；提交具体语种时 MUST 原样透传。响应的 `language` MUST 取模型实际识别出的语种，而非请求参数回显。

#### Scenario: 未提交语言参数
- **当** 客户端未提交 `language`，或提交 `auto`
- **则** 系统不向 Qwen 服务发送 `language` 字段，由模型自动识别语种，响应 `language` 为识别结果

#### Scenario: 提交具体语种
- **当** 客户端提交 `language=en`
- **则** 系统把 `en` 透传给 Qwen 服务

#### Scenario: Qwen 服务不支持该语种
- **当** 提交的语种不在 Qwen 服务支持的清单内
- **则** 系统返回 `{"msg": "不支持的语言", "code": 4009}`

### Requirement: 返回语种占比字典
系统 MUST 在 `/v2/seacraft_asr` 的响应顶层返回 `language_ratio` 字典，键为模型识别出的语种的**简写标识**（例如中文 `zh`、英文 `en`、粤语 `yue`），值为该语种文本字符数占总字符数的百分比（保留一位小数，各值之和为 100.0）；无有效文本时返回空字典。语种判定来源与判定逻辑 MUST NOT 改变，仅对键名做简写映射；无法映射的语种 MUST 原样透传其原始名称。

#### Scenario: 单语种音频
- **当** 音频全部识别为中文
- **则** 响应 `language_ratio` 为 `{"zh": 100.0}`

#### Scenario: 中英混合音频
- **当** 音频中部分段落识别为英文、其余为中文
- **则** 响应 `language_ratio` 同时包含 `zh` 与 `en` 两个键，值为各自文本字符占比

#### Scenario: 无法映射的语种
- **当** 模型识别出简写表中不存在的语种
- **则** 响应 `language_ratio` 中该语种以模型返回的原始名称作为键

#### Scenario: 无有效文本
- **当** 没有任何有效转写文本
- **则** 响应 `language_ratio` 为空字典

### Requirement: 限制同时处理的音频数量
系统 MUST 限制同时处理的音频文件数量（默认 8），并 MUST 在处理结束（含异常）时立即释放槽位、删除该请求产生的全部临时文件。默认配置下超出上限的请求 MUST 立即返回队列已满错误，不得静默排队。

#### Scenario: 请求数超过音频并发上限
- **当** 并发到达的音频请求数超过配置的上限
- **则** 超出的请求立即返回 `code=4013`，且 `msg` 包含当前音频并发 `active/capacity`

#### Scenario: 排队等待超时
- **当** 配置了大于 0 的排队等待超时，且请求在等待中超过该时长仍未获得处理槽位
- **则** 系统返回 `{"msg": "请求排队超时，请稍后再试", "code": 4010}`

#### Scenario: 队列容量已满
- **当** 等待队列长度达到配置的上限且无法再接纳
- **则** 系统返回 `{"msg": "服务繁忙，请稍后再试", "code": 4013}`

#### Scenario: 音频处理结束释放槽位
- **当** 一个音频请求处理完成
- **则** 系统释放其占用槽位并让队列中的下一个请求开始处理

#### Scenario: 音频处理异常时释放槽位
- **当** 一个音频请求在处理过程中抛出异常
- **则** 系统同样释放其占用槽位并删除全部临时文件，不影响后续请求

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时发往 Qwen 服务的 VAD 段请求数量（`segment_concurrency`，默认 16），并 MUST 限制所有音频合计发往 Qwen 服务的请求数量（`request_concurrency`，默认 128），超出部分 MUST 等待可用名额后再发出。两个默认值 MUST 与 Qwen 服务的实际承载能力匹配；扩容收益由 Qwen 服务侧承担。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出超过 16 个 VAD 段
- **THEN** 系统按 `segment_concurrency=16` 分批向 Qwen 服务发起请求，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计发往 Qwen 服务的并发请求数不超过 `request_concurrency=128`

### Requirement: 保留扩展字段
`/v2/seacraft_asr` 的每个响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组（Qwen 服务不返回词级时间戳）；`role` 与 `emotion` 在对应能力未开启或推断失败时为 `null`。

#### Scenario: 返回扩展字段
- **当** 响应包含分段
- **则** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **当** 请求未携带 `showSpk` / `showEmotion`
- **则** 每个分段的 `role` 与 `emotion` 均为 `null`

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

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并 MUST 与 Qwen 转写并行执行：每个 VAD 段的转写任务与该段的 role/emotion 任务 MUST 在同一时刻发起，MUST NOT 等待全部转写完成后再计算 role/emotion。两个模型 MUST 分别由独立配置开关控制（`open_v2_role` 与 `open_v2_emotion`，默认均为 true）；实际计算 MUST 由请求参数 `showSpk` / `showEmotion` 触发（默认均为 false）；emotion2vec MUST 以半精度加载。

#### Scenario: 开启角色与情绪能力
- **当** 请求参数 `showSpk` 与 `showEmotion` 均为 true
- **则** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行

#### Scenario: 角色与情绪能力未开启
- **当** 请求未携带 `showSpk` / `showEmotion`（默认 false）
- **则** 响应分段的 `role` 与 `emotion` 保持 `null`

#### Scenario: 配置未开启说话人能力
- **当** 配置中 `open_v2_role` 为 false
- **则** 系统不加载 CAM++ 模型，响应分段的 `role` 保持 `null`

#### Scenario: 配置未开启情绪能力
- **当** 配置中 `open_v2_emotion` 为 false
- **则** 系统不加载 emotion2vec 模型，响应分段的 `emotion` 保持 `null`

#### Scenario: 配置未同时开启说话人与情绪能力
- **当** 配置中 `open_v2_role` 与 `open_v2_emotion` 中任一为 false
- **则** 系统只跳过该开关对应的模型加载，另一个模型仍可按请求参数独立启用

#### Scenario: 角色或情绪推断失败
- **当** 说话人或情绪模型推理失败
- **则** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **当** emotion2vec 模型被加载
- **则** 模型以半精度（fp16）加载，显存占用约 1 GB 级别，情绪精度不做额外调优

### Requirement: 验证效果与性能边界
系统 MUST 在 `test/audio_file/` 提供的音频上完成效果与性能验证，覆盖 1、2、4、8 路并发下的 GPU 算力占用、端到端时延、转写准确度、时间戳准度，并记录出现幻觉或重复的并发边界。

#### Scenario: 完成并发压测
- **当** 开发人员在目标服务器上按 1、2、4、8 路并发发起请求
- **则** 测试记录包含各并发档位的时延、GPU 算力占用与转写质量观测

#### Scenario: 产出测试报告
- **当** 验证完成
- **则** 仓库中留下一份可归档的测试报告，说明准确度、时间戳准度与幻觉/重复出现的边界

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 在 `test/audio_file/` 的音频上验证 role 与 emotion 的效果与性能：与 `/v1.1.8` 链路的逐段结果比对一致率，并测量开启后的时延、吞吐与显存增量；开启 role/emotion MUST NOT 改变转写文本。

#### Scenario: 角色与情绪效果比对
- **当** 在同一音频、同一 VAD 参数下分别调用 `/v2/seacraft_asr`（`showSpk=true`、`showEmotion=true`）与 `/v1.1.8/seacraft_asr`
- **则** 逐段比较 role 的角色语义与 emotion 的业务标签，记录一致率与不一致样本

#### Scenario: 并行开销可接受
- **当** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **则** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **当** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **则** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **当** 在 1 / 4 / 8 音频并发下运行 role/emotion 全开
- **则** 记录本机 GPU 的 SM 利用率与显存占用增量、吞吐（音频/分钟、段/秒），并写入测试报告

### Requirement: 规范化输出分段文本的标点
系统 MUST 在文本规范化之后对 `/v2/seacraft_asr` 的分段文本做标点卫生处理：当一个分段的文本去除空白后只剩标点符号时，该分段 MUST 从响应中移除；当一个分段的文本首个字符是标点符号时，MUST 移除该开头标点。该规则 MUST 独立于文本规范化开关生效，即规范化关闭时同样执行。

#### Scenario: 分段只有标点
- **当** 某个分段的文本只有标点符号（如 `。`、`！`）
- **则** 该分段不出现在响应的 `segments` 数组中

#### Scenario: 分段以标点开头
- **当** 某个分段的文本以标点符号开头（如 `，就。`）
- **则** 响应中该分段的文本不再以标点开头

#### Scenario: 规范化关闭时规则仍生效
- **当** 文本规范化被关闭
- **则** 上述两条标点处理规则仍然作用于响应分段

### Requirement: 提供文本规范化开关
系统 MUST 支持通过配置控制文本规范化的启用与行为：启用开关默认开启（`enabled` 为 true），并支持独立控制独立数字转换（`enable_standalone_number`）、数字逐位读出（`enable_0_to_9`）与语气词剔除（`remove_interjections`）。语气词剔除开关 MUST 可在配置中显式设置，默认行为 MUST 为剔除「呃」「啊」等语气词；剔除产生的标点副作用 MUST 由标点卫生规则清理。

#### Scenario: 默认开启文本规范化
- **当** 部署配置未显式覆盖文本规范化开关
- **则** 系统对转写文本执行规范化，并按配置剔除语气词

#### Scenario: 显式保留语气词
- **当** 部署配置把语气词剔除开关设为 false
- **则** 系统保留转写文本中的语气词

#### Scenario: 关闭文本规范化
- **当** 部署配置关闭文本规范化
- **则** 系统直接返回模型原始转写文本，不做数字与标点规范化

#### Scenario: 规范化依赖不可用
- **当** 文本规范化依赖缺失或初始化失败且失败策略为 `passthrough`
- **则** 请求仍成功返回，文本保持 ASR 原始输出

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
