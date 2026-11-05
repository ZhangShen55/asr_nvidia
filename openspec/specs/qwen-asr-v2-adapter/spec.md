# qwen-asr-v2-adapter Specification

## Purpose

定义基于 Qwen3-ASR 的 `/v2/seacraft_asr` 离线转写能力：本地 VAD 切分与时间轴映射、对 Qwen 服务的并发调用与音频排队、模型自动发现、与 `/v1.1.8` 同构的响应封装，以及后续真实说话人角色与情绪的异步并行计算。

## Requirements

### Requirement: 提供 Qwen 版转写接口
系统 MUST 提供 POST /v3/seacraft_asr，其 multipart 表单参数与 POST /v1.1.8/seacraft_asr 一致（音频文件、语言、词级时间戳、热词、说话人、情绪、语速、角色识别），且 /v1.1.8/seacraft_asr 的行为 MUST NOT 因本能力而改变。

#### Scenario: 提交音频到 v2 接口
- **WHEN** 客户端向 /v3/seacraft_asr 上传有效音频
- **THEN** 系统返回与 /v1.1.8/seacraft_asr 同构的成功响应，顶层包含 language、segments、text、speed_info、load_audio_time_ms、gpu_time_ms

#### Scenario: 未提交可选参数
- **WHEN** 客户端只提交音频文件
- **THEN** 系统按 language=auto、wordTimestamps=false、showSpk=false、showEmotion=false、showRoleIdentify=false 处理

#### Scenario: v1.1.8 行为保持不变
- **WHEN** 客户端向 /v1.1.8/seacraft_asr 提交请求
- **THEN** 系统仍按原有 Paraformer/Whisper 分流逻辑处理，响应结构与错误码不变
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
系统 MUST 使用 `audio_concurrency` 限制 A+B 同时处理的音频文件总数（默认 8）。每个请求 MUST 从准入成功持有槽位至处理及该请求全部临时文件清理完成，并在成功、异常或取消时准确释放。默认不等待，超过总上限 MUST 立即返回总处理队列满；B 类请求另受增强音频并发上限限制。

#### Scenario: 请求数超过音频并发上限
- **WHEN** `queue_wait_timeout_s=0` 且活动音频总数已达到 `audio_concurrency`
- **THEN** 新的 A 或 B 请求立即返回 `code=4013`，`msg` 包含当前总音频并发 `active/capacity`；即使 B 容量也已满，仍优先返回 `4013`

#### Scenario: 排队等待超时
- **WHEN** 配置了大于 0 的排队等待超时，且请求在等待中超过该时长仍未获得完整处理槽位
- **THEN** 系统返回 `{"msg": "请求排队超时，请稍后再试", "code": 4010}`，并回收等待名额

#### Scenario: 队列容量已满
- **WHEN** 等待模式下请求不能立即获准且 A+B 共享等待队列长度达到 `queue_max_size`
- **THEN** 系统返回 `{"msg": "服务繁忙，请稍后再试", "code": 4013}`，不占用活动槽位或增加等待计数

#### Scenario: 音频处理结束释放槽位
- **WHEN** 一个音频请求处理完成且其临时文件已清理
- **THEN** 系统准确释放其总音频槽位；B 同时释放增强槽位，并使最早满足容量条件的等待请求开始处理

#### Scenario: 音频处理异常时释放槽位
- **WHEN** 一个音频请求在处理过程中抛出异常
- **THEN** 系统清理该请求产生的全部临时文件后准确释放其全部活动槽位，不影响后续请求

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时发往 Qwen 服务的 VAD 段请求数量（`segment_concurrency`，默认 16），并 MUST 限制所有音频合计发往 Qwen 服务的请求数量（`request_concurrency`，默认 128），超出部分 MUST 等待可用名额后再发出。两个默认值 MUST 与 Qwen 服务的实际承载能力匹配；扩容收益由 Qwen 服务侧承担。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出超过 16 个 VAD 段
- **THEN** 系统按 `segment_concurrency=16` 分批向 Qwen 服务发起请求，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计发往 Qwen 服务的并发请求数不超过 `request_concurrency=128`

### Requirement: 保留扩展字段
`/v2/seacraft_asr` 的每个成功响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组。`role`、`emotion` 在对应能力未被请求或已启用模型的段级推断失败时为 `null`；请求要求的能力被配置关闭时 MUST 返回请求级错误，不得作为成功转写返回 `null`。

#### Scenario: 返回扩展字段
- **WHEN** 成功响应包含分段
- **THEN** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify` 均未提供或均为 `false`
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`

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
- **THEN** 服务使用默认 `vad_device=["cpu","cpu","cpu","cpu","cuda:0","cuda:0","cuda:0","cuda:0"]`、`segment_concurrency=16`、`request_concurrency=128`、`audio_concurrency=8`、`audio_spk_emo_concurrency=4`，默认满额立即拒绝

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
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并与 Qwen 转写并行执行：每个 VAD 段的转写与所请求的 role/emotion 任务 MUST 同时发起。CAM++ 与 emotion2vec 的段级并发容量 MUST 分别由各自设备数组的成功实例数决定，不增加共享模型推理闸门；B 类整音频准入限制不改变模型池实例容量。两个模型 MUST 分别由独立配置开关控制，emotion2vec 的 PyTorch CUDA 路径 MUST 沿用既有半精度加载策略，TensorRT 路径 MUST 按 emotion2vec_tensorrt_precision 选择 fp16 或 fp32，NPU/CPU 精度 MUST 独立于 TensorRT 配置；请求关闭的能力 MUST 在音频准入前报错。

#### Scenario: 开启角色与情绪能力
- **WHEN** 两个功能开关均为 `true`，请求 `showSpk=true`、`showEmotion=true` 且模型推断成功
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行并分别使用对应模型池

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk`、`showEmotion`、`showRoleIdentify`（均默认 `false`）
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`，两个模型池均不因该请求加载

#### Scenario: 配置未开启说话人能力
- **WHEN** `open_v2_role=false` 且请求需要说话人能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4015`，不开始该请求的模型加载或转写；未请求说话人能力的请求不因该开关报错

#### Scenario: 配置未开启情绪能力
- **WHEN** `open_v2_emotion=false` 且请求需要情绪能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4016`，不开始该请求的模型加载或转写；未请求情绪能力的请求不因该开关报错

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** `open_v2_role` 与 `open_v2_emotion` 中至少一项为 `false`
- **THEN** 仅请求已开启能力时正常进入 B 处理；请求任何关闭的能力时整个请求失败；同时请求两项且两项都关闭时返回 `4018`

#### Scenario: 角色或情绪推断失败
- **WHEN** 请求的增强能力均已开启且存在可用模型实例，但某个分段的说话人或情绪推理失败
- **THEN** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **WHEN** emotion2vec 使用 PyTorch CUDA 后端
- **THEN** 沿用既有 fp16 加载和推理策略，TensorRT 精度配置不改变该路径

#### Scenario: 情绪 TensorRT 精度
- **WHEN** emotion2vec CUDA 实例启用 TensorRT 并显式配置 fp32
- **THEN** 加载该模型 FP32 engine，若精度未配置则选择 FP16 engine，业务情绪映射不变

#### Scenario: 两个模型使用不同后端
- **WHEN** CAM++ CUDA 实例启用 TensorRT，emotion2vec CUDA 实例关闭 TensorRT
- **THEN** 两种增强任务仍与该段 Qwen 任务并行发起，各自按模型池容量执行，emotion2vec 继续使用 PyTorch CUDA

### Requirement: 验证效果与性能边界
系统 MUST 在 `test/audio_file/` 提供的音频上完成效果与性能验证，覆盖 1、2、4、8 路并发下的 GPU 算力占用、端到端时延、转写准确度、时间戳准度，并记录出现幻觉或重复的并发边界。

#### Scenario: 完成并发压测
- **当** 开发人员在目标服务器上按 1、2、4、8 路并发发起请求
- **则** 测试记录包含各并发档位的时延、GPU 算力占用与转写质量观测

#### Scenario: 产出测试报告
- **当** 验证完成
- **则** 仓库中留下一份可归档的测试报告，说明准确度、时间戳准度与幻觉/重复出现的边界

### Requirement: 验证角色与情绪的效果与性能

系统 MUST 在 `test/audio_file/` 的音频上验证 role 与 emotion 的效果与性能：与 `/v1.1.8` 链路的逐段结果比对一致率，并测量开启后的时延、吞吐与显存增量；验证 MUST 覆盖由设备数组决定的非对称模型实例组合，并记录模型池各自的等待情况。

#### Scenario: 角色与情绪效果比对
- **WHEN** 在同一音频、同一 VAD 参数下分别调用 `/v2/seacraft_asr`（`showSpk=true`、`showEmotion=true`）与 `/v1.1.8/seacraft_asr`
- **THEN** 逐段比较 role 的角色语义与 emotion 的业务标签，记录一致率与不一致样本

#### Scenario: 8 路非对称模型池压测
- **WHEN** CAM++ 与 emotion2vec 均部署在 `cuda:0`，设备数组分别配置 6 个和 2 个元素，并发提交 8 个完整音频请求
- **THEN** 记录端到端 P50/P95、CAM++/emotion2vec 各自 busy/waiting、GPU 利用率、显存、错误和吞吐；6:2 仅作为测试组合，不成为固定默认值

#### Scenario: 并行开销可接受
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **THEN** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **WHEN** 在非对称模型实例组合下运行 role/emotion 全开
- **THEN** 记录本机 GPU 的 SM 利用率与显存占用增量、吞吐（音频/分钟、段/秒）以及两个模型池的等待耗时，并写入中文测试报告

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
- **THEN** 服务启动失败并返回明确配置错误，不能以没有 VAD worker 的状态接收 v2 请求

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

### Requirement: 检测并修复复读分段文本
系统 MUST 在分段文本进入文本规范化之前检测复读（同一短语被连续重复输出的生成循环），并对命中的分段把重复部分折叠为一次出现。检测 MUST 依据两项判据：分段文本长度超过按该段语音时长估算的合理上限，且文本中存在连续重复的子串。长度合理的分段 MUST NOT 被修改。

#### Scenario: 复读长串被折叠
- **当** 一个 6 秒的分段文本为 1863 个字符的「最小S零，这个这个这个这个，……」
- **则** 该段文本被折叠为长度正常的短文本，且以「最小S零」开头

#### Scenario: 正常长度的分段不受影响
- **当** 一个分段文本的长度未超过该段时长对应的合理上限
- **则** 该段文本原样保留，不做任何改写

#### Scenario: 长度正常但存在叠词的文本不受影响
- **当** 一个分段文本为「温度载荷，温度载荷，温度载荷，」且长度未超限
- **则** 该段文本原样保留

### Requirement: 长度异常但无复读结构时不得截断
系统 MUST 在分段文本长度超过合理上限、但未检出连续重复结构时保留原文并记录告警，MUST NOT 以截断或截取前 N 个字符的方式修改该文本。系统 MUST NOT 对未检出复读结构的文本做任何形式的内容裁剪。

#### Scenario: 超长但无复读结构的文本被保留
- **当** 一个 5 秒的分段文本长度为 300 字符且其中不存在连续重复的子串
- **则** 该段文本原样返回，其长度与内容均未被修改

#### Scenario: 超长无复读结构时留下告警
- **当** 出现上一场景的文本
- **则** 服务日志中记录一条告警，包含原长度、时长上限与该段时长

### Requirement: 复读判据的阈值可配置且有明确的标定依据
系统 MUST 支持通过配置调整复读检测的阈值，至少包括：是否启用修复的开关、按语音时长估算的字符数上限系数、以及重复子串的最小单元长度与最小重复次数。默认的字符数上限系数 MUST 为 40 字符/秒，其标定依据 MUST 为真实样本统计：正常段的高分位值约 20.3 字符/秒、观测到的最高值约 26.2，而复读段为 282~313。判定为异常的最低字符数 MUST 不低于 120。

#### Scenario: 默认阈值下的判定边界
- **当** 一个 10 秒的分段文本长度为 300 字符
- **则** 合理上限为 400 字符，该段不被判定为长度异常

#### Scenario: 关闭复读修复开关
- **当** 配置把复读修复开关设为关闭
- **则** 系统不做任何复读检测与折叠，分段文本原样返回

#### Scenario: 调整字符数上限系数
- **当** 运维人员把字符数上限系数从 40 调整为 60 并重启服务
- **则** 系统按新的系数估算合理上限并据此判定长度异常

### Requirement: 复读修复独立于文本规范化开关
系统 MUST 在文本规范化之前执行复读检测与修复，该规则 MUST 独立于文本规范化开关生效，即规范化关闭时同样执行。修复 MUST NOT 改变文本规范化与标点卫生规则的行为，两类处理 MUST NOT 互相覆盖。

#### Scenario: 文本规范化关闭时仍修复复读
- **当** 配置关闭文本规范化且某个分段为复读长串
- **则** 该段仍被折叠为短文本

#### Scenario: 复读修复与标点卫生并存
- **当** 一个分段既是复读长串又以标点开头
- **则** 先完成复读折叠，再由标点卫生规则处理其段首标点，两条规则都生效

### Requirement: 复读修复行为可观测
系统 MUST 在发生折叠时记录日志，日志 MUST 包含原文本长度、折叠后长度、按该段时长计算的长度上限与该段时长。系统 MUST NOT 把被折叠文本的完整内容写入日志，MUST 只记录其前缀。

#### Scenario: 折叠发生时留下日志
- **当** 某个分段被折叠
- **则** 服务日志中出现一条包含原长度、折叠后长度、时长上限与时长的记录

#### Scenario: 未触发修复时不产生噪声日志
- **当** 全部请求的分段文本均未触发复读修复
- **则** 不产生任何复读告警日志

### Requirement: 提供细小声音统计报告
系统 MUST 为参考音频的 46 个人工标注区间输出可复核统计，至少包含每段时长、RMS dBFS、峰值 dBFS、VAD 后验分数摘要、VAD 命中状态，并同时提供未标注对照区间的分布统计。

#### Scenario: 生成参考音频统计
- **WHEN** 执行参考音频统计入口并读取目标音频及人工标注
- **THEN** 报告包含 46 个细小声音区间的统计、未标注区间统计、音频总时长和统计单位，且不修改原始音频与标注文件

#### Scenario: 标注格式异常
- **WHEN** 人工标注文件包含说明文字、裸标记或缺失逗号
- **THEN** 统计入口跳过非数据说明并解析有效区间；无法解析的对象被计数并在报告中告警，不得静默伪造统计结果

### Requirement: 配置低音量过滤参数
系统 MUST 在 `[qwen_asr]` 中提供低音量过滤参数，明确其物理单位、默认值和边界；服务启动时 MUST 校验参数，并将生效值传入每个 VAD worker，修改后重启生效。

#### Scenario: 使用配置的低音量门限
- **WHEN** 运维人员设置低音量门限并重启服务
- **THEN** 所有 VAD worker 使用同一门限判定帧或分段，状态和诊断报告能够显示该生效值

#### Scenario: 配置值非法
- **WHEN** 低音量门限缺失、非有限数值或超出支持范围
- **THEN** 服务启动失败并返回包含配置路径和允许范围的可读错误

### Requirement: 自动搜索可行低音量阈值
系统 MUST 支持围绕统计分布自动生成并滑动搜索候选低音量阈值；每个候选值 MUST 同时评估细小声音抑制率、未标注区间保留率、VAD 段数和 Qwen 请求数，并输出排序后的候选结果。

#### Scenario: 统计值附近找到可行阈值
- **WHEN** 候选阈值存在同时满足配置的细小声音抑制率和正常语音保留率的值
- **THEN** 报告给出该值、搜索范围、步长、命中指标和推荐理由，且推荐值可直接写入标准配置

#### Scenario: 单一阈值没有可行解
- **WHEN** 在初始范围和统计值附近的细化范围内没有候选值同时满足两类指标
- **THEN** 报告明确标记“无可行单阈值”，不得仅以段数下降判定成功，并给出组合音量/信噪比/连续有效帧或段级能量规则的下一步候选

### Requirement: 低音量段不得发送给 Qwen
系统 MUST 在向 Qwen-ASR 发起请求前优先应用配置驱动的低音量过滤；被判定为低音量无效段 MUST 不产生 Qwen 请求、不进入响应分段和最终统计，正常语音段的时间轴保持不变。音频段时长 MUST NOT 作为第一阶段过滤依据。

#### Scenario: 过滤细小声音
- **WHEN** VAD 输出段满足低音量过滤条件
- **THEN** 系统丢弃该段并减少对应的 Qwen 请求数，日志或测试报告记录丢弃原因，且该标注时间区间不出现有效识别内容

#### Scenario: 保留正常语音
- **WHEN** VAD 输出段未满足低音量过滤条件
- **THEN** 系统按原有流程发送给 Qwen，响应段的 `bg`、`ed` 和扩展字段语义保持不变

### Requirement: 验证细小声音区间内无识别内容
系统 MUST 对人工标注的细小声音时间区间检查最终识别结果；与标注区间重叠的响应分段及其 `segment_text` MUST 被记录，并区分“未产生识别内容”“仅有口气词”和“包含有效词语”三种结果。

#### Scenario: 标注区间没有识别内容
- **WHEN** 低音量过滤生效且某标注区间没有与响应分段重叠的有效文本
- **THEN** 该区间计为声音过滤成功，不产生对应 Qwen 请求或最终响应分段

#### Scenario: 标注区间仍有识别内容
- **WHEN** 某标注区间内出现有效 `segment_text`
- **THEN** 报告将该区间计为低音量过滤失败，即使后续文本后处理删除了口气词，也不得把它计为 VAD 侧成功

### Requirement: 参考音频验收必须区分帧和段
参考音频验收 MUST 同时报告逐帧静音判定、最终段级区间命中和区间内识别内容；只有细小声音抑制率、标注区间无有效识别内容和正常语音保留率均达到配置目标时，阈值搜索才可标记为成功。

#### Scenario: 逐帧降低但段级仍命中
- **WHEN** 某阈值降低了细小声音帧的语音判定比例但最终 VAD 段仍覆盖标注区间
- **THEN** 报告将其标记为段级过滤失败，不得将逐帧比例下降当作端到端成功

#### Scenario: 端到端请求量下降
- **WHEN** 某阈值同时达到两类区间指标并减少发往 Qwen 的段请求
- **THEN** 报告标记为可选部署值，并保存音频时长、候选参数、段数和请求数以便复核

#### Scenario: 声音规则无可行解
- **WHEN** 低音量、SNR 和连续有效帧规则仍无法同时满足三类验收指标
- **THEN** 报告明确标记声音规则无可行解，只有在后续阶段才允许评估音频段时长条件

### Requirement: v2 本地增强能力按模型池独立排队

系统 MUST 让每个 VAD 段的 CAM++ 请求进入 CAM++ 独立模型池、emotion2vec 请求进入 emotion2vec 独立模型池；本地增强能力的等待 MUST 不改变 Qwen-ASR 的请求并发配置，也 MUST NOT 让一个模型池借用另一个模型池的实例。

#### Scenario: CAM++ 槽位少于 emotion2vec 槽位
- **WHEN** CAM++ 设备数组配置 3 个实例且 emotion2vec 设备数组配置 1 个实例
- **THEN** CAM++ 和 emotion2vec 分别按自身容量排队，空闲的 CAM++ 实例不能被 emotion2vec 请求占用

#### Scenario: 本地模型池等待期间 Qwen 仍可调度
- **WHEN** 某个 VAD 段等待 CAM++ 或 emotion2vec 实例
- **THEN** 其它已获得本地模型实例的段仍可并行调用 Qwen，且 Qwen 的 `segment_concurrency` 与 `request_concurrency` 上限保持生效

### Requirement: 记录本地模型池排队对端到端耗时的影响

系统 MUST 为 v2 请求记录 CAM++ 和 emotion2vec 的调用次数、推理耗时和模型池等待耗时，并 MUST 能在测试报告中分别比较本地模型等待、VAD、Qwen 分段处理和其它后处理阶段。

#### Scenario: 8 路完整音频压测
- **WHEN** 使用同一份完整音频并发提交 8 个 `showSpk=true`、`showEmotion=true` 请求
- **THEN** 结果包含两个模型池各自的 busy/waiting、等待耗时、P50/P95、错误数和音频吞吐

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量而不新增槽位配置
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

### Requirement: 完成固定配置的全量 WAV 质量压测
系统 MUST 支持对目标目录递归发现的全部 WAV 执行固定配置的 `/v2/seacraft_asr` 压测，并保存每个请求的响应、错误、耗时和过滤统计。测试配置 MUST 使用 RMS `-41.0 dBFS`、低音量过滤开启、`segment_concurrency=4`、说话人与情绪均关闭、音频并发 16。

#### Scenario: 45 个 WAV 全部配对并执行
- **WHEN** 目标目录包含 45 个 WAV 和对应的 `seaco_paraformer-<WAV 文件名>.json`
- **THEN** 系统按同目录文件名配对并执行 45 个请求，缺失或多余配对文件时测试失败并列出清单

#### Scenario: 记录端到端分位数
- **WHEN** 全量请求完成
- **THEN** 报告给出成功数、失败数、超时数、端到端 P50、P95、最大值和每个文件的耗时

### Requirement: 按时间和文本证据评估低音量过滤
系统 MUST 将 Qwen 响应段与 Paraformer 基线段按时间重叠对齐，区分基线独有、Qwen 独有和双方重叠段，并为基线独有段保留基线文本、时间、RMS、时长和上下文。系统 MUST 将“细小声音无关内容”作为候选分类结果，不得仅凭段数减少宣称准确率提升。

#### Scenario: 基线独有段被识别为过滤候选
- **WHEN** Paraformer 段在相同时间范围内没有 Qwen 响应文本
- **THEN** 该段进入候选清单，并记录其基线 `segment_text`、前后相邻文本、音量和时间重叠证据

#### Scenario: 候选经过人工复核分类
- **WHEN** 候选清单生成
- **THEN** 每个候选被分类为无关细小声音、有效授课内容、VAD/边界差异或不确定，并统计各类数量，未复核项不得计入过滤成功率

#### Scenario: 无人工真值时限制准确性表述
- **WHEN** 测试仅有 Paraformer JSON 而没有人工转写真值
- **THEN** 报告使用相对基线一致性和人工复核结果，不输出绝对 WER、准确率或召回率结论

### Requirement: 审计复读后处理并识别漏网模式
系统 MUST 对最终 Qwen 响应扫描段内连续复读、字符级复读、相邻段重复、跨段重复、异常字符速率和超长文本，并与同一测试窗口的复读修复及复读检测日志对账，区分已修复、检测后保留、漏网和误报。

#### Scenario: 已修复复读可追溯
- **WHEN** Qwen 原始输出触发复读后处理
- **THEN** 响应中不再出现原始长复读串，日志包含原长度、修复后长度、段时长和阈值，报告将其计为已修复

#### Scenario: 复读漏网被标记
- **WHEN** 最终响应仍包含重复结构、异常字符速率或跨段重复，且没有对应的修复日志
- **THEN** 报告列出文件、时间段、文本样例、触发的检测规则和建议的后处理覆盖方向

#### Scenario: 正常重复口语不误报
- **WHEN** 文本长度正常但包含自然叠词或课堂中合法的重复表达
- **THEN** 系统不得将其计为复读漏网，报告保留该样例及不计入的理由

### Requirement: 形成可归档报告和后续优化决策
系统 MUST 输出机器可读明细和中文汇总报告，报告包含配置、文件配对、性能分位数、过滤候选分类、复读对账、失败清单和证据路径。若确认存在稳定复读漏网，报告 MUST 提出独立的后处理优化 change 范围、验收指标和回归样本；本验证不能以未验证的规则直接改变生产响应。

#### Scenario: 全量报告可复核
- **WHEN** 测试和人工复核结束
- **THEN** 任何汇总数字都能通过逐文件 JSON、原始响应、基线文件和日志快照重新计算

#### Scenario: 发现漏网后形成下一步方案
- **WHEN** 漏网样本满足同一模式在至少两个文件或两个请求中重复出现
- **THEN** 报告创建后处理优化建议，包含模式定义、误伤防护、开关/阈值策略和回归测试样本，不在本验证中直接上线修改

### Requirement: 按请求的增强能力分类音频
系统 MUST 将 `showSpk`、`showEmotion`、`showRoleIdentify` 均为 `false` 的请求归为 A，任一为 `true` 归为 B；三个参数默认值均为 `false`。`showSpk` 或 `showRoleIdentify` 为 `true` MUST 视为需要说话人能力，`showEmotion=true` MUST 视为需要情绪能力；每个音频至多占用一个增强槽位。

#### Scenario: 默认普通转写
- **WHEN** 请求未提供三个增强参数
- **THEN** 请求归为 A，只消耗一个总音频槽位，且参数文档展示三个默认值均为 `false`

#### Scenario: 单项增强请求
- **WHEN** 请求仅设置 `showSpk=true` 或仅设置 `showEmotion=true`，对应能力已开启
- **THEN** 请求归为 B，同时消耗一个总音频槽位和一个增强槽位

#### Scenario: 角色识别请求
- **WHEN** 请求仅设置 `showRoleIdentify=true` 且说话人能力已开启
- **THEN** 请求归为 B，需要说话人能力并同时消耗一个总音频槽位和一个增强槽位

#### Scenario: 同时请求多项增强
- **WHEN** 请求同时开启两个或三个增强参数且所需能力均已开启
- **THEN** 请求归为 B，仅占用一个总音频槽位及一个增强槽位

### Requirement: 增强音频容量是总容量的子集
系统 MUST 始终满足 `active_A + active_B <= audio_concurrency` 且 `active_B <= audio_spk_emo_concurrency`。A 仅占总容量，B 原子取得总容量和增强容量；等待或拒绝的请求 MUST NOT 预占任何活动槽位。默认不等待时，总容量满返回 `4013`；仅 B 容量满返回 `4017`，消息 MUST 标识 `spk_emo` 和增强容量使用比。

#### Scenario: 三个 A 和四个 B
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `3A+4B`
- **THEN** 新 B 返回 `4017` 和 `spk_emo队列已满，请稍后再试（当前增强音频并发：4/4）`；新 A 可获准形成 `4A+4B`

#### Scenario: 五个 A 和三个 B
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `5A+3B`
- **THEN** 新 A 和新 B 均返回 `4013`，B 不因还有一个增强名额而获准

#### Scenario: 八个普通音频占满总容量
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `8A`
- **THEN** 新 A 和新 B 均返回 `4013`，系统不为 B 预留总容量

#### Scenario: 总容量和增强容量同时已满
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `4A+4B`
- **THEN** 新 A 和新 B 均返回 `4013`，总容量错误优先于增强容量错误

#### Scenario: 两类容量均有余量
- **WHEN** 总上限为 8、增强上限为 4，正在处理 `4A+3B` 且无等待请求
- **THEN** 下一请求无论为 A 或 B 均可获准，所有容量约束仍成立

#### Scenario: 增强容量满不占满总容量
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `0A+4B`
- **THEN** 新 B 返回 `4017`，新 A 可获准且增强活动数保持 4

### Requirement: 关闭能力在准入之前拒绝
系统 MUST 在音频准入、等待排队、音频预处理和模型调用之前校验请求所需的功能开关。仅所请求的说话人能力关闭返回 `4015`，仅所请求的情绪能力关闭返回 `4016`，同时请求两项且两项都关闭返回 `4018`。业务错误 MUST 保持 HTTP 200 和 `{"msg": ..., "code": ...}` 格式，不返回成功转写分段。

#### Scenario: 两种能力关闭但只做普通转写
- **WHEN** 两个功能开关均为 `false` 且三个增强参数均为 `false`
- **THEN** 请求按 A 处理，只受总音频容量限制，增强活动数不增加

#### Scenario: 两种能力关闭且同时请求两种增强
- **WHEN** 两个功能开关均为 `false`，请求需要说话人及情绪能力
- **THEN** 请求返回 `4018`，消息说明两种能力均未启用；即使总容量已满也不返回 `4013`，且所有活动、等待计数不变

#### Scenario: 同时请求两种能力但只有一种关闭
- **WHEN** 请求需要说话人及情绪能力，恰好只有一个功能开关为 `false`
- **THEN** 说话人关闭返回 `4015`，情绪关闭返回 `4016`，请求整体失败且不进行部分转写

#### Scenario: 两种能力关闭但仅请求一种
- **WHEN** 两个功能开关均为 `false`，但请求仅需要说话人能力或仅需要情绪能力
- **THEN** 分别返回 `4015` 或 `4016`，不使用组合错误 `4018`

### Requirement: 等待队列按可用容量调度
在 `queue_wait_timeout_s>0` 时，系统 MUST 以 `queue_max_size` 限制 A+B 的共享等待总数，保持同类请求的到达顺序，并优先准入最早满足容量条件的请求。B 被增强容量阻塞但总容量有余量时，A MUST 能绕过该 B；当较早 B 已满足两类容量条件时，较晚 A MUST NOT 抢占其名额。

#### Scenario: A 绕过被增强容量阻塞的 B
- **WHEN** `3A+4B`、总上限 8、增强上限 4，等待队列已有 B，此时新 A 到达
- **THEN** B 继续等待且不预占总槽位，A 能使用空闲总槽位；B 不因本次 A 获准而失去其同类顺序

#### Scenario: 较早 B 变为可处理
- **WHEN** 较早等待的 B 和较晚等待的 A 同时满足容量条件
- **THEN** B 先获准，不被后到 A 持续抢占；同类请求按到达顺序处理

#### Scenario: 两类共享等待上限
- **WHEN** A 等待数与 B 等待数之和达到 `queue_max_size`，新请求也不能立即获准
- **THEN** 新请求返回 `4013`，不会因分成两类而把等待容量扩大为两倍

#### Scenario: 等待 B 不阻塞空闲总容量
- **WHEN** B 仅因增强容量不足而等待，其他容量仍有余量
- **THEN** 该 B 不占用总音频活动名额或增强活动名额，等待超时仍返回 `4010`

### Requirement: 取消和并发竞争不泄漏名额
系统 MUST 在等待取消、准入与取消竞争、准入与超时竞争、处理取消和清理异常时保持计数一致，每个已授予的活动名额至多释放一次且最终必定释放；未授予的请求只移除等待名额。B 的两个活动名额 MUST 作为同一个处理周期取得和释放。

#### Scenario: 等待中取消
- **WHEN** 等待中的 A 或 B 请求被取消
- **THEN** 对应等待数回收且活动计数不增加，后续满足容量条件的请求可继续获得名额

#### Scenario: 获准与取消同时发生
- **WHEN** 槽位刚授予请求时该请求同时取消，调用方尚未开始处理
- **THEN** 系统回收已授予的全部槽位，不泄漏、不重复释放，也不产生负数计数

#### Scenario: 获准与等待超时竞争
- **WHEN** 等待截止时刻与槽位可用同时发生
- **THEN** 请求仅有一种结果：成功取得完整名额并负责释放，或返回 `4010` 且不保留名额

#### Scenario: 处理取消或清理异常
- **WHEN** 已获准的 B 请求处理取消，或其临时文件清理发生异常
- **THEN** 系统完成必要的清理尝试并释放一个总音频槽位及一个增强槽位，保留原有错误或取消语义

### Requirement: 本地后端切换保持音频准入和取消契约
系统 MUST 在切换 CAM++/emotion2vec 后端时保留既有 A/B 分类、总音频/B 音频上限、段/Qwen 并发、能力开关、错误优先级和清理规则。请求取消 MUST 在已启动的线程/设备工作、结果读取和必要清理完成后归还对应资源，MUST NOT 因 TensorRT 异步提交返回而提前释放租约。

#### Scenario: B 上限不因后端切换而变化
- **WHEN** 总音频上限为 8、B 上限为 4，已有 3A+4B，后端从 PyTorch 改为 TensorRT
- **THEN** 新 B 仍返回 4017，新 A 仍可获准，模型实例容量仍由各自设备数组决定

#### Scenario: 普通转写不触发增强模型
- **WHEN** 三个增强请求参数均为 false，模型 TensorRT 开关为 true
- **THEN** 请求仍为 A，不因后端配置加载或调用 CAM++/emotion2vec，也不要求其 engine

#### Scenario: 启用 TensorRT 后请求关闭能力
- **WHEN** 模型 TensorRT 开关为 true，但请求要求的功能被 features 配置关闭
- **THEN** 仍在音频准入前按既有 4015/4016/4018 返回，不因存在 engine 而绕过校验

#### Scenario: CUDA 工作尚未完成时取消
- **WHEN** 已准入 B 请求取消，而某实例已经提交 TensorRT 工作且设备尚未完成
- **THEN** 请求仍持有该模型及整音频租约，完成设备工作和必要清理后才归还，最终计数一致

#### Scenario: engine 初始化失败
- **WHEN** 已准入请求所需模型的全部 TensorRT 实例初始化失败
- **THEN** 返回对应模型不可用 4015 或 4016，并完成请求清理和音频租约释放，不自动切为 PyTorch 或改用组合能力关闭错误

### Requirement: 移除 v2 Qwen 路由
服务 MUST NOT 再注册或文档化 POST /v2/seacraft_asr；客户端访问该路径 MUST 得到未找到路由的结果，v3 是本链路唯一公开版本。

#### Scenario: 访问已移除的 v2 路径
- **WHEN** 客户端向 /v2/seacraft_asr 发起请求
- **THEN** 服务不执行转写处理并返回未找到路由的结果

#### Scenario: v3 语义迁移
- **WHEN** 客户端把原 v2 请求路径替换为 /v3/seacraft_asr
- **THEN** VAD 切分、Qwen 调用、A/B 音频准入、role/emotion 错误码和响应字段继续遵循本能力其余要求
