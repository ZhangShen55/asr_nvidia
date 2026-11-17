# Spec Delta

## MODIFIED Requirements

### Requirement: 自动发现 Qwen 服务模型
系统 MUST 在未显式配置模型名时，从配置的本地模型目录加载 Qwen3-ASR 权重与处理器，MUST NOT 依赖外部服务的模型发现接口；模型加载 MUST 放置到配置的 NPU 设备，MUST NOT 携带任何 API Key。加载失败 MUST 映射为 `4011`。

#### Scenario: 未配置模型名
- **WHEN** 配置中的模型名为空
- **THEN** 系统从本地模型目录加载权重与处理器并放置到目标 NPU 设备，用于后续转写

#### Scenario: 显式配置模型名
- **WHEN** 配置中填写了模型名
- **THEN** 系统直接使用该模型名标识本地加载的模型，不发起任何模型发现请求

#### Scenario: 模型发现失败
- **WHEN** 本地模型目录缺失、权重不可读或目标设备不可用
- **THEN** 系统返回 `{"msg": "Qwen 服务不可用或未发现可用模型", "code": 4011}`

### Requirement: 解析 Qwen 转写结果
系统 MUST 解析转写链路返回的原始文本，剥离 `language <语种><asr_text>` 前缀，取出转写文本与识别语种；返回文本为空或语种为空的段 MUST NOT 产出响应分段。进程内推理的原始输出 MUST 保持与外部服务一致的前缀格式，既有解析契约 MUST NOT 失效。

#### Scenario: 返回带语种前缀的文本
- **WHEN** 转写链路返回 `language Chinese<asr_text>识别文本`
- **THEN** 系统产出文本为 `识别文本`、语种为 `Chinese` 的分段

#### Scenario: 段内无有效语音
- **WHEN** 转写链路返回 `language None<asr_text>`
- **THEN** 该段不产出响应分段

#### Scenario: 服务返回多段拼接文本
- **WHEN** 转写链路返回包含多个 `language <语种><asr_text>` 片段的文本
- **THEN** 系统拼接各片段文本并取第一个可识别的语种

#### Scenario: Qwen 服务调用失败
- **WHEN** 转写链路不可用、超时或返回错误状态
- **THEN** 系统按配置的失败策略重试，重试耗尽后返回 `{"msg": "Qwen 服务调用失败，请稍后再试", "code": 4012}`

### Requirement: 按规则透传语言参数
系统 MUST 保留 `/v3/seacraft_asr` 的 `language` 请求参数（默认 `auto`），但当取值为 `auto`、空串或未提交时 MUST NOT 指定具体语种，由模型自动识别；提交具体语种时 MUST 传入处理器，处理器接受语种全名或 ISO 简写。系统 MUST 锁定并记录处理器实现与版本；当前目标环境的 system hint 行为 MUST 通过已安装 5.13.0 代码及实机语种用例验证，MUST NOT 将 latest 文档中的 assistant forced-language 行为当作本机事实。响应的 `language` MUST 取模型实际识别出的语种，而非请求参数回显。语种不受支持时 MUST 返回 `4009`，MUST NOT 降级为自动识别。

#### Scenario: 未提交语言参数
- **WHEN** 客户端未提交 `language`，或提交 `auto`
- **THEN** 系统不指定具体语种，由模型自动识别语种，响应 `language` 为识别结果

#### Scenario: 提交具体语种
- **WHEN** 客户端提交 `language=en`
- **THEN** 系统把 `en` 传入处理器进行识别

#### Scenario: Qwen 服务不支持该语种
- **WHEN** 提交的语种不在模型支持的清单内
- **THEN** 系统返回 `{"msg": "不支持的语言", "code": 4009}`

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时执行的转写推理任务数量（`segment_concurrency`，由实测标定），并 MUST 限制所有音频合计同时执行的转写推理任务数量（`request_concurrency`，由实测标定），超出部分 MUST 等待可用名额后再执行。两个交付默认值 MUST 与本地推理的实测承载能力匹配，并记录依据，MUST NOT 将历史 16/128 或当前配置快照视为已验收档位。物理执行可在同一实例内采用已验证的微批，但每个逻辑段任务仍分别计入对应并发闸门；增加信号量本身 MUST NOT 被描述为吞吐扩容。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出的有效段数超过配置的 segment_concurrency
- **THEN** 系统按配置的 `segment_concurrency` 执行逻辑段任务，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计执行的转写推理并发数不超过 配置的 `request_concurrency`

### Requirement: 通过配置外置运行参数
Qwen 适配链路的本地模型目录、设备标识、请求并发度、总音频并发度、增强音频并发度、队列容量、超时和 VAD 参数 MUST 来自配置文件；VAD 参数 MUST 只从 `[vad]` 读取，VAD 实例数 MUST 只由 `[vad].vad_device` 数组长度决定，动态 VAD 池参数 MUST 不再作为有效配置。

#### Scenario: 修改并发参数后重启生效
- **WHEN** 运维把 `[vad].vad_device` 改为经验证的设备数组、`[qwen_asr].segment_concurrency` 改为有标定依据的值、`[qwen_asr].audio_concurrency` 改为 8、`[qwen_asr].audio_spk_emo_concurrency` 改为 4 并重启服务
- **THEN** 服务按设备数组长度启动固定 VAD worker，并按新的转写、总音频和增强音频并发参数运行

#### Scenario: VAD 旧配置不作为回退
- **WHEN** 配置只在 `[qwen_asr]` 中提供 `vad_device` 或其它 VAD 参数，而 `[vad]` 未提供对应配置
- **THEN** 系统不读取旧位置的 VAD 值，使用 `[vad]` 的默认值或返回明确的配置校验错误，并记录迁移提示

#### Scenario: 配置缺省时使用默认值
- **WHEN** 配置文件缺少相关新配置项
- **THEN** 服务使用配置加载器与交付模板一致的默认值；VAD 设备配比及段级/全局并发须有实机依据，不能静默恢复 NVIDIA 时期固定 16/128，默认满额立即拒绝策略保持不变

### Requirement: 配置 v2 链路模型的落卡
该历史 Requirement 名称中的 v2 仅为能力标识，本次仅作用于现行 `/v3/seacraft_asr` 链路，不恢复 v2 路由。系统 MUST 通过 `[model_device_pool]` 下的 `speech_campplus_devices` 与 `emotion2vec_devices` 数组配置 CAM++ 和 emotion2vec 的实例设备；数组元素数量分别决定两个模型池的配置容量，旧 `[model_device]` 单设备字段和 `[model_device.local_model_pool]` 结构 MUST 不再生效。设备元素 MUST 支持 `cpu` 与 `npu:N`。

#### Scenario: 修改落卡配置
- **WHEN** `speech_campplus_devices=["npu:0","npu:0","npu:0"]` 且 `emotion2vec_devices=["npu:0"]`
- **THEN** 服务按 3 个 CAM++ 实例和 1 个 emotion2vec 实例建立独立模型池，两个池互不借用实例

#### Scenario: 配置缺省
- **WHEN** 任一模型设备数组为空且请求对应的 `showSpk` 或 `showEmotion` 为 true
- **THEN** 请求返回该模型明确的未启用或无可用实例错误，不伪造标签；未请求该能力时不加载该模型

#### Scenario: 旧单设备字段残留
- **WHEN** 配置同时包含旧 `[model_device]` 字段和新的 `[model_device_pool]` 数组
- **THEN** 服务只使用新数组，并记录配置迁移提示

### Requirement: 每个 VAD 段通过内存音频请求 Qwen
系统 MUST 保持一个 VAD 段对应一个独立转写任务，但 MUST 直接使用该段标准化后的内存音频数据；段请求 MUST NOT 依赖为转写反复创建、打开和删除临时 WAV 文件。音频内容必须保持 VAD 起止样本、16 kHz、单声道和 PCM 内容不变。

#### Scenario: 正常段请求
- **WHEN** VAD 产出一个有效段
- **THEN** 转写链路收到一个独立的 16 kHz 单声道 PCM 段，响应分段时间轴仍使用该 VAD 段的 `bg` 和 `ed`

#### Scenario: 多段音频
- **WHEN** 一个完整音频包含多个 VAD 段
- **THEN** 每个段建立独立逻辑转写任务，不拼接相邻段；允许经验证的微批物理执行，但各段输入、输出和时间轴独立，逻辑段并发仍受 `segment_concurrency` 限制

#### Scenario: Qwen 请求失败
- **WHEN** 段转写超时、模型不可用或推理返回错误
- **THEN** 按既有重试和错误码契约处理，不能因为取消临时 clip 清理而泄漏文件或改变其它段的结果

### Requirement: VAD worker 数量由设备数组固定决定
系统 MUST 在启动时按 `[vad].vad_device` 数组元素顺序以 spawn 预热固定数量的 FSMN-VAD worker；数组长度决定配置容量，实际健康 worker 决定可用容量。worker MUST 按 FIFO 接收完整音频，每个 worker 同时只处理一个完整音频，运行期 MUST NOT 动态扩容或回收。合法设备为 `cpu` 与 `npu:N`，旧 `cuda:N` MUST 在就绪前给出迁移诊断，MUST NOT 隐式回退。该能力服务于现行v3，不恢复v2路由。下列历史Scenario标题保留作稳定标识，实际设备与行为以现行CPU/NPU正文为准，不恢复CUDA。

#### Scenario: CPU 与 GPU 混合 VAD 配置
- **WHEN** 宿主使用 `[vad].vad_device=["cpu","cpu","npu:7","npu:7"]`
- **THEN** 启动四个固定worker，其中两个CPU、两个物理7号卡NPU，健康时最多同时处理四个完整音频；仅暴露物理7号卡的容器按可见映射使用逻辑 `npu:0`

#### Scenario: 超过 VAD worker 数量的请求
- **WHEN** 同时提交的完整音频数超过健康VAD worker容量
- **THEN** 超出部分按既有音频准入和固定池FIFO排队规则等待或返回容量错误，不创建临时worker，不改变A/B分类或名额优先级

#### Scenario: VAD 数组元素无效
- **WHEN** `[vad].vad_device` 含空字符串、未知类型、`cuda:N` 或格式不是 `cpu`/`npu:N` 的元素
- **THEN** 服务就绪前的配置校验失败并指出具体数组下标与迁移提示，不以隐式CPU回退

### Requirement: 本地后端切换保持音频准入和取消契约
系统 MUST 在将CAM++/emotion2vec迁移为原生torch_npu后保留既有A/B分类、总音频/B上限、段/Qwen并发、能力开关、错误优先级和缓存清理规则。设备为NPU时实际走torch_npu，CPU时走PyTorch CPU，不以已退役TensorRT开关、engine存在或设备数组绕过功能校验。请求取消 MUST 等待已启动线程、NPU实际工作、结果读取/必要同步及清理完成后再准确归还模型实例和整音频租约；异步提交或线程返回不代表设备完成，重复取消不得遗弃收尾。下列历史Scenario标题仅为兼容的规格标识，实际WHEN/THEN均采用现行原生后端，不恢复CUDA/TensorRT执行。

#### Scenario: B 上限不因后端切换而变化
- **WHEN** 总音频上限8、B上限4，已有3A+4B，增强模型由CPU参考路径迁移到torch_npu
- **THEN** 新B仍返回4017，新A仍可获准，实例容量仍由各模型设备数组和健康状态决定；总容量也满时仍按既有4013优先级返回

#### Scenario: 普通转写不触发增强模型
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify`均为false，即使NPU增强设备数组非空或模型已加载
- **THEN** 请求仍是A，不加载、租用或调用CAM++/emotion2vec，不要求任何engine；本地Qwen转写按自身契约处理

#### Scenario: 启用 TensorRT 后请求关闭能力
- **WHEN** 请求需要的增强功能被features关闭，即使对应NPU实例曾加载或设备数组非空
- **THEN** 仍在音频准入前按既有单项/组合4015、4016、4018错误返回，不因后端选择绕过校验或伪造标签

#### Scenario: CUDA 工作尚未完成时取消
- **WHEN** 已准入B请求被取消或重复取消，而某NPU实例的设备工作或结果读取尚未完成
- **THEN** 原任务继续持有该模型和整音频租约，等待实际设备工作、同步和必要清理后才归还；不提前复用可写缓冲，最终计数一致且不重复释放

#### Scenario: engine 初始化失败
- **WHEN** 已准入请求所需增强模型的全部实例权重加载或目标设备初始化失败
- **THEN** 返回对应模型不可用4015或4016，完成请求清理和音频租约释放；不自动改走CPU、CUDA或组合能力关闭错误

### Requirement: 全局 FunASR 参数与段级参数职责分离
系统 MUST 区分 `[vad]` 配置声明、nested `vad_kwargs` 传递、实际模型 `model_conf`/选项及来源、`core.vad` 段级后过滤四层证据，不凭声明或同值匹配声称覆盖构造。迁移 MUST 保留当前模型与调用方式的实际行为、兼容配置键和独立段级过滤，不修改算法/阈值，不把本轮有效值永久规定为其它模型默认；详细保留约束与后续变更边界见下列场景。

#### Scenario: 全局阈值保持原有语义
- **WHEN** 声明的 `vad_decibel_thres=-100.0` 或 `vad_snr_thres=-100.0` 沿既有nested位置传递，或核对到模型实际选项为-100
- **THEN** 记录声明、转发形态与模型实际采用值及来源，不从传入或模型内-100推断段级过滤开关状态；段级仍按服务实际读取的门限、enable与既有-100禁用边界处理，不改写布尔开关或新增全局覆盖

#### Scenario: 段级 decibel 使用独立门限
- **WHEN** `vad_segment_decibel_thres=-40.0` 且段级decibel开关为true、该段处于既有过滤范围
- **THEN** 服务按段内帧dBFS与-40比较并结合低能量帧比例规则；此段级门限不被当作FunASR模型实际log能量选项，也不以模型内decibel=-100取消这条独立过滤。`vad_decibel_thres`的既有声明、nested及兼容getter仍保留，不因本场景误判为废弃键；本场景不建立global decibel到segment门限的隐式绑定

#### Scenario: 段级 SNR 与 RMS 使用显式开关
- **WHEN** `vad_snr_thres=-18.0` 或 `vad_low_volume_rms_dbfs_thres=-40.0` 且对应enable为true
- **THEN** 系统按现有段级 `snr_db`/`rms_dbfs` 口径执行过滤，阈值等于或低于-100时对应规则不丢弃段；本轮服务SNR18与段级能量/RMS门限-40的实际生效层次单独记录，不能由nested SNR18未覆盖模型内-100推断其在服务中无效

#### Scenario: 本轮模型实际选项与nested声明不同
- **WHEN** 核验本轮固定FSMN权重与当前FunASR/原生构造链，nested声明为maxsingle12000、end_silence800、speech_noise0.8、decibel-40、snr18
- **THEN** 记录原CPU实例实读及native构造链核对得到的model_conf选项分别为8000、800、0.6、-100、-100及其模型YAML来源；明确nested未覆盖这些构造选项，800同值也不单独构成覆盖证明，同时记录服务段级SNR18、独立decibel/RMS门限和开关的实际读取。不得把本轮结果写成新模型必须永久采用这些值，或冒充未执行的NPU实例选项实读

#### Scenario: 后续调整模型或参数绑定
- **WHEN** 后续更换权重/模型版本、升级FunASR、调整nested与顶层绑定或希望使声明值实际覆盖构造
- **THEN** 作为独立行为变更先复核有效选项来源、分段与过滤参考并执行适用验收，更新相应规格与报告；本次NVIDIA到NPU迁移不借此更改算法、模型默认值、段上限或现有过滤阈值

#### Scenario: 四层证据与兼容配置完整保留
- **WHEN** 审查配置传递、模型有效选项或服务段级过滤
- **THEN** 下列来源、兼容读取、门限和开关约束均须保持，不以重新排版删除原约束。

系统 MUST 区分四层证据：`[vad]` 的配置声明、传给调用链的 nested `vad_kwargs`、模型构造后实际采用的 `model_conf`/选项及来源、服务 `core.vad` 的段级后过滤。`vad_decibel_thres`、`vad_snr_thres` 等声明 MUST 保留既有nested传递与兼容读取形态，MUST NOT 仅凭配置文字、字典中存在或同值匹配就宣称已覆盖FunASR模型的全局帧级参数。模型全局log能量/状态机选项与段级dBFS/SNR统计 MUST 分别记录单位和实际读取点，不能混称或互相替代。

迁移 MUST 保留当前原模型与调用方式的实际行为，不把nested参数顺手提升为模型构造顶层覆盖，也不因未覆盖构造就删除仍承担声明、nested传递或兼容getter职责的配置键。`vad_segment_decibel_thres`有独立配置默认值，MUST NOT 假定其缺省取 `vad_decibel_thres`；`core.vad_volume`已有的low_energy门限到RMS门限的统计回退按原调用参数保留，不能误称为global decibel到segment配置的回退。段级过滤继续按 `vad_segment_decibel_thres`、低能量帧比例、服务读取的 `vad_snr_thres`、RMS dBFS及各自enable/禁用边界执行；模型内某选项为-100不证明服务后过滤已关闭。验收记录 MUST 关联模型/配置/代码版本并区分实例实读与构造链推导，不把本轮模型的实际值永久规定为其它模型的默认值。

### Requirement: 提供 Qwen 版转写接口
系统 MUST 提供 POST `/v3/seacraft_asr`，保留既有 multipart 字段、默认值、响应序列化、错误码及计时契约；历史接口仅作为字段演进参考。`/v2/seacraft_asr`、`/v1.1.7/seacraft_asr` 与 `/v1.1.8/seacraft_asr` 已下线，MUST 返回未找到路由，MUST NOT 为迁移恢复 Paraformer/Whisper 分流或旧接口。下列历史 Scenario 标题仅作稳定标识，实际请求路径和行为以当前正文为准。

#### Scenario: 提交音频到 v2 接口
- **WHEN** 客户端向 `/v3/seacraft_asr` 上传有效音频
- **THEN** 系统按现行v3合同返回成功响应，顶层包含 language、language_ratio、segments、text、speed_info、load_audio_time_ms、gpu_time_ms，保留既有字段类型与可选字段规则

#### Scenario: 未提交可选参数
- **WHEN** 客户端只提交音频文件
- **THEN** 系统按 language=auto、wordTimestamps=false、showSpk=false、showEmotion=false、showRoleIdentify=false 处理

#### Scenario: v1.1.8 行为保持不变
- **WHEN** 客户端向 `/v1.1.8/seacraft_asr`、`/v1.1.7/seacraft_asr` 或 `/v2/seacraft_asr` 提交请求
- **THEN** 系统返回 HTTP 404，不注册旧路由、不加载其专用模型；既有v3字段与业务行为保持不变

### Requirement: 返回语种占比字典
系统 MUST 在 `/v3/seacraft_asr` 的响应顶层返回 `language_ratio` 字典，键为模型识别出的语种的**简写标识**（例如中文 `zh`、英文 `en`、粤语 `yue`），值为该语种文本字符数占总字符数的百分比（保留一位小数，各值之和为 100.0）；无有效文本时返回空字典。语种判定来源与判定逻辑 MUST NOT 改变，仅对键名做简写映射；无法映射的语种 MUST 原样透传其原始名称。

#### Scenario: 单语种音频
- **WHEN** 音频全部识别为中文
- **THEN** 响应 `language_ratio` 为 `{"zh": 100.0}`

#### Scenario: 中英混合音频
- **WHEN** 音频中部分段落识别为英文、其余为中文
- **THEN** 响应 `language_ratio` 同时包含 `zh` 与 `en` 两个键，值为各自文本字符占比

#### Scenario: 无法映射的语种
- **WHEN** 模型识别出简写表中不存在的语种
- **THEN** 响应 `language_ratio` 中该语种以模型返回的原始名称作为键

#### Scenario: 无有效文本
- **WHEN** 没有任何有效转写文本
- **THEN** 响应 `language_ratio` 为空字典

### Requirement: 保留扩展字段
`/v3/seacraft_asr` 的每个成功响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组。`role`、`emotion` 在对应能力未被请求或已启用模型的段级推断失败时为 `null`；请求要求的能力被配置关闭时 MUST 返回请求级错误，不得作为成功转写返回 `null`。

#### Scenario: 返回扩展字段
- **WHEN** 成功响应包含分段
- **THEN** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify` 均未提供或均为 `false`
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`

### Requirement: 规范化输出分段文本的标点
系统 MUST 在文本规范化之后对 `/v3/seacraft_asr` 的分段文本做标点卫生处理：当一个分段的文本去除空白后只剩标点符号时，该分段 MUST 从响应中移除；当一个分段的文本首个字符是标点符号时，MUST 移除该开头标点。该规则 MUST 独立于文本规范化开关生效，即规范化关闭时同样执行。

#### Scenario: 分段只有标点
- **WHEN** 某个分段的文本只有标点符号（如 `。`、`！`）
- **THEN** 该分段不出现在响应的 `segments` 数组中

#### Scenario: 分段以标点开头
- **WHEN** 某个分段的文本以标点符号开头（如 `，就。`）
- **THEN** 响应中该分段的文本不再以标点开头

#### Scenario: 规范化关闭时规则仍生效
- **WHEN** 文本规范化被关闭
- **THEN** 上述两条标点处理规则仍然作用于响应分段

### Requirement: VAD 配置删除动态池参数
系统 MUST 以 `[vad].vad_device` 数组作为 VAD 实例配置来源，并 MUST 不再读取或要求 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`、`vad_call_timeout_s`；VAD 排队由音频并发准入和固定 worker FIFO 调度承担。

#### Scenario: 旧动态池字段残留
- **WHEN** 配置文件仍包含已删除的动态池字段，同时提供合法的 `[vad].vad_device` 数组
- **THEN** 服务不使用这些字段改变 worker 数量或扩缩容行为，并记录迁移提示

#### Scenario: 固定池启动失败
- **WHEN** `[vad].vad_device` 为空或没有任何 worker 成功预热
- **THEN** 服务启动失败并返回明确配置错误，不能以没有 VAD worker 的状态接收 v3 请求

### Requirement: 严格清理服务自有缓存文件
系统 MUST 登记每个请求创建的源音频临时文件、兼容路径下的段缓存和其它服务自有缓存，并 MUST 在成功、VAD 异常、裁剪异常、Qwen/本地模型失败、请求取消和超时的所有退出路径执行幂等清理；清理范围 MUST 仅限服务可识别的自有路径。

#### Scenario: 请求成功完成
- **WHEN** v3 请求成功返回或返回无有效人声
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

### Requirement: 记录本地模型池排队对端到端耗时的影响

系统 MUST 为 v3 请求记录 CAM++ 和 emotion2vec 的调用次数、推理耗时和模型池等待耗时，并 MUST 能在测试报告中分别比较本地模型等待、VAD、Qwen 分段处理和其它后处理阶段。

#### Scenario: 8 路完整音频压测
- **WHEN** 对本轮八条完整SHA256去重WAV按4B+4A执行同条件g1/g2混合批次
- **THEN** 结果包含总/B实际活动峰值、两个模型池各自的busy/waiting与调用/跳过/失败情况、等待耗时、客户端端到端P50/P95、错误数和音频吞吐，A请求不触发未请求增强能力

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量而不新增槽位配置
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

### Requirement: 以 VAD 段作为转写与时间轴单元
系统 MUST 在服务本地对音频做 VAD 切分，并以「一个 VAD 段对应一个独立本地 Qwen 逻辑任务、至多一个有效响应分段」组织结果，`bg` 与 `ed` MUST 取原 VAD 边界。系统 MUST NOT 合并相邻段或通过落盘、拼接、补零改变段音频；本轮原生 eager 单段执行，经验证的候选批执行亦须保留独立归属和无有效文本的过滤规则。

#### Scenario: 音频被切分为多段
- **WHEN** 音频包含多段人声
- **THEN** 系统按原VAD顺序为每段建立独立本地Qwen任务，响应起止时间与对应段边界一致

#### Scenario: 音频无人声
- **WHEN** VAD 未检测到任何有效人声
- **THEN** 系统返回 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`

#### Scenario: 过滤过短语音段
- **WHEN** 某VAD段时长小于配置的最小语音段时长
- **THEN** 系统不执行该段的本地Qwen任务，且该段不产出响应分段

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 使用现有 CAM++ 与 emotion2vec 模型按请求能力产生角色和情绪，与同段 Qwen 任务并行发起。各模型按独立设备数组的健康实例提供容量，不增加共享模型槽位；总/B准入不改变模型池容量。CPU 使用 PyTorch CPU，`npu:N` 使用原生 torch_npu，CAM++ 与 emotion2vec MUST 保持已验证的 FP32 参数、浮点缓冲和计算策略，MUST NOT 沿用旧 CUDA FP16 或 TensorRT engine 精度配置。能力开关、段级失败降级和请求级错误优先级 MUST 保留。下列旧 Scenario 标题仅为稳定标识。

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
- **WHEN** emotion2vec 在CPU或NPU原生后端加载和推理
- **THEN** 参数与浮点缓冲均按已验证FP32策略构造、加载及放置，实际状态仅在核验成功后报告fp32；加载不得受并发Qwen HF临时BF16默认dtype污染，未使用的engine配置不决定精度

#### Scenario: 情绪 TensorRT 精度
- **WHEN** 旧部署或配置仍提及TensorRT精度
- **THEN** 该配置不触发engine加载或FP16切换，迁移提示与实际torch_npu/pytorch_cpu及FP32状态可追溯，原业务情绪映射不变

#### Scenario: 两个模型使用不同后端
- **WHEN** CAM++ 与emotion2vec分别采用经配置的CPU或NPU实例，或采用非对称实例数量
- **THEN** 增强任务仍与同段Qwen任务并行发起，各自按所属模型池健康容量执行，不借用其它池槽位、不隐式回退、不改变业务映射

### Requirement: 验证效果与性能边界
系统 MUST 使用 `test/音频文件/` 实际可用且可追溯的音频完成本轮NPU效果与性能验证。报告 MUST 区分单路、混合并发、冷启动、稳态与候选实验，覆盖完整音频边界、幻觉/复读观测、NPU资源与客户端端到端时延；本轮正式混合验收按八条完整SHA256去重WAV的4B+4A及g1/g2条件执行。历史1/2/4/8逐档专项与当前验收范围 MUST 分别列明，未执行档位不得冒称完成；无人工真值时不得输出绝对准确率、WER或时间戳准确率。

#### Scenario: 完成并发压测
- **WHEN** 按本轮测试计划执行单路、同源暖标定或八条独立完整音频混合批次
- **THEN** 分别记录样本、并发/实际活动峰值、g1/g2、加载是否计入、NPU与host资源、端到端耗时、错误、语种和质量观测；同源重复暖标定不冒充八条独立文件，未执行的历史并发专项列为未执行/不适用及原因

#### Scenario: 产出测试报告
- **WHEN** 交付本轮验证结果
- **THEN** 仓库保留可复算中文报告与原始响应/日志/清单，说明文本一致性、分段/时间边界、幻觉/复读样例及证据限制；没有人工标注或可追溯旧设备基准时明确未知，不将短片一致或无错误宣称绝对准确率或最优性能

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 在本轮可用音频上验证role/emotion及其模型数值和业务映射，测量时延、吞吐、模型池等待与明确定义口径的NPU/host资源增量。参考优先使用可追溯同权重CUDA golden；缺少时使用CPU FP32参考并明确限制，MUST NOT 恢复已下线接口来制造基准。正式八路验收 MUST 使用同一清单的八条完整去重WAV、4B+4A、NPU原生三CAM++/一emotion2vec与实际健康容量，按相同条件比较g1/g2；不把旧CUDA6:2组合作为本轮默认。

#### Scenario: 角色与情绪效果比对
- **WHEN** 使用同段PCM、原始权重、特征与映射规则比较CPU FP32参考和NPU原生输出
- **THEN** 保留CAM++192维embedding、emotion2vec完整分类及误差/容差证据，再核对角色聚类与业务标签；记录不一致样本，无CUDA golden或人工角色/情绪标注时明确未验证相应准确率，不调用已404的v2/v1.1.8充作参考

#### Scenario: 8 路非对称模型池压测
- **WHEN** 宿主npu:7使用三CAM++/一emotion2vec、单Qwen和固定VAD池，对八条完整去重WAV执行4B+4A并比较g1=1/1与g2=2/2
- **THEN** 记录实际健康实例、总/B峰值、各池busy/waiting、逐请求客户端耗时及线性插值P50/P95、吞吐、错误、NPU/host与共享卡干扰；不同容量另列，不与本轮条件混算，按服务八路报告契约保留原始证据并关闭owned资源

#### Scenario: 并行开销可接受
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **THEN** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **WHEN** 在已记录的非对称模型实例组合下运行所请求增强能力
- **THEN** 记录NPU利用率与HBM、当前服务进程allocator、RSS/PSS等可观测值及其范围，未知标明原因；加载窗口delta不能相加当逐模型独占量。记录吞吐（音频/分钟、段/秒）和两个模型池等待，不以不存在的CUDA SM指标充作NPU数据，写入中文报告

### Requirement: 完成固定配置的全量 WAV 质量压测
系统 MUST 支持按显式清单对 `test/音频文件/` 的完整WAV执行 `/v3/seacraft_asr` 质量与性能验证，保存每请求响应、错误、耗时和过滤/复读统计。本轮完整批次 MUST 为八条按完整源SHA256去重的WAV、4B+4A、总8/B4、Qwen单副本、三CAM++/一emo，分别执行g1=1/1与g2=2/2对照，按完整实测和质量/资源约束选择并记录最终默认；VAD、RMS及其它阈值保持迁移前实际行为。MUST NOT 自动套用历史RMS-41、segment4/audio16或要求恢复旧v2及其45文件配对资产。历史专项与本轮清单分别记录，不将八文件结果冒称旧专项已执行。

#### Scenario: 45 个 WAV 全部配对并执行
- **WHEN** 准备本轮实际八条完整WAV清单，或另行提供历史45文件及其Paraformer配对参考
- **THEN** 本轮验证先核八文件ID、路径编码、大小/完整时长/SHA256及实际配置，按固定清单完成4B+4A；历史标题保留作稳定标识，不以缺少旧45配对阻塞本轮迁移，也不称其已完成。若另行执行历史参考专项则逐一核对配对，缺失/多余明确失败，不补造旧接口或基线

#### Scenario: 记录端到端分位数
- **WHEN** 完整清单请求全部完成或任一失败
- **THEN** 报告每档成功数/8、失败/超时与原始逐文件耗时；成功完整批次按既有八路契约计算客户端端到端P50/P95和最大值、RTF及吞吐，失败批次完整保留并标记不完整，不丢失败后冒称八路成功
