# qwen-asr-v2-adapter Specification

## Purpose

定义现行 `/v3/seacraft_asr` 中基于 Qwen3-ASR 的离线转写链路：本地 VAD 切分及时间轴映射、独立内存音频段的本地推理、语种与原始输出解析、并发排队和整音频准入、既有响应封装，以及说话人角色与情绪计算。能力名称保留历史 v2 标识，不恢复已下线的 v2 或 v1.1.8 路由；迁移保持既有错误、取消、缓存清理及段级后处理语义。

## Requirements

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
系统 MUST 解析并剥离`language <语种><asr_text>`前缀，保持内嵌与原链路格式一致；空文本或空语种不产出响应段。legacy保留原配置失败重试；continuous仅明确未接收的容量拒绝可从首次拒绝起30秒内最多重试3次，沿用同一段ID，超限按既有错误收尾。已接收或接收状态未知 MUST NOT 重发，不重试整音频；内容异常沿段警告/恢复规则，基础设施错误不得伪装文本成功。

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
- **THEN** legacy按配置失败策略重试，耗尽后保留`{"msg": "Qwen 服务调用失败，请稍后再试", "code": 4012}`；continuous仅对明确未接收的容量拒绝从首次拒绝起30秒内最多重试3次同段ID，已接收或接收状态未知不重发、不重试整音频，按既有4011/4012等适用错误安全收尾

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

### Requirement: 限制同时处理的音频数量
continuous模式总音频配置容量 MUST 由Qwen实例数N派生（最终默认4、合法1～8），新请求只可原子取得一个健康可租owner和音频槽位；legacy显式模式保留其配置上限。每请求 MUST 持有同一owner至处理及全部设备/临时资源安全收尾，成功、异常、取消均准确释放。默认不等待，总满4013优先；B另受增强子上限限制。健康下降仅限制新授予，MUST NOT 伪释放仍在收尾的租约。

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

#### Scenario: 默认四实例音频准入
- **WHEN** continuous配置N=4且四owner健康，已准入四条A/B音频
- **THEN** 新A或B按既有默认立即拒绝策略返回4013，不创建第五Qwen、不按文件名或SHA分流；现有请求的owner归属不变

#### Scenario: 任意文件与后续请求复用
- **WHEN** 两个同名或同内容音频并发提交，随后另一个新音频到达
- **THEN** 每个请求按request_id独立租健康owner并保持到安全归还；新请求可以复用已安全归还的owner，不要求测试源清单

### Requirement: 限制对 Qwen 服务的并发请求数
continuous中每owner及其所绑定音频的Qwen在途上限 MUST 为配置S（默认16），全局配置上限 MUST 由N×S派生，最终默认64；超出等待，不重复维护冲突的request_concurrency。名额覆盖准备、待执行、设备执行和安全收尾，段独立完成即归还，不等待整组或增强汇合。物理步不得超过该owner的S，增加信号量不代表吞吐扩容；legacy显式模式保留原参数契约。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出的有效段数超过配置的 segment_concurrency
- **THEN** 系统按配置的 `segment_concurrency` 执行逻辑段任务，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计执行的转写推理并发数不超过 continuous的派生N×S（legacy为其配置的`request_concurrency`）

#### Scenario: 短段结束及时补位
- **WHEN** 同一owner一个段已安全结束而其它段仍生成，且音频还有待处理段
- **THEN** 立即归还该段credit并在下个可用安全步接纳后续段，保留其它段独立KV；不等待16段一起结束、不跨owner迁移已执行段

### Requirement: 保留扩展字段
`/v3/seacraft_asr` 的每个成功响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组。`role`、`emotion` 在对应能力未被请求或已启用模型的段级推断失败时为 `null`；请求要求的能力被配置关闭时 MUST 返回请求级错误，不得作为成功转写返回 `null`。

#### Scenario: 返回扩展字段
- **WHEN** 成功响应包含分段
- **THEN** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify` 均未提供或均为 `false`
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`

### Requirement: 通过配置外置运行参数
Qwen本地目录、设备、实例数N、每owner段上限S、B上限、队列、超时及VAD参数 MUST 来自配置。continuous总音频N/全局N×S MUST 派生；旧audio_concurrency/request_concurrency显式相等可兼容并提示，冲突或非法值须启动前报迁移错误，根与Docker模板不重复维护派生值。VAD参数 MUST 只从`[vad]`读取，VAD实例数仅由`[vad].vad_device`数组长度决定，旧动态池字段不再有效；显式legacy保留其原参数行为。

#### Scenario: 修改并发参数后重启生效
- **WHEN** 运维在continuous中配置实例数N、经验证的段上限S、B上限4与合法`[vad].vad_device`，或显式选择legacy配置，并重启服务
- **THEN** 服务按VAD数组长度启动固定worker；continuous按N与N×S派生总音频/全局段窗口并展示B有效容量，legacy按其独立参数运行，配置修改需重启

#### Scenario: VAD 旧配置不作为回退
- **WHEN** 配置只在 `[qwen_asr]` 中提供 `vad_device` 或其它 VAD 参数，而 `[vad]` 未提供对应配置
- **THEN** 系统不读取旧位置的 VAD 值，使用 `[vad]` 的默认值或返回明确的配置校验错误，并记录迁移提示

#### Scenario: 配置缺省时使用默认值
- **WHEN** 配置文件缺少相关新配置项
- **THEN** 服务使用配置加载器与交付模板一致的默认值；VAD 设备配比及段级/全局并发须有实机依据，不能静默恢复 NVIDIA 时期固定 16/128，默认满额立即拒绝策略保持不变

#### Scenario: 旧派生字段冲突
- **WHEN** continuous配置N=4、S=16但旧audio_concurrency或request_concurrency显式值与4或64冲突
- **THEN** 服务启动前明确指出冲突字段、派生值和迁移办法，不静默忽略或启动成另一容量；相等值仅兼容读取并提示迁移

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

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 在VAD过滤、段credit与PCM预算取得及有效段裁剪后，并发发起同段Qwen、所请求CAM++及emo任务；各池独立等待，不把增强串成Qwen完成后调用，不承诺同卡kernel物理同时。CAM/emo按独立设备数组的健康容量执行，不随N复制或互借；CPU用PyTorch、NPU用torch_npu，保持已验证FP32参数/缓冲/计算，不使用旧FP16或TensorRT。设备数组派生启用状态、段级失败降级、错误优先级、现有聚类/ITN及业务映射 MUST 保留。

#### Scenario: 开启角色与情绪能力
- **WHEN** 两个设备数组均非空，请求 `showSpk=true`、`showEmotion=true` 且模型推断成功
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行并分别使用对应模型池

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk`、`showEmotion`、`showRoleIdentify`（均默认 `false`）
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`，两个模型池均不因该请求加载

#### Scenario: 配置未开启说话人能力
- **WHEN** CAM++设备数组为空 且请求需要说话人能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4015`，不开始该请求的模型加载或转写；未请求说话人能力的请求不因另一能力关闭报错

#### Scenario: 配置未开启情绪能力
- **WHEN** emotion2vec设备数组为空 且请求需要情绪能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4016`，不开始该请求的模型加载或转写；未请求情绪能力的请求不因另一能力关闭报错

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** CAM++与emotion2vec设备数组中至少一项为空
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

#### Scenario: 增强时长边界
- **WHEN** 有效段小于0.3秒，或保留段处于0.3～不足1秒、1～10秒含边界、超过10秒
- **THEN** 小于0.3秒三路均不执行；保留且不少于0.3秒的段可做CAM，emo仅1～10秒真实推理，区间外按原规则返回平淡；skip单列且不计NN推理成功

#### Scenario: Qwen先完成而增强仍在途
- **WHEN** 同段Qwen已达到真实安全终态，CAM或emo仍等待或执行
- **THEN** Qwen段credit先归还使后续段可补位；该段PCM仍由请求持有至三路均安全终态，取消不能提前释放PCM或正在执行的模型实例

### Requirement: 验证效果与性能边界
系统 MUST 使用显式清单、完整可追溯音频保存效果与端到端性能证据。下列八文件4B+4A/g1g2场景仅为显式选择的历史legacy专项，不是新continuous缺省验收，未执行不得称完成。正式continuous按最终N4/CAM4/emo1的B4及增强驻留A4要求验证，N6 A6/A4/B4与共驻失败保留历史，两类矩阵不得替代。报告须区分冷态/稳态/候选、过滤/复读/NPU资源及客户端时延；无人工真值不得输出绝对准确率、WER或时间戳准确率。

#### Scenario: 完成并发压测
- **WHEN** 显式选择历史legacy测试计划执行单路、同源暖标定或八条独立完整音频混合批次
- **THEN** 分别记录样本、并发/实际活动峰值、g1/g2、加载是否计入、NPU与host资源、端到端耗时、错误、语种和质量观测；同源重复暖标定不冒充八条独立文件，未执行的历史并发专项列为未执行/不适用及原因

#### Scenario: 产出测试报告
- **WHEN** 交付本轮验证结果
- **THEN** 仓库保留可复算中文报告与原始响应/日志/清单，说明文本一致性、分段/时间边界、幻觉/复读样例及证据限制；没有人工标注或可追溯旧设备基准时明确未知，不将短片一致或无错误宣称绝对准确率或最优性能

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 验证CAM/emo数值、业务标签、聚类和段结构，记录端到端延迟、吞吐、各池等待及资源范围；优先可追溯同权重CUDA golden，缺少则用CPU FP32并说明限制，不恢复旧接口。下列八路非对称池/g1g2与单路30%场景仅在显式历史legacy专项适用；正式continuous按独立B4要求标定，不以三CAM/一emo或旧6:2充当新默认，不把纯A约200秒目标承诺给B4。

#### Scenario: 角色与情绪效果比对
- **WHEN** 使用同段PCM、原始权重、特征与映射规则比较CPU FP32参考和NPU原生输出
- **THEN** 保留CAM++192维embedding、emotion2vec完整分类及误差/容差证据，再核对角色聚类与业务标签；记录不一致样本，无CUDA golden或人工角色/情绪标注时明确未验证相应准确率，不调用已404的v2/v1.1.8充作参考

#### Scenario: 8 路非对称模型池压测
- **WHEN** 显式选择历史legacy专项，在宿主npu:7使用三CAM++/一emotion2vec、单Qwen和固定VAD池，对八条完整去重WAV执行4B+4A并比较g1=1/1与g2=2/2
- **THEN** 记录实际健康实例、总/B峰值、各池busy/waiting、逐请求客户端耗时及线性插值P50/P95、吞吐、错误、NPU/host与共享卡干扰；不同容量另列，不与本轮条件混算，按服务八路报告契约保留原始证据并关闭owned资源

#### Scenario: 并行开销可接受
- **WHEN** 显式选择历史legacy单路开销专项，比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **THEN** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **WHEN** 在已记录的非对称模型实例组合下运行所请求增强能力
- **THEN** 记录NPU利用率与HBM、当前服务进程allocator、RSS/PSS等可观测值及其范围，未知标明原因；加载窗口delta不能相加当逐模型独占量。记录吞吐（音频/分钟、段/秒）和两个模型池等待，不以不存在的CUDA SM指标充作NPU数据，写入中文报告

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

### Requirement: 提供文本规范化开关
系统 MUST 支持通过配置控制文本规范化的启用与行为：启用开关默认开启（`enabled` 为 true），并支持独立控制独立数字转换（`enable_standalone_number`）、数字逐位读出（`enable_0_to_9`）与语气词剔除（`remove_interjections`）。语气词剔除开关 MUST 可在配置中显式设置，默认行为 MUST 为剔除「呃」「啊」等语气词；剔除产生的标点副作用 MUST 由标点卫生规则清理。

#### Scenario: 默认开启文本规范化
- **WHEN** 部署配置未显式覆盖文本规范化开关
- **THEN** 系统对转写文本执行规范化，并按配置剔除语气词

#### Scenario: 显式保留语气词
- **WHEN** 部署配置把语气词剔除开关设为 false
- **THEN** 系统保留转写文本中的语气词

#### Scenario: 关闭文本规范化
- **WHEN** 部署配置关闭文本规范化
- **THEN** 系统直接返回模型原始转写文本，不做数字与标点规范化

#### Scenario: 规范化依赖不可用
- **WHEN** 文本规范化依赖缺失或初始化失败且失败策略为 `passthrough`
- **THEN** 请求仍成功返回，文本保持 ASR 原始输出

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

### Requirement: VAD 配置删除动态池参数
系统 MUST 以 `[vad].vad_device` 数组作为 VAD 实例配置来源，并 MUST 不再读取或要求 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s`、`vad_call_timeout_s`；VAD 排队由音频并发准入和固定 worker FIFO 调度承担。

#### Scenario: 旧动态池字段残留
- **WHEN** 配置文件仍包含已删除的动态池字段，同时提供合法的 `[vad].vad_device` 数组
- **THEN** 服务不使用这些字段改变 worker 数量或扩缩容行为，并记录迁移提示

#### Scenario: 固定池启动失败
- **WHEN** `[vad].vad_device` 为空或没有任何 worker 成功预热
- **THEN** 服务启动失败并返回明确配置错误，不能以没有 VAD worker 的状态接收 v3 请求

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
系统 MUST 在分段文本进入文本规范化之前检测复读，并对命中的分段把重复部分折叠为一次出现，保留前文。常规生成后检测 MUST 依据文本长度超过按段时长估算的合理上限且存在连续重复子串；长度合理的正常结束分段 MUST NOT 被修改。continuous在线保护已按既有持续确认和强循环条件判定的异常，MUST 在恢复阶段直接折叠已确认循环，不再次用常规绝对长度或恢复短句字数门槛将其丢弃；恢复仍带段警告，不能视为准确率认证。

#### Scenario: 复读长串被折叠
- **WHEN** 一个 6 秒的分段文本为 1863 个字符的「最小S零，这个这个这个这个，……」
- **THEN** 该段文本被折叠为长度正常的短文本，且以「最小S零」开头

#### Scenario: 正常长度的分段不受影响
- **WHEN** 一个正常结束且未被在线保护确认异常的分段文本长度未超过该段时长对应的合理上限
- **THEN** 该段文本原样保留，不做任何改写

#### Scenario: 长度正常但存在叠词的文本不受影响
- **WHEN** 一个分段文本为「温度载荷，温度载荷，温度载荷，」且长度未超限
- **THEN** 该段文本原样保留

#### Scenario: 在线强循环停止后的短句折叠
- **WHEN** continuous在线保护确认强循环而提前停止，正文尚未达到常规后处理的120字符下限
- **THEN** 恢复仍将已确认循环保留一次，并保留前文及段警告；未确认异常的正常短句和教学重复不改变

### Requirement: 长度异常但无复读结构时不得截断
系统 MUST 在分段文本长度超过合理上限、但未检出连续重复结构时保留原文并记录告警，MUST NOT 以截断或截取前 N 个字符的方式修改该文本。系统 MUST NOT 对未检出复读结构的文本做任何形式的内容裁剪。

#### Scenario: 超长但无复读结构的文本被保留
- **WHEN** 一个 5 秒的分段文本长度为 300 字符且其中不存在连续重复的子串
- **THEN** 该段文本原样返回，其长度与内容均未被修改

#### Scenario: 超长无复读结构时留下告警
- **WHEN** 出现上一场景的文本
- **THEN** 服务日志中记录一条告警，包含原长度、时长上限与该段时长

### Requirement: 复读判据的阈值可配置且有明确的标定依据
系统 MUST 支持通过配置调整复读检测的阈值，至少包括：是否启用修复的开关、按语音时长估算的字符数上限系数、以及重复子串的最小单元长度与最小重复次数。默认的字符数上限系数 MUST 为 40 字符/秒，其标定依据 MUST 为真实样本统计：正常段的高分位值约 20.3 字符/秒、观测到的最高值约 26.2，而复读段为 282~313。常规生成后长度异常判定的绝对字符下限 MUST 不低于120；在线强循环路径按其独立的持续确认条件判断，不受此常规门槛二次阻挡。

#### Scenario: 默认阈值下的判定边界
- **WHEN** 一个 10 秒的分段文本长度为 300 字符
- **THEN** 合理上限为 400 字符，该段不被判定为长度异常

#### Scenario: 关闭复读修复开关
- **WHEN** 配置把生成后复读修复开关repetition_guard_enabled设为关闭
- **THEN** 常规生成后路径不做复读检测与折叠，分段文本原样返回；独立启用的online_repetition_enabled仍按其条件停止异常生成并执行段级恢复及警告

#### Scenario: 调整字符数上限系数
- **WHEN** 运维人员把字符数上限系数从 40 调整为 60 并重启服务
- **THEN** 系统按新的系数估算合理上限并据此判定长度异常

### Requirement: 复读修复独立于文本规范化开关
系统 MUST 在文本规范化之前执行复读检测与修复，该规则 MUST 独立于文本规范化开关生效，即规范化关闭时同样执行。修复 MUST NOT 改变文本规范化与标点卫生规则的行为，两类处理 MUST NOT 互相覆盖。

#### Scenario: 文本规范化关闭时仍修复复读
- **WHEN** 配置关闭文本规范化且某个分段为复读长串
- **THEN** 该段仍被折叠为短文本

#### Scenario: 复读修复与标点卫生并存
- **WHEN** 一个分段既是复读长串又以标点开头
- **THEN** 先完成复读折叠，再由标点卫生规则处理其段首标点，两条规则都生效

### Requirement: 复读修复行为可观测
系统 MUST 在发生折叠时记录日志，日志 MUST 包含原文本长度、折叠后长度、按该段时长计算的长度上限与该段时长。系统 MUST NOT 把被折叠文本的完整内容写入日志，MUST 只记录其前缀。

#### Scenario: 折叠发生时留下日志
- **WHEN** 某个分段被折叠
- **THEN** 服务日志中出现一条包含原长度、折叠后长度、时长上限与时长的记录

#### Scenario: 未触发修复时不产生噪声日志
- **WHEN** 全部请求的分段文本均未触发复读修复
- **THEN** 不产生任何复读告警日志

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
系统 MUST 让每段CAM++和emotion2vec进入各自独立共享池，不把每个Qwen owner复制成一套增强池，也不让两个池互借实例。增强等待 MUST NOT 持有已经安全结束的Qwen段credit；全服务PCM仍在全部所需分支终态后释放，并允许正常有界反压。段的三路入口仍受Qwen准备窗口及PCM预算约束；这不意味着每池实际并发等于N×S。

#### Scenario: CAM++ 槽位少于 emotion2vec 槽位
- **WHEN** CAM++ 设备数组配置 3 个实例且 emotion2vec 设备数组配置 1 个实例
- **THEN** CAM++ 和 emotion2vec 分别按自身容量排队，空闲的 CAM++ 实例不能被 emotion2vec 请求占用

#### Scenario: 本地模型池等待期间 Qwen 仍可调度
- **WHEN** 某个 VAD 段等待 CAM++ 或 emotion2vec 实例
- **THEN** 其它获准段仍可推进Qwen，安全结束的Qwen credit可先归还；每owner S及派生N×S继续限制Qwen，全服务PCM约束仍生效

### Requirement: 记录本地模型池排队对端到端耗时的影响
系统 MUST 为v3记录CAM++/emo实际调用、推理、等待、时长skip与失败，分别比较VAD、裁剪、Qwen、各池等待、汇合、聚类和ITN。下列八路g1/g2为显式历史legacy专项，正式continuous使用最终N4固定A4/B4及增强驻留A4，不能互作验收。所有实际能力调用与请求开关关联，A不调用未请求增强；本次不改变聚类/ITN执行器或算法。

#### Scenario: 8 路完整音频压测
- **WHEN** 显式选择历史legacy专项，对八条完整SHA256去重WAV按4B+4A执行同条件g1/g2混合批次
- **THEN** 结果包含总/B实际活动峰值、两个模型池各自的busy/waiting与调用/跳过/失败情况、等待耗时、客户端端到端P50/P95、错误数和音频吞吐，A请求不触发未请求增强能力

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量而不新增槽位配置
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

#### Scenario: 正式四路双开增强对照
- **WHEN** 正式continuous在四份Qwen及最终CAM4/emo1共驻下，对同一a1～a4分别执行A4与showSpk/showEmotion均true的B4
- **THEN** 记录完整n4或删失子集口径、真实role/emotion字段与调用/skip覆盖、各池等待/推理、聚类/ITN及HBM；没有双开真实调用不能仅凭HTTP200宣称增强通过

### Requirement: 完成固定配置的全量 WAV 质量压测
系统 MUST 支持显式清单完整WAV的v3质量与性能验证并保存响应、错误、耗时、过滤和复读证据。本需求中八去重文件4B+4A、总8/B4、单Qwen、三CAM/一emo及g1/g2仅作用于显式历史legacy专项；新continuous使用最终N4/CAM4/emo1的B4及增强驻留A4矩阵，原N6/N5结果保留，不能勾选旧任务。VAD/RMS等阈值保持，不自动套用RMS-41或旧segment4/audio16，也不恢复v2或补造45文件配对。

#### Scenario: 45 个 WAV 全部配对并执行
- **WHEN** 显式选择历史legacy八条完整WAV专项，或另行提供历史45文件及其Paraformer配对参考
- **THEN** 本轮验证先核八文件ID、路径编码、大小/完整时长/SHA256及实际配置，按固定清单完成4B+4A；历史标题保留作稳定标识，不以缺少旧45配对阻塞本轮迁移，也不称其已完成。若另行执行历史参考专项则逐一核对配对，缺失/多余明确失败，不补造旧接口或基线

#### Scenario: 记录端到端分位数
- **WHEN** 显式历史legacy完整八文件清单请求全部完成或任一失败
- **THEN** 报告每档成功数/8、失败/超时与原始逐文件耗时；成功完整批次按既有八路契约计算客户端端到端P50/P95和最大值、RTF及吞吐，失败批次完整保留并标记不完整，不丢失败后冒称八路成功

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
系统 MUST 满足配置总容量内的A+B活动约束及配置B子上限；continuous总容量由N派生，新准入同时受实际健康可租owner数H限制，B配置默认4、有效新准入上限min(B配置,H)。A仅占总容量，B原子取得总/B名额及owner；等待或拒绝不预占。默认总满4013优先、仅B满4017且消息含spk_emo与有效使用比。H下降时既有收尾租约 MUST 保留并单列，不能为降低计数而伪释放。

#### Scenario: 三个 A 和四个 B
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4、默认不等待，正在处理 `3A+4B`
- **THEN** 新 B 返回 `4017` 和 `spk_emo队列已满，请稍后再试（当前增强音频并发：4/4）`；新 A 可获准形成 `4A+4B`

#### Scenario: 五个 A 和三个 B
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4、默认不等待，正在处理 `5A+3B`
- **THEN** 新 A 和新 B 均返回 `4013`，B 不因还有一个增强名额而获准

#### Scenario: 八个普通音频占满总容量
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4、默认不等待，正在处理 `8A`
- **THEN** 新 A 和新 B 均返回 `4013`，系统不为 B 预留总容量

#### Scenario: 总容量和增强容量同时已满
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4、默认不等待，正在处理 `4A+4B`
- **THEN** 新 A 和新 B 均返回 `4013`，总容量错误优先于增强容量错误

#### Scenario: 两类容量均有余量
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4，正在处理 `4A+3B` 且无等待请求
- **THEN** 下一请求无论为 A 或 B 均可获准，所有容量约束仍成立

#### Scenario: 增强容量满不占满总容量
- **WHEN** 显式选择N=8且owner健康的continuous或总上限8的legacy配置、增强上限为 4、默认不等待，正在处理 `0A+4B`
- **THEN** 新 B 返回 `4017`，新 A 可获准且增强活动数保持 4

#### Scenario: 显式六owner四B仍允许两个A
- **WHEN** 显式continuous N=6且六owner健康、B配置4，已准入0A+4B且默认不等待（原N6实测历史）
- **THEN** 第五B返回4017，另两个A可以获准形成2A+4B；此时总满，新A/B均4013，不因模型池仍有空实例绕过准入

#### Scenario: 健康容量下降
- **WHEN** 配置N=6/B4但可用于新授予的健康owner仅3，已有故障请求仍在安全收尾
- **THEN** 状态分列configured、healthy、leased/closing占用；B有效新准入容量不超过3，不迁移在途KV、不伪释放旧请求，健康资源实际可租后才授予新请求

#### Scenario: 最终默认四B使总容量同时满
- **WHEN** 最终continuous N=4/B4、四owner健康且已准入4B或任意总计4条A/B，默认不等待
- **THEN** 新A/B均4013，优先于B满4017；本配置不存在B满而总容量有余的状态，不能单独触发4017，也不删除显式N6/N8的4017契约及历史实测

### Requirement: 关闭能力在准入之前拒绝
系统 MUST 在音频准入、等待排队、音频预处理和模型调用之前校验请求所需能力的设备数组派生启用状态。仅所请求的说话人能力关闭返回 `4015`，仅所请求的情绪能力关闭返回 `4016`，同时请求两项且两项都关闭返回 `4018`。业务错误 MUST 保持 HTTP 200 和 `{"msg": ..., "code": ...}` 格式，不返回成功转写分段。

#### Scenario: 两种能力关闭但只做普通转写
- **WHEN** 两个增强设备数组均为空 且三个增强参数均为 `false`
- **THEN** 请求按 A 处理，只受总音频容量限制，增强活动数不增加

#### Scenario: 两种能力关闭且同时请求两种增强
- **WHEN** 两个增强设备数组均为空，请求需要说话人及情绪能力
- **THEN** 请求返回 `4018`，消息说明两种能力均未启用；即使总容量已满也不返回 `4013`，且所有活动、等待计数不变

#### Scenario: 同时请求两种能力但只有一种关闭
- **WHEN** 请求需要说话人及情绪能力，恰好只有一个增强设备数组为空
- **THEN** 说话人关闭返回 `4015`，情绪关闭返回 `4016`，请求整体失败且不进行部分转写

#### Scenario: 两种能力关闭但仅请求一种
- **WHEN** 两个增强设备数组均为空，但请求仅需要说话人能力或仅需要情绪能力
- **THEN** 分别返回 `4015` 或 `4016`，不使用组合错误 `4018`

### Requirement: 等待队列按可用容量调度
queue_wait_timeout_s>0时，系统 MUST 以queue_max_size限制A+B共享等待总数、保持同类FIFO并准入最早同时满足总/B/健康owner容量的请求；等待不预占owner。B仅因增强容量被阻塞而总容量有余时，A可绕过；较早B满足全部条件时，后A不得抢占。超时4010与总等待满4013保持，continuous由N派生容量，legacy保留显式容量。

#### Scenario: A 绕过被增强容量阻塞的 B
- **WHEN** 显式N=8且owner健康（或legacy总8）的`3A+4B`、增强上限 4，等待队列已有 B，此时新 A 到达
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
系统 MUST 在等待取消、授予/取消/超时竞争、处理取消和清理异常中保持总/B/owner计数一致。未授予只移除等待者；授予的总/B名额及同一request_id的owner租约至安全收尾后至多归还一次。设备未结束不得归还owner，段Qwen credit与三路PCM按各自安全终态释放；健康下降不以伪释放掩盖仍在收尾的占用。

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

#### Scenario: 同内容请求取消不影响另一owner
- **WHEN** 两个独立request_id提交同名或同内容音频，其中一个取消
- **THEN** 只收尾并释放该请求持有的owner及分支资源，另一请求的段、KV、总/B计数和临时文件不受影响

### Requirement: 本地后端切换保持音频准入和取消契约
系统 MUST 在将CAM++/emotion2vec迁移为原生torch_npu后保留既有A/B分类、总音频/B上限、段/Qwen并发、数组派生的能力启用状态、错误优先级和缓存清理规则。设备为NPU时实际走torch_npu，CPU时走PyTorch CPU，不以已退役TensorRT开关、engine存在或设备数组绕过功能校验。请求取消 MUST 等待已启动线程、NPU实际工作、结果读取/必要同步及清理完成后再准确归还模型实例和整音频租约；异步提交或线程返回不代表设备完成，重复取消不得遗弃收尾。下列历史Scenario标题仅为兼容的规格标识，实际WHEN/THEN均采用现行原生后端，不恢复CUDA/TensorRT执行。

#### Scenario: B 上限不因后端切换而变化
- **WHEN** 总音频上限8、B上限4，已有3A+4B，增强模型由CPU参考路径迁移到torch_npu
- **THEN** 新B仍返回4017，新A仍可获准，实例容量仍由各模型设备数组和健康状态决定；总容量也满时仍按既有4013优先级返回

#### Scenario: 普通转写不触发增强模型
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify`均为false，即使NPU增强设备数组非空或模型已加载
- **THEN** 请求仍是A，不因该请求加载、租用或调用CAM++/emotion2vec，不要求任何engine；本地Qwen转写按自身契约处理

#### Scenario: 启用 TensorRT 后请求关闭能力
- **WHEN** 请求需要的增强能力设备数组为空，或直接调用试图绕过数组关闭
- **THEN** 仍在音频准入前按既有单项/组合4015、4016、4018错误返回，不因后端选择绕过校验或伪造标签

#### Scenario: CUDA 工作尚未完成时取消
- **WHEN** 已准入B请求被取消或重复取消，而某NPU实例的设备工作或结果读取尚未完成
- **THEN** 原任务继续持有该模型和整音频租约，等待实际设备工作、同步和必要清理后才归还；不提前复用可写缓冲，最终计数一致且不重复释放

#### Scenario: engine 初始化失败
- **WHEN** 已准入请求所需增强模型的全部实例权重加载或目标设备初始化失败
- **THEN** 返回对应模型不可用4015或4016，完成请求清理和音频租约释放；不自动改走CPU、CUDA或组合能力关闭错误

### Requirement: 移除 v2 Qwen 路由
服务 MUST NOT 再注册或文档化 POST /v2/seacraft_asr；客户端访问该路径 MUST 得到未找到路由的结果，v3 是本链路唯一公开版本。

#### Scenario: 访问已移除的 v2 路径
- **WHEN** 客户端向 /v2/seacraft_asr 发起请求
- **THEN** 服务不执行转写处理并返回未找到路由的结果

#### Scenario: v3 语义迁移
- **WHEN** 客户端把原 v2 请求路径替换为 /v3/seacraft_asr
- **THEN** VAD 切分、Qwen 调用、A/B 音频准入、role/emotion 错误码和响应字段继续遵循本能力其余要求

### Requirement: 正式持续多实例采用独立完整音频验收
正式continuous多实例 MUST 按2026-11-24用户范围在正式源码真实v3上完成最终N4/CAM4/emo1的同源a1～a4 B4及增强池驻留后的纯A4；已执行N6 A6/A4/B4、共驻OOM和N5补测完整保留，不能用N4冒称N6通过；B4的showSpk/showEmotion均true、showRoleIdentify=false。输入身份、冷暖态、生成与复读配置、每owner S及派生容量须固定；组件、实验194.81秒或历史八路成绩 MUST NOT 替代本矩阵。纯A约200秒仍是单列目标，B4只如实报告实测，不作同样承诺；336/OOM/owned故障停止、正常关闭及原质量限制不变。

#### Scenario: 正式源码完整验收
- **WHEN** 完成正式多实例接入并执行验收
- **THEN** 原a1～a4完整源真实经过加载/VAD/过滤/裁剪/三路调度/ITN及HTTP响应，分别报告最终B4、增强驻留A4的完整或删失结果及具体差异，另保留原N6/N5历史，不减少输入工作量或用缓存VAD替代

#### Scenario: 历史专项仍未完成
- **WHEN** 新continuous矩阵已有结果，而旧legacy八路或旧open change仍有未验收任务
- **THEN** 两类结果、任务与适用范围分别保留，新结果不能勾选旧任务，也不把旧八路条件当成N4缺省启动条件
