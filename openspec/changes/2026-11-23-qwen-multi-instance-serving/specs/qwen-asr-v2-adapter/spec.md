# Spec Delta

## MODIFIED Requirements

### Requirement: 限制同时处理的音频数量
continuous模式总音频配置容量 MUST 由Qwen实例数N派生（默认6），新请求只可原子取得一个健康可租owner和音频槽位；legacy显式模式保留其配置上限。每请求 MUST 持有同一owner至处理及全部设备/临时资源安全收尾，成功、异常、取消均准确释放。默认不等待，总满4013优先；B另受增强子上限限制。健康下降仅限制新授予，MUST NOT 伪释放仍在收尾的租约。

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

#### Scenario: 默认六实例音频准入
- **WHEN** continuous配置N=6且六owner健康，已准入六条音频
- **THEN** 新A或B按既有默认立即拒绝策略返回4013，不创建第七Qwen、不按文件名或SHA分流；现有请求的owner归属不变

#### Scenario: 任意文件与后续请求复用
- **WHEN** 两个同名或同内容音频并发提交，随后另一个新音频到达
- **THEN** 每个请求按request_id独立租健康owner并保持到安全归还；新请求可以复用已安全归还的owner，不要求测试源清单

### Requirement: 限制对 Qwen 服务的并发请求数
continuous中每owner及其所绑定音频的Qwen在途上限 MUST 为配置S（默认16），全局配置上限 MUST 由N×S派生，默认96；超出等待，不重复维护冲突的request_concurrency。名额覆盖准备、待执行、设备执行和安全收尾，段独立完成即归还，不等待整组或增强汇合。物理步不得超过该owner的S，增加信号量不代表吞吐扩容；legacy显式模式保留原参数契约。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出的有效段数超过配置的 segment_concurrency
- **THEN** 系统按配置的 `segment_concurrency` 执行逻辑段任务，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计执行的转写推理并发数不超过 continuous的派生N×S（legacy为其配置的`request_concurrency`）

#### Scenario: 短段结束及时补位
- **WHEN** 同一owner一个段已安全结束而其它段仍生成，且音频还有待处理段
- **THEN** 立即归还该段credit并在下个可用安全步接纳后续段，保留其它段独立KV；不等待16段一起结束、不跨owner迁移已执行段

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
- **WHEN** continuous配置N=6、S=16但旧audio_concurrency或request_concurrency显式值与6或96冲突
- **THEN** 服务启动前明确指出冲突字段、派生值和迁移办法，不静默忽略或启动成另一容量；相等值仅兼容读取并提示迁移

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 在VAD过滤、段credit与PCM预算取得及有效段裁剪后，并发发起同段Qwen、所请求CAM++及emo任务；各池独立等待，不把增强串成Qwen完成后调用，不承诺同卡kernel物理同时。CAM/emo按独立设备数组的健康容量执行，不随N复制或互借；CPU用PyTorch、NPU用torch_npu，保持已验证FP32参数/缓冲/计算，不使用旧FP16或TensorRT。功能开关、段级失败降级、错误优先级、现有聚类/ITN及业务映射 MUST 保留。

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

#### Scenario: 增强时长边界
- **WHEN** 有效段小于0.3秒，或保留段处于0.3～不足1秒、1～10秒含边界、超过10秒
- **THEN** 小于0.3秒三路均不执行；保留且不少于0.3秒的段可做CAM，emo仅1～10秒真实推理，区间外按原规则返回平淡；skip单列且不计NN推理成功

#### Scenario: Qwen先完成而增强仍在途
- **WHEN** 同段Qwen已达到真实安全终态，CAM或emo仍等待或执行
- **THEN** Qwen段credit先归还使后续段可补位；该段PCM仍由请求持有至三路均安全终态，取消不能提前释放PCM或正在执行的模型实例

### Requirement: v2 本地增强能力按模型池独立排队
系统 MUST 让每段CAM++和emotion2vec进入各自独立共享池，不把每个Qwen owner复制成一套增强池，也不让两个池互借实例。增强等待 MUST NOT 持有已经安全结束的Qwen段credit；全服务PCM仍在全部所需分支终态后释放，并允许正常有界反压。段的三路入口仍受Qwen准备窗口及PCM预算约束；这不意味着每池实际并发等于N×S。

#### Scenario: CAM++ 槽位少于 emotion2vec 槽位
- **WHEN** CAM++ 设备数组配置 3 个实例且 emotion2vec 设备数组配置 1 个实例
- **THEN** CAM++ 和 emotion2vec 分别按自身容量排队，空闲的 CAM++ 实例不能被 emotion2vec 请求占用

#### Scenario: 本地模型池等待期间 Qwen 仍可调度
- **WHEN** 某个 VAD 段等待 CAM++ 或 emotion2vec 实例
- **THEN** 其它获准段仍可推进Qwen，安全结束的Qwen credit可先归还；每owner S及派生N×S继续限制Qwen，全服务PCM约束仍生效

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

#### Scenario: 默认四B仍允许两个A
- **WHEN** continuous N=6且六owner健康、B配置4，已准入0A+4B且默认不等待
- **THEN** 第五B返回4017，另两个A可以获准形成2A+4B；此时总满，新A/B均4013，不因模型池仍有空实例绕过准入

#### Scenario: 健康容量下降
- **WHEN** 配置N=6/B4但可用于新授予的健康owner仅3，已有故障请求仍在安全收尾
- **THEN** 状态分列configured、healthy、leased/closing占用；B有效新准入容量不超过3，不迁移在途KV、不伪释放旧请求，健康资源实际可租后才授予新请求

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

### Requirement: 验证效果与性能边界
系统 MUST 使用显式清单、完整可追溯音频保存效果与端到端性能证据。下列八文件4B+4A/g1g2场景仅为显式选择的历史legacy专项，不是新continuous缺省验收，未执行不得称完成。正式continuous另按新增A6/A4/B4及共驻要求验证，两类矩阵不得替代。报告须区分冷态/稳态/候选、过滤/复读/NPU资源及客户端时延；无人工真值不得输出绝对准确率、WER或时间戳准确率。

#### Scenario: 完成并发压测
- **WHEN** 显式选择历史legacy测试计划执行单路、同源暖标定或八条独立完整音频混合批次
- **THEN** 分别记录样本、并发/实际活动峰值、g1/g2、加载是否计入、NPU与host资源、端到端耗时、错误、语种和质量观测；同源重复暖标定不冒充八条独立文件，未执行的历史并发专项列为未执行/不适用及原因

#### Scenario: 产出测试报告
- **WHEN** 交付本轮验证结果
- **THEN** 仓库保留可复算中文报告与原始响应/日志/清单，说明文本一致性、分段/时间边界、幻觉/复读样例及证据限制；没有人工标注或可追溯旧设备基准时明确未知，不将短片一致或无错误宣称绝对准确率或最优性能

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 验证CAM/emo数值、业务标签、聚类和段结构，记录端到端延迟、吞吐、各池等待及资源范围；优先可追溯同权重CUDA golden，缺少则用CPU FP32并说明限制，不恢复旧接口。下列八路非对称池/g1g2与单路30%场景仅在显式历史legacy专项适用；正式continuous按独立B4要求标定，不以三CAM/一emo或旧6:2充当新默认，不把A6目标承诺给B4。

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

### Requirement: 记录本地模型池排队对端到端耗时的影响
系统 MUST 为v3记录CAM++/emo实际调用、推理、等待、时长skip与失败，分别比较VAD、裁剪、Qwen、各池等待、汇合、聚类和ITN。下列八路g1/g2为显式历史legacy专项，正式continuous使用固定A4/B4及增强驻留A6，不能互作验收。所有实际能力调用与请求开关关联，A不调用未请求增强；本次不改变聚类/ITN执行器或算法。

#### Scenario: 8 路完整音频压测
- **WHEN** 显式选择历史legacy专项，对八条完整SHA256去重WAV按4B+4A执行同条件g1/g2混合批次
- **THEN** 结果包含总/B实际活动峰值、两个模型池各自的busy/waiting与调用/跳过/失败情况、等待耗时、客户端端到端P50/P95、错误数和音频吞吐，A请求不触发未请求增强能力

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量而不新增槽位配置
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

#### Scenario: 正式四路双开增强对照
- **WHEN** 正式continuous在六份Qwen均驻留下，对同一a1～a4分别执行A4与showSpk/showEmotion均true的B4
- **THEN** 记录完整n4或删失子集口径、真实role/emotion字段与调用/skip覆盖、各池等待/推理、聚类/ITN及HBM；没有双开真实调用不能仅凭HTTP200宣称增强通过

### Requirement: 完成固定配置的全量 WAV 质量压测
系统 MUST 支持显式清单完整WAV的v3质量与性能验证并保存响应、错误、耗时、过滤和复读证据。本需求中八去重文件4B+4A、总8/B4、单Qwen、三CAM/一emo及g1/g2仅作用于显式历史legacy专项；新continuous使用独立A6/A4/B4及共驻矩阵，不能勾选旧任务。VAD/RMS等阈值保持，不自动套用RMS-41或旧segment4/audio16，也不恢复v2或补造45文件配对。

#### Scenario: 45 个 WAV 全部配对并执行
- **WHEN** 显式选择历史legacy八条完整WAV专项，或另行提供历史45文件及其Paraformer配对参考
- **THEN** 本轮验证先核八文件ID、路径编码、大小/完整时长/SHA256及实际配置，按固定清单完成4B+4A；历史标题保留作稳定标识，不以缺少旧45配对阻塞本轮迁移，也不称其已完成。若另行执行历史参考专项则逐一核对配对，缺失/多余明确失败，不补造旧接口或基线

#### Scenario: 记录端到端分位数
- **WHEN** 显式历史legacy完整八文件清单请求全部完成或任一失败
- **THEN** 报告每档成功数/8、失败/超时与原始逐文件耗时；成功完整批次按既有八路契约计算客户端端到端P50/P95和最大值、RTF及吞吐，失败批次完整保留并标记不完整，不丢失败后冒称八路成功

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

## ADDED Requirements

### Requirement: 正式持续多实例采用独立完整音频验收
正式continuous多实例 MUST 在正式源码启动的真实v3上验证原a1～a6 A6、同源a1～a4 A4/B4及增强池驻留后的A6；B4的showSpk/showEmotion均true、showRoleIdentify=false。输入身份、冷暖态、生成与复读配置、每owner S及派生容量须固定；组件、实验194.81秒或历史八路成绩 MUST NOT 替代本矩阵。A6约200秒是目标，B4只如实报告实测，不作同样承诺。

#### Scenario: 正式源码完整验收
- **WHEN** 完成正式多实例接入并执行验收
- **THEN** 原六完整源真实经过加载/VAD/过滤/裁剪/三路调度/ITN及HTTP响应，分别报告A6、A4/B4、增强驻留A6的完整或删失结果及具体差异，不减少输入工作量或用缓存VAD替代

#### Scenario: 历史专项仍未完成
- **WHEN** 新continuous矩阵已有结果，而旧legacy八路或旧open change仍有未验收任务
- **THEN** 两类结果、任务与适用范围分别保留，新结果不能勾选旧任务，也不把旧八路条件当成N6缺省启动条件
