## MODIFIED Requirements

### Requirement: 以 VAD 段作为转写与时间轴单元
系统 MUST 在服务本地对音频做 VAD 切分，并以「一个 VAD 段对应一个独立本地 Qwen 逻辑任务、至多一个有效响应分段」组织结果，`bg` 与 `ed` MUST 取原 VAD 边界。系统 MUST NOT 合并相邻段或通过落盘、拼接、补零改变段音频；正式执行允许经验证的独立段批处理，padding仅限模型输入且须有有效mask，不能改变有效PCM；批内保留独立归属和无有效文本的过滤规则。

#### Scenario: 音频被切分为多段
- **WHEN** 音频包含多段人声
- **THEN** 系统按原VAD顺序为每段建立独立本地Qwen任务，响应起止时间与对应段边界一致

#### Scenario: 音频无人声
- **WHEN** VAD 未检测到任何有效人声
- **THEN** 系统返回 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`

#### Scenario: 过滤过短语音段
- **WHEN** 某VAD段时长小于配置的最小语音段时长
- **THEN** 系统在三个模型分发前移除该段，不执行Qwen、CAM++或emo且不产出响应分段；本轮最短有效段为0.3秒，恰好0.3秒保留

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 将`segment_concurrency`和`request_concurrency`分别用于单音频和全局Qwen阶段的逻辑在途段数，超出部分有界等待；物理批内每段独立计数。系统 MUST 实施每音频16、全局128的验证目标并结合真实批调度，不把等待128段当成实际128批执行。Qwen完成必要收尾后 MUST 独立释放自己的段名额，不等待CAM++或emo。交付默认值 MUST 具有同条件实测依据，单音频/全局窗口、批上限和实例数分别可观测。

#### Scenario: 单个音频产生大量分段
- **WHEN** 一个音频切分出的有效段数超过配置的 segment_concurrency
- **THEN** 系统按配置的 `segment_concurrency` 执行逻辑段任务，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **WHEN** 多个音频同时进入处理
- **THEN** 全部音频合计执行的转写推理并发数不超过 配置的 `request_concurrency`

#### Scenario: 二四八完整音频供给Qwen
- **WHEN** 每音频窗口16、全局128，分别有2、4、8条已准入音频且段数充足
- **THEN** 在途Qwen逻辑段上限分别为32、64、128；调度不等待凑满128，实际批大小受已就绪数和已验证预算限制

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 按请求能力将有效段独立分发给Qwen、CAM++和emo，各自有界排队并释放阶段名额，结果按原请求和段序号汇合。各池健康设备数组决定实例容量，不共享模型槽位；总/B准入不改变模型池容量。CPU使用原生PyTorch，NPU使用torch_npu；CAM++/emo MUST 保持已验证FP32参数、浮点缓冲和计算策略。MUST 保留功能开关、段级失败降级和请求级错误优先级；CAM从0.3秒起实际推理，emo仅1～10秒实际推理，区间外按原规则“平淡”。旧Scenario标题仅为稳定标识。

#### Scenario: 开启角色与情绪能力
- **WHEN** 两个功能开关均为 `true`，请求 `showSpk=true`、`showEmotion=true` 且模型推断成功
- **THEN** 响应分段按适用时长得到真实role/emotion，emo范围外直接“平淡”；三个分支独立排队并分别使用所属模型池，慢分支不在预算有余时阻塞已完成Qwen名额释放

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

#### Scenario: 本轮不启用教师学生判定
- **WHEN** 请求`showSpk=true`而`showRoleIdentify`未传或为false
- **THEN** CAM++和既有说话人聚类仍运行，不执行额外教师/学生判定规则；现有可选角色识别接口能力不被删除

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

#### Scenario: 八CPU实例同时处理八音频
- **WHEN** 八条完整音频通过准入且八个CPU VAD worker均健康空闲
- **THEN** 每条完整音频分配一个worker，八条在VAD阶段并发处理，状态区分配置八份、健康八份与实际忙碌数

#### Scenario: 四worker及时接力八音频
- **WHEN** 八条完整音频通过准入但仅四个健康VAD worker，前四条已开始
- **THEN** 后四条按FIFO等待，任一worker完成VAD及必要交接即供下一条使用，不等前四条全部结束，也不等已完成VAD音频的Qwen/CAM/emo结束；完整音频准入租约不随VAD租约释放

### Requirement: v2 本地增强能力按模型池独立排队
该历史标题仅对应现行v3。系统 MUST 将CAM++与emo分别送入各自有界独立模型队列，与Qwen阶段解耦。任何模型池 MUST NOT 借用另一池实例或改变Qwen配置窗口；预算未满时增强等待 MUST NOT 占住已完成Qwen名额。系统 MUST 分别定义实例数、预取等待深度、实际批大小，维持任务连续供给及按原段汇合。

#### Scenario: CAM++ 槽位少于 emotion2vec 槽位
- **WHEN** CAM++ 设备数组配置 3 个实例且 emotion2vec 设备数组配置 1 个实例
- **THEN** CAM++ 和 emotion2vec 分别按自身容量排队，空闲的 CAM++ 实例不能被 emotion2vec 请求占用

#### Scenario: 本地模型池等待期间 Qwen 仍可调度
- **WHEN** 某个 VAD 段等待 CAM++ 或 emotion2vec 实例
- **THEN** 其它就绪段可按Qwen自身窗口进入转写，无需先获得任何增强实例；已完成Qwen名额及时释放，segment/request上限保持生效，有界缓存耗尽时明确记录反压

### Requirement: 记录本地模型池排队对端到端耗时的影响
系统 MUST 为v3记录CAM++和emo的调用、跳过、失败、推理包络及队列等待，并与音频准入、VAD、Qwen、汇合及后处理分开统计。性能报告 MUST 使用同源A/B配对和相同过滤配置，区分阶段累计时间与关键路径，不将并行耗时相加冒充端到端，也不因B样本段更少就宣称增强更快。

#### Scenario: 8 路完整音频压测
- **WHEN** 对本轮相同八条完整SHA256去重音频执行1/2/4/8并发及同源A/B配对，并补默认B4的4A+4B八路混合批次
- **THEN** 结果包含总/B实际活动峰值、两个模型池各自的busy/waiting与调用/跳过/失败情况、等待耗时、客户端端到端P50/P95、错误数和音频吞吐，A请求不触发未请求增强能力

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量，保持其它模型、批大小、预取深度及样本条件不变
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

### Requirement: 验证效果与性能边界
系统 MUST 使用`test/音频文件/`的实际音频建立按源SHA256去重的八条完整文件清单，分别完成1/2/4/8完整音频并发测试，每档处理同一八文件总工作量。系统 MUST 区分冷启动、暖态、同源A/B、实际批大小及多副本候选，覆盖完整时间轴、复读、资源、客户端P50/P95与吞吐。420秒触发诊断的轮次 MUST 标为未完成，不盲跑。旧g1/g2及三CAM/一emo仅作历史参考；无人工真值不输出绝对准确率、WER或时间戳准确率。

#### Scenario: 完成并发压测
- **WHEN** 按本轮测试计划依次执行1/2/4/8完整音频并发、同源A/B及八路混合批次
- **THEN** 分别记录相同八文件、并发/实际活动峰值、配置/真实批大小、各阶段等待与执行、加载是否计入、NPU与host资源、端到端耗时、错误、语种和质量观测；同源短片不冒充完整文件，触发诊断或未执行档位如实列明，不冒称通过

#### Scenario: 产出测试报告
- **WHEN** 交付本轮验证结果
- **THEN** 仓库保留可复算中文报告与原始响应/日志/清单，说明文本一致性、分段/时间边界、幻觉/复读样例及证据限制；没有人工标注或可追溯旧设备基准时明确未知，不将短片一致或无错误宣称绝对准确率或最优性能

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 验证role/emotion模型数值和业务映射，采用同权重可追溯CUDA golden或CPU FP32参考并明确限制，不恢复旧接口制造基准。本轮 MUST 用CAM++八NPU:7/emo八NPU:7、VAD八CPU及单Qwen优先完成同源A/B的1/2/4/8完整音频测量；B8另配子上限8，默认B4保持不变。MUST 记录短CAM参与聚类、emo时长跳过、健康容量和资源增量，不能将旧三/一池或旧g1/g2结果冒充本轮验收。

#### Scenario: 角色与情绪效果比对
- **WHEN** 使用同段PCM、原始权重、特征与映射规则比较CPU FP32参考和NPU原生输出
- **THEN** 保留CAM++192维embedding、emotion2vec完整分类及误差/容差证据，再核对角色聚类与业务标签；记录不一致样本，无CUDA golden或人工角色/情绪标注时明确未验证相应准确率，不调用已404的v2/v1.1.8充作参考

#### Scenario: 8 路非对称模型池压测
- **WHEN** 宿主npu:7使用八CAM++/八emo、单Qwen优先及八CPU VAD，执行八条完整去重音频的1/2/4/8同源A/B和默认B4的八路混合测试
- **THEN** 记录实际健康实例、总/B峰值、各池busy/waiting、逐请求客户端耗时及线性插值P50/P95、吞吐、错误、NPU/host与共享卡干扰；不同容量另列，不与本轮条件混算，按服务八路报告契约保留原始证据并关闭owned资源

#### Scenario: 并行开销可接受
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 保留单路全开组P50增幅不超过30%的既有门槛；各档按同源配对量化增幅，超过门槛或420秒诊断线时定位对应排队/计算卡点，不以不同源文件比较代替；未解决偏差不得标为通过

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **THEN** 两组的VAD段清单及时间轴一致；最终响应因既有空文本/标点过滤发生差异时逐段解释。以相同批调度设置复核段内文本差异，不高于仅转写组自身两次独立请求的波动幅度，不能以开增强为由改变有效PCM或VAD边界

#### Scenario: 显存与吞吐可记录
- **WHEN** 在已记录的非对称模型实例组合下运行所请求增强能力
- **THEN** 记录NPU利用率与HBM、当前服务进程allocator、RSS/PSS等可观测值及其范围，未知标明原因；加载窗口delta不能相加当逐模型独占量。记录吞吐（音频/分钟、段/秒）和两个模型池等待，不以不存在的CUDA SM指标充作NPU数据，写入中文报告

## ADDED Requirements

### Requirement: 性能验证使用指定段过滤配置
本轮性能验证 MUST 固定`min_segment_duration_s=0.3`，三个服务段级过滤开关`vad_snr_thres_filter_enabled`、`vad_decibel_thres_filter_enabled`、`vad_low_volume_filter_enabled`均为false，`vad_decibel_thres=-100`、`after_start=-1`、`before_end=-1`。MUST 记录FSMN实际选项及来源，保留未改变的兼容参数与绑定，不把声明等同于模型生效选项，不改变原FSMN算法。

#### Scenario: 三项段过滤关闭
- **WHEN** 本轮基线与候选执行相同音频
- **THEN** 两者均只按0.3秒最短段规则及原FSMN检测产生清单，不运行三项已关闭的服务后过滤；-1/-1只表示不限位置，不充当过滤开关

#### Scenario: 与历史开启过滤结果比较
- **WHEN** 引用旧迁移报告的完整音频耗时
- **THEN** 明确旧过滤和段数差异，仅作背景；严格加速比使用本轮相同PCM段清单、模型、生成参数和过滤配置的对照
