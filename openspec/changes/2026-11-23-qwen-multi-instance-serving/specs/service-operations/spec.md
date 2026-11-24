# Spec Delta

## MODIFIED Requirements

### Requirement: 明确实例数量的语义
系统 MUST 保留instance_count表示完整HTTP服务进程数量、缺省1的语义，不能用它表示Qwen owner数。正式continuous多owner默认及本次验收 MUST 为一个HTTP服务内N个owned模型进程，N缺省4，合法整数1～8仍保留；增加N不增加HTTP、VAD或增强服务副本。显式多HTTP部署的各服务仍独立持有请求容量和模型资源，不把局部容量称为跨服务全局限制。

#### Scenario: 配置多个实例
- **WHEN** 配置中instance_count大于1
- **THEN** 容器内按原部署方式运行对应数量的完整HTTP服务并负载均衡；每服务分别创建其配置的Qwen/VAD/增强资源，报告总资源放大，不将该配置冒称本次单HTTP多owner验收

#### Scenario: 配置缺省
- **WHEN** 配置文件中缺少实例数量配置
- **THEN** 系统按默认一个HTTP实例运行；默认四Qwen owner不改变该值，也不生成四套请求处理服务

### Requirement: 维护配置项清单
系统 MUST 只消费实际生效配置，删除下线功能专属字段，保留既有旧VAD和设备结构迁移诊断。正式continuous MUST 以N、S分别派生audio_concurrency=N和request_concurrency=N×S，根与Docker模板不重复维护这两个值；兼容显式相等值须提示迁移，冲突或非法类型/范围须就绪前报错。legacy显式选择时保留原独立容量参数，不能通过忽略冲突配置静默切换语义；修改均重启生效。

#### Scenario: 配置文件包含已下线功能的配置项
- **WHEN** 部署配置中残留vad_pool_size、vad_pool_max、vad_queue_watermark、vad_collapse_timeout_s、vad_queue_timeout_s或vad_call_timeout_s
- **THEN** 服务不使用这些字段改变固定VAD数组，并记录迁移提示或配置校验告警

#### Scenario: 配置文件包含旧模型设备结构
- **WHEN** 部署配置中残留model_device或model_device.local_model_pool结构
- **THEN** 服务不使用旧结构覆盖model_device_pool，并提示迁移到新的设备数组

#### Scenario: 缺少可选配置项
- **WHEN** 配置文件缺少某个仍有效的可选配置项
- **THEN** 服务使用loader与交付模板一致的声明默认值，continuous的总音频及全局段容量仍从N、S派生，不因旧字段缺失回退到8或128

#### Scenario: 旧容量值与派生值一致
- **WHEN** continuous配置N=4、S=16且旧字段显式audio_concurrency=4、request_concurrency=64
- **THEN** 允许启动并给出删除重复字段的迁移提示，状态报告其派生来源，模板不继续复制这两个赋值

#### Scenario: 旧容量值冲突或类型非法
- **WHEN** continuous的旧显式容量与N或N×S不一致，或N、S、容量字段包含布尔值、非整数、越界值
- **THEN** 就绪前报具体字段及期望派生值，不静默取较小值、忽略冲突或将非法值纠正为1

### Requirement: 启动时固定预热 VAD 实例
服务 MUST 仅按vad.vad_device数组在启动阶段以spawn创建固定VAD worker池，每元素一个独立worker，FIFO分派完整音频；旧qwen_asr.vad_device不得回退生效。正式本轮配置 MUST 保留八个CPU VAD实例，不随Qwen N扩缩；通用合法CPU/NPU数组仍按原规则支持。启动完成前记录逐worker设备、预热及失败，无可用worker拒绝就绪；运行期不自动扩缩，NPU失败不静默走CPU。

#### Scenario: 默认混合设备配置
- **WHEN** 使用本轮正式配置vad_device的八个cpu元素启动服务
- **THEN** 建立八个固定CPU worker并报告逐份健康结果；历史场景标题不表示本轮默认仍为混合落卡，Qwen四owner不改变VAD数量

#### Scenario: 启动时设备不可用
- **WHEN** VAD数组中的某目标设备不可用或对应worker预热失败
- **THEN** 记录该worker设备及原因，不计健康容量；无成功worker拒绝启动，不动态补worker或以CPU回退掩盖失败

#### Scenario: 只改变Qwen实例数
- **WHEN** N或S改变但VAD设备数组未改
- **THEN** VAD仍按原固定数组创建，状态区分配置八份、实际健康与busy数量，不把Qwen容量当作VAD容量

### Requirement: 默认配置体现有效并发来源
正式最终缺省 MUST 为continuous/process、N=4（合法1～8保留）、宿主npu:7、S=16、preload_on_startup=true、总音频派生N、全局段派生N×S及B配置上限4；HTTP缺省1。VAD、CAM++和emo配置实例数仍各由设备数组决定，最终增强池默认为用户选择的CAM4/emo1，须在四Qwen共驻B4及增强驻留A4中真实验证并冻结；不能由N推导或仅凭选择宣称已通过。每owner物理步及输入预算与全服务PCM预算 MUST 分开说明，不能增加队列即声称增加执行容量；loader、根/Docker模板、README及部署说明的有效值与迁移规则保持一致。

#### Scenario: 检查默认配置
- **WHEN** 运维读取config.toml或docker/config.toml
- **THEN** 能区分HTTP1、Qwen N4、每音频S16、总4/全局64、B4、独立CAM4/emo1及PCM预算；根配置设备为宿主npu:7，单卡容器按物理7实际可见映射使用逻辑npu:0，不把音频容量当模型池份数

#### Scenario: 功能关闭时模型池不生效
- **WHEN** open_v2_role=false或open_v2_emotion=false
- **THEN** 对应模型数组不触发加载或推理，配置可保留供重启后开启；Qwen默认预加载不绕过增强功能开关

### Requirement: 支持安全的测试收尾
运维工具 MUST 停止新请求和owner授予，等待已启动设备工作、分支及必要清理，并关闭本次owned服务、Qwen owner、增强/VAD worker及代理。正式验收 HTTP drain缺省30秒、模型关闭独立grace120秒、外层owned回收210秒，不改业务调用超时。资源启动时登记PID及出生身份，部分启动也回收；共享服务保留，不按名称或端口批量终止。强制、超时、坏应答或非零退出 MUST 保留关闭失败，不只以解除监听证明全部正常。

#### Scenario: 收到测试终止信号
- **WHEN** 资源安全闸门触发或运维结束测试
- **THEN** 停止提交和授予，按30/120/210秒各自边界等待本次在途及owned关闭；逐份记录退出码、强制与出生身份，核验无孤儿、端口至少连续5秒空闲和私有音频缓存为空，取消不遗弃收尾

#### Scenario: Qwen 服务保持运行
- **WHEN** 服务器存在不属于本次测试生命周期的Qwen或其它共享服务
- **THEN** 这些资源保持原状；无需外部Qwen服务或固定转写端口，本次owned内嵌owner随服务关闭，不继续后台占用，共享服务不能因本次超时被终止

### Requirement: 增强音频并发配置具有确定默认值
系统 MUST 从audio_spk_emo_concurrency读取B配置上限，缺省4，正整数且重启生效，根与Docker模板均展示；关闭能力使用功能开关。正式continuous的B新准入有效上限 MUST 不超过min(配置B上限,健康Qwen容量)，不得据此改写配置或复制增强池。健康下降时closing请求仍计原活动，安全终态前不伪释放；各HTTP服务独立维护，不是跨服务全局上限。

#### Scenario: 缺少新增配置项
- **WHEN** 旧配置未声明audio_spk_emo_concurrency
- **THEN** 服务使用B配置上限4，不自动回退为N；文档说明有效上限还受总音频与健康owner约束

#### Scenario: 显式配置其他上限
- **WHEN** 运维显式配置continuous N=8、相等旧audio_concurrency=8、audio_spk_emo_concurrency=2并重启
- **THEN** 每个HTTP服务配置总容量8、B上限2，健康时最多8音频且至多2B；旧相等容量仅兼容并给迁移提示

#### Scenario: 非法增强上限
- **WHEN** B配置值为0、负数、布尔值、浮点数或字符串
- **THEN** 就绪前报告字段和正整数约束，不静默纠正为1，不把0当作功能关闭

#### Scenario: 增强上限大于总上限
- **WHEN** continuous N=2且B配置上限4，两个owner健康
- **THEN** 配置有效，B有效至多2；状态仍分别报告配置总2和B4及有效值，不静默改写B配置

#### Scenario: 多服务实例
- **WHEN** 同一部署显式启动多个完整HTTP服务
- **THEN** 每个服务独立维护总音频、B和其健康owner容量，文档说明这些不是跨服务共享上限，不能用多服务成绩代替本次单HTTP验收

### Requirement: 暴露音频分类准入状态
GET /get_status MUST 保留audioConcurrency的configured_total、active_total、waiting_total及spkEmoConcurrency的configured、active、waiting，来自同一次准入快照；configured_total在continuous为N。附加状态 MUST 显示健康owner与当前有效总/B新准入容量，失效/closing中的活动租约按实际保留，不将其减至H。查询不触发模型加载，原业务和各池语义不变，A活动/等待仍可由总数减B数取得。

#### Scenario: 查询混合活动及等待状态
- **WHEN** 显式N=8、B上限4，正在处理5A+3B、等待1A和2B
- **THEN** audioConcurrency仍返回configured_total=8、active_total=8、waiting_total=3，spkEmoConcurrency为configured=4、active=3、waiting=2；A活动5、等待1，另列健康与有效容量

#### Scenario: 首次业务请求前查询
- **WHEN** 服务已启动但尚无音频请求，运维查询状态
- **THEN** 配置与实际预加载/健康状态如实返回，活动等待均0；CAM++和emo不因查询加载，preload=false的Qwen也不加载

#### Scenario: 功能关闭并拒绝增强请求
- **WHEN** 两种增强能力关闭且增强请求在准入前被拒绝
- **THEN** B活动与等待仍为0、B配置正常展示，不授予owner或增加总活动

#### Scenario: owner失效但请求尚未收尾
- **WHEN** 六个活动请求中一个owner失效，H降为5而其请求仍closing
- **THEN** configured_total保持6、active_total仍按真实未释放租约计6，健康与有效新准入容量另列5；不提前减活动或重租该owner，只在安全终态后释放

### Requirement: 文档明确分类限制与能力错误
README和部署说明 MUST 说明最终默认N4/S16、派生总4/global64、B4及CAM4/emo1、请求owner生命周期、健康降级、等待调度及按HTTP服务生效范围。错误码仍区分4013、4015、4016、4017、4018及等待4010，保留能力关闭优先和总容量满优先于B满；配置关闭、模型池不可用与普通段级失败降级不能混淆。legacy和旧A8矩阵保留各自历史范围，不用最终N4或历史N6/N5结果改写。

#### Scenario: 查阅默认并发示例
- **WHEN** 用户查阅正式最终默认总4、B4的说明
- **THEN** 文档说明4A、4B或总计4个混合请求使总满，新A/B均4013优先；本配置不能单独触发4017。显式N6下1A+4B新B4017、新A可形成2A+4B的通用契约及原实测保留为历史，不伪装N4可达状态；B双开仍只占一个B槽

#### Scenario: 查阅能力关闭行为
- **WHEN** 用户查询功能开关为false、增强请求参数为true的处理规则
- **THEN** 文档说明准入前请求失败及4015/4016/4018单项组合错误，不承诺配置关闭时以null成功；普通A仍按自身容量处理，已开启模型的普通段级失败另列null降级

### Requirement: 启动前校验部署前置条件
昇腾部署 MUST 在就绪前校验设备可见性、必需资产、torch_npu运行时及CONFIG_PATH可读性，失败明确报告并阻止就绪。三个必需模型资产与Qwen权重策略分开：preload=false时Qwen缺失在转写触发按4011处理，不为预检加载；正式默认preload=true时全部N份权重及processor须顺序加载成功且健康后才就绪，任何部分启动失败统一回收。校验不得触发未请求增强模型加载或绕过功能开关。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用NPU、必需模型目录缺失或配置不可读
- **THEN** 启动指出具体前置条件并失败，不接受请求；默认Qwen预加载失败同样阻止就绪并回收已启动owner，显式懒加载的Qwen缺失仍按原4011处理

#### Scenario: 前置条件通过
- **WHEN** 设备、目录和配置检查通过，且按当前preload策略完成必要加载
- **THEN** 服务就绪并报告镜像tag、实际配置路径、后端和N份实际健康加载状态；不把预检、READY或模型名当真实语音forward暖身

## ADDED Requirements

### Requirement: 每owner预算与配置边界明确
正式配置的max_inference_batch_size MUST 为可选项，未显式提供时随S派生，根/Docker模板不得固定物理步或prefill值16；显式物理步大于S须拒绝，prefill缺省派生min(16,S)、显式值不得超过该值；每owner帧/token缺省14400/10000，不因N变化缩放。S、max_new_tokens或预算改变时 MUST 校验支持范围及单段可容纳性，说明可按预算拆prefill；全服务PCM256段/256MiB另列且不复制。各默认、有效值、重启语义与类型约束 MUST 同步loader、宿主/Docker模板及文档。

#### Scenario: 仅修改N
- **WHEN** N在支持范围内改变而S16及输入预算不变
- **THEN** 派生总音频及全局段窗口改变，每owner仍帧14400/token10000，物理步上限16，PCM仍为同一个全服务预算，不把原global60000/86400当作每owner默认

#### Scenario: 只改变每音频窗口
- **WHEN** 根或Docker配置仅把S由16改为8而未显式提供物理步上限
- **THEN** 物理步上限派生为8、prefill有效上限至多8，不因模板遗留固定16而冲突；每owner帧/token预算仍按显式值或缺省值校验

#### Scenario: 非法物理步或预算配置
- **WHEN** 显式物理步上限大于S、prefill上限大于min(16,S)，或预算类型/范围及单段可容纳性不满足约束
- **THEN** 在服务就绪前报告具体字段和支持条件，不能静默截断、放大预算或改变段音频；有效但只能较小prefill的预算须如实显示实际限制

### Requirement: 多owner状态身份和资源可核对
状态 MUST 按owner区分配置、加载、健康、租用、可用数量、请求归属、设备、活跃段、预算、失败与PID/出生身份，保留父/worker及共享外部资源的范围和采样age。查询 MUST 仅读有界缓存，不等待忙worker、不保存正文、不触发spawn或加载；无法观测字段明确unknown，不能用父进程0代替各worker资源或将配置N冒充健康N。

#### Scenario: 四owner均健康并出租
- **WHEN** 最终N4的A4或B4活动且查询状态
- **THEN** 可核对四个不同owned出生身份、四个请求各自唯一owner、每owner至多S及合计至多N×S的逻辑在途；物理步大小和逻辑窗口分别报告

#### Scenario: 部分失败或资源快照过期
- **WHEN** 某owner失效、仍closing，或资源采样失败/过期
- **THEN** 分别保留健康下降、原未终态租约和失败历史；资源附age及unknown原因，不用旧采样、端口可通或配置数伪造实时健康容量

### Requirement: 多owner暖身覆盖和正式计时分离
正式多owner性能轮 MUST 先验证全部已配置owner的实际加载与真实语音forward/prefill，联合PID出生及请求/段执行证据确认覆盖；READY、模型名或发送N次HTTP不等于N份覆盖。暖身 MUST 有界、原件独立保存，并在全部安全归零后保存暖身后计数基线；正式HTTP计时、分位数和模型阶段增量不得混入加载暖身。暖身不依赖SHA路由或新增公开指定owner参数。

#### Scenario: 请求次数足够但owner覆盖不足
- **WHEN** 多次短clip真实v3请求只覆盖部分owner
- **THEN** 不能开始正式性能轮或宣称N份就绪，记录实际覆盖并在有限暖身界限内处理；界限耗尽则诊断及owned收尾，不无限补请求

#### Scenario: 全部owner覆盖完成
- **WHEN** 四owner各有真实forward/prefill且窗口、PCM归零
- **THEN** 独立保存四份身份覆盖及暖身后计数，再共同起跑最终N4的指定A4/B4；统计从暖身后基线扣除，不能用暖身前initial状态冒充暖身后基线

### Requirement: 增强池交付默认来自最终四Qwen共驻标定
CAM++和emo最终交付数组数量 MUST 按2026-11-24用户选择采用CAM4/emo1，并在四份Qwen真实驻留的B4及增强驻留A4中独立验证、标定并冻结；原N6各1共驻OOM、N5各1补测保留，不能拿旧小池成绩交付未经验证的4/1。MUST 不盲扫组合，也不把N4、B4或旧各8本身当已通过。最终值写入loader、根/Docker模板及说明，真实调用、skip、段级失败和质量限制分别报告，不以仅加载或HTTP200证明增强通过。

#### Scenario: 一份增强实例成为瓶颈
- **WHEN** 原N6/N5共驻B4显示CAM等待显著，用户据实选择最终N4/CAM4/emo1
- **THEN** 仅按该明确选择验证最终组合，保留原瓶颈与失败，不再自动扫档；VAD、精度、生成和文本策略不变，真实故障停止后续压力并保留证据

#### Scenario: 选择最终增强池数量
- **WHEN** 最终N4/CAM4/emo1完成B4共驻标定及同a1～a4增强驻留纯A4验证
- **THEN** 交付模板与loader采用同一冻结组合，记录完整/警告/资源与正常关闭；纯A约200秒目标单列，B4无200秒硬门槛。N4为用户批准的替代最终范围，不声称原N6共驻已通过
