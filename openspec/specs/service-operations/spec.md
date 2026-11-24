# service-operations Specification

## Purpose
定义离线语音服务在运行期的可观测性与配置契约：服务日志的保留策略、状态查询接口的字段集合、实例数量的语义，以及配置文件中实际生效的配置项清单，避免无效配置长期沉积。

## Requirements

### Requirement: 限制服务日志的保留规模
系统 MUST 将运行日志写入配置的日志文件路径，并 MUST 对日志做保留控制：日志文件按天切分，单个日志文件大小 MUST NOT 超过配置上限（默认 100 MB），超过保留天数的日志文件（默认 7 天）MUST 被清理。清理 MUST 在服务启动时与写入过程中均生效。

#### Scenario: 日志文件超过大小上限
- **WHEN** 当前日志文件增长到配置的大小上限
- **THEN** 系统切分出新的日志文件继续写入，旧文件按保留策略处理

#### Scenario: 清理过期日志
- **WHEN** 存在超过保留天数的日志文件
- **THEN** 系统在启动时与写入过程中删除这些过期文件

#### Scenario: 配置缺省
- **WHEN** 配置文件中缺少日志保留相关配置项
- **THEN** 系统按保留 7 天、单文件上限 100 MB 运行

### Requirement: 提供稳定的状态查询契约
`GET /get_status` MUST 返回服务标识、状态、版本、运行时长与引擎标识等运维字段，其中版本 MUST 固定为 `latest`；引擎标识字段 MUST 保留在响应中并返回固定占位值，MUST NOT 依赖部署方单独配置。

状态响应 MUST 将 VAD 状态集中放在 `vadPool`，`localModels` 只返回 CAM++ 与 emotion2vec；固定设备数组不应返回无意义的 `pool_size` 或 `pool_max` 字段。

#### Scenario: 查询服务状态
- **WHEN** 客户端调用 `GET /get_status`
- **THEN** 响应包含 `status`、`appVersion`（取值 `latest`）、运行时长字段与引擎标识字段（返回固定占位值）

#### Scenario: 部署配置缺少引擎标识项
- **WHEN** 部署配置中不再提供引擎标识配置项
- **THEN** 状态接口仍返回引擎标识字段，取值为固定占位值，运维平台的字段结构保持不变

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

### Requirement: 降低控制台 HTTP 访问日志噪声
服务 MUST 仅在控制台处理 `httpx`/`httpcore` 成功请求的 `INFO` 访问日志，使其不再逐段打印 HTTP 200 行；`WARNING`、`ERROR` 和业务处理日志 MUST 继续输出。文件日志是否保留完整 HTTP 访问记录 MUST 与控制台过滤解耦，并按现有文件日志策略保留。

#### Scenario: 成功请求不刷屏
- **WHEN** Qwen 客户端收到 HTTP 2xx 成功响应
- **THEN** 控制台不打印对应的 `HTTP Request ... 200 OK` `INFO` 行，进度和业务耗时日志仍正常显示

#### Scenario: HTTP 警告和错误保留
- **WHEN** Qwen 请求产生连接警告、非 2xx 响应或异常
- **THEN** 控制台保留对应的 warning/error 记录，并包含可定位请求失败的上下文

#### Scenario: 非 TTY 日志输出
- **WHEN** 服务输出被重定向到文件、管道或测试捕获器
- **THEN** 不启用 tqdm 的 ANSI 控制序列；普通日志可被稳定解析，不能因为进度显示改变日志级别或丢失错误

### Requirement: 暴露资源诊断状态
`GET /get_status` MUST 在不改变既有字段语义的前提下提供可用于资源诊断的主进程、VAD worker、模型池和请求资源快照；字段缺失或采集失败 MUST 明确标记，不得伪造为零值。

#### Scenario: 查询资源状态
- **WHEN** 运维调用 `GET /get_status`
- **THEN** 响应可区分模型池实例、VAD worker、busy/waiting、主机内存/Swap 和 GPU 资源状态，并保留现有服务状态字段

#### Scenario: 资源指标不可用
- **WHEN** 当前环境无法读取某项主机或 GPU 指标
- **THEN** 该指标返回明确的不可用状态或原因，其他状态字段仍正常返回

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

### Requirement: 分类准入记录有界且可区分错误原因
系统 MUST 为 v3 请求记录 A/B 分类、音频准入等待耗时、准入时的总容量与 B 容量快照及有限集合的失败原因。日志和遥测 MUST 能分别统计总容量满、增强容量满、共享等待队列满、等待超时及能力关闭；遥测 MUST 保持有界存储，不保存音频、转写正文或模型对象，且保留既有阶段字段的语义。

#### Scenario: 区分两个满额错误
- **WHEN** 请求分别因总容量满返回 `4013`、因增强容量满返回 `4017`
- **THEN** 日志与最近请求遥测能够区分类别、错误码及满额原因，包含对应容量快照，汇总分别计数

#### Scenario: 同码不同拒绝原因
- **WHEN** 请求因共享等待队列满返回 `4013`
- **THEN** 记录等待队列满原因及等待使用量，与总活动容量满的 `4013` 分开统计

#### Scenario: 组合关闭错误和提前返回
- **WHEN** 请求因两项所需能力都关闭返回 `4018`
- **THEN** 日志和遥测记录 B 类及组合关闭原因，不虚构模型推理耗时，也不继承其他请求的阶段数据

#### Scenario: 等待和处理取消
- **WHEN** 请求等待超时或被取消
- **THEN** 遥测区分超时和取消，不将取消误记为处理成功，已回收的名额与状态快照一致

#### Scenario: 长期运行的遥测存储
- **WHEN** 请求数量超过既有遥测事件上限
- **THEN** 继续按有界历史保留策略更新事件，分类及拒绝原因统计明确使用该保留窗口，不创建按请求无限增长的记录或计数键

### Requirement: 文档明确分类限制与能力错误
README和部署说明 MUST 说明最终默认N4/S16、派生总4/global64、B4及CAM4/emo1、请求owner生命周期、健康降级、等待调度及按HTTP服务生效范围。错误码仍区分4013、4015、4016、4017、4018及等待4010，保留能力关闭优先和总容量满优先于B满；配置关闭、模型池不可用与普通段级失败降级不能混淆。legacy和旧A8矩阵保留各自历史范围，不用最终N4或历史N6/N5结果改写。

#### Scenario: 查阅默认并发示例
- **WHEN** 用户查阅正式最终默认总4、B4的说明
- **THEN** 文档说明4A、4B或总计4个混合请求使总满，新A/B均4013优先；本配置不能单独触发4017。显式N6下1A+4B新B4017、新A可形成2A+4B的通用契约及原实测保留为历史，不伪装N4可达状态；B双开仍只占一个B槽

#### Scenario: 查阅能力关闭行为
- **WHEN** 用户查询功能开关为false、增强请求参数为true的处理规则
- **THEN** 文档说明准入前请求失败及4015/4016/4018单项组合错误，不承诺配置关闭时以null成功；普通A仍按自身容量处理，已开启模型的普通段级失败另列null降级

### Requirement: 明确实例设备与 TensorRT 配置的优先级
模型设备数组 MUST 接受 cpu 与 npu:N。NPU MUST 固定走 torch_npu，CPU MUST 使用 PyTorch CPU；功能开关、请求能力与设备数组容量的既有含义 MUST 保留。系统 MUST NOT 保留仅对 NVIDIA 实例生效的加速器开关与引擎精度配置项。

#### Scenario: NPU 配置同时打开开关
- **WHEN** `[model_device_pool]` 中某实例为 npu:0
- **THEN** 该实例实际后端为 torch_npu，不要求任何 NVIDIA engine 或 TensorRT 依赖

#### Scenario: 重复 CUDA 设备
- **WHEN** CAM++ 数组为三个 npu:0
- **THEN** 配置容量仍为 3 个独立实例，不因共用同一设备或权重文件变成单实例

#### Scenario: 功能关闭时的配置保留
- **WHEN** 对应 features 功能开关关闭但设备数组非空
- **THEN** 保留配置供后续开启使用，当前不加载或调用该模型，请求关闭能力时沿用准入前错误

### Requirement: 运维状态必须区分配置值和实际后端
`GET /get_status` 或等价输出 MUST 为 CAM++ 和 emotion2vec 按实例展示配置设备、实际后端、实际计算精度和加载/失败原因。新增信息 MUST 保留既有音频准入、模型池及 VAD 状态语义，查询 MUST 不触发模型加载。

#### Scenario: 查看 CUDA TensorRT 实例
- **WHEN** CAM++ 的某 NPU 实例已成功加载
- **THEN** 可查询 configured_device、effective_backend=torch_npu、实际精度及健康状态

#### Scenario: 查看未生效的 TensorRT 配置
- **WHEN** 某实例设备为 npu:0 且配置中保留了已废弃的 TensorRT 开关
- **THEN** 该实例显示 effective_backend=torch_npu，不把已废弃的配置精度当作 NPU 实际精度

#### Scenario: 区分整音频等待和模型池等待
- **WHEN** B 类音频等待准入，同时已准入音频的 CAM++ 段请求等待模型实例
- **THEN** spkEmoConcurrency.waiting 与 CAM++ 模型池 waiting 分别表示整音频和段请求，不混合计数或改变容量

#### Scenario: 后端与资产失败
- **WHEN** 目标 NPU 设备不可用或权重加载失败
- **THEN** 状态包含对应实例失败原因，日志可关联模型、设备与权重路径，不显示后端已成功生效，也不自动回退 CPU

### Requirement: 后端文档和性能报告必须反映实际配置
模型说明及运维文档 MUST 说明设备优先级、实际后端、NPU 设备分配和容量口径。性能报告 MUST 记录总/B/段并发、设备数组和 A/B 请求比例，区分模型推理、模型池等待、音频准入、转写推理与端到端指标。

#### Scenario: 默认 B 上限的性能报告
- **WHEN** 使用 audio_concurrency=8、audio_spk_emo_concurrency=4 比较后端
- **THEN** 报告包含准入/拒绝/等待结果，不能以两组不同 B 上限的结果计算加速倍数

#### Scenario: 非 PyTorch 分配内存
- **WHEN** NPU 运行时使用未被 PyTorch 分配器统计的设备内存
- **THEN** 报告明确统计口径和可获得的设备/分配指标；无法获取时说明原因，不以 PyTorch 计数为 0 宣称无占用

### Requirement: 八路完整音频报告必须提供可复算的 P50 和 P95
八路验收 MUST 使用同一份按完整源文件 SHA256 去重的八条完整音频清单，不截取短片或将同源副本、重复请求当成独立音频。本轮对比维度 MUST 为同一 torch_npu 原生后端在 segment/request 闸门 g1=1/1 与 g2=2/2 下的表现，不要求已退役的 PyTorch CUDA/TensorRT 三后端对比。除闸门外 MUST 保持相同样本、VAD 参数、模型权重/实际精度、设备映射、Qwen 单副本与 CAM++ 三实例/emo 一实例；总音频上限8、B上限4，以4B+4A混合请求执行。MUST 明确标注冷启动批次或预热后批次及模型加载是否计入；如有预热，须独立记录且不混入稳态分位数，不得把含首次加载的端到端时延冒充暖态。报告记录实测 A/B 活动峰值、共享卡其它进程与外负载，不能把发送八个请求等同于实际同时活动八个。无法控制的共享干扰 MUST 明示，不声称 NVIDIA 对比、独占峰值或单因素因果加速。

#### Scenario: 八个成功完整请求的分位数
- **WHEN** 某档位的一轮八个完整请求全部成功完成
- **THEN** 从逐请求单调时钟的客户端端到端耗时计算秒单位 P50/P95，起点为该请求上传/HTTP调用，终点为完整读取响应；报告 N=8、线性插值方法与小样本限制，不以单段、原音频时长或整批墙钟代替。重复轮次分别报告，合并统计另列轮数和 N

#### Scenario: 测量清单与复算证据
- **WHEN** 交付 g1 或 g2 的八路测量结果
- **THEN** 保留每个样本的稳定ID、原路径编码、完整时长/大小/SHA256与验证方式、请求能力、配置/环境、原始客户端端到端 `elapsed_s`、计时边界和可获得的时间戳、RTF、VAD段数、状态及适用的增强调用/跳过/失败/null情况；记录实际总/B活动峰值、整批墙钟、成功数/8、音频吞吐、HBM/host资源与共享卡干扰。逐请求原始 `elapsed_s` 与计算方法可独立复算；若缺少单独请求start/end字段，MUST 明示并结合ownership及采样的批次时间证据说明范围，不追造原始时间戳。不能用重复短片暖标定替代完整清单

#### Scenario: 八路批次存在失败或拒绝
- **WHEN** 八个请求中任一被拒绝、超时、异常或未完成
- **THEN** 完整列出错误并标记该轮不完整，不丢弃失败或补测后仍称原轮八个完整成功请求；成功子集另列时明确样本数。每批次统一关闭 owned 服务与worker，等待在途工作结束并检查逐worker退出、无孤儿/剩余监听；共享既有服务保留，关闭超时或强制处置不得被改写为验收通过

### Requirement: 提供可复现的 v3 Docker 部署入口
Docker 部署 MUST 在 docker/ 下提供 README.md、Dockerfile 和 docker-compose.yaml；文档 MUST 给出构建、启动、健康检查、停止和配置挂载命令，并说明 v3 路由、昇腾设备绑定方式及模型内置策略。

#### Scenario: 查看部署目录
- **WHEN** 运维只查看 docker/ 目录
- **THEN** 可以获得构建镜像、启动 NPU 容器、挂载配置和验证服务状态所需的完整说明

#### Scenario: compose 启动服务
- **WHEN** 运维使用 docker-compose.yaml 启动服务并提供昇腾容器运行时
- **THEN** 容器声明 NPU 设备使用、特权模式、IPC 共享、配置文件挂载、服务端口和 v3 镜像入口，服务可通过健康检查确认就绪；设备绑定参数位于 compose 或外部 docker run 入口，容器内 start.sh 仅负责环境、预检和服务进程管理

### Requirement: 镜像标签符合 v3 时间格式
部署构建 MUST 生成或接受 asr_offline:v3_<时间> 格式的镜像标签；compose 默认引用该格式的镜像变量，文档 MUST 说明如何覆盖标签。

#### Scenario: 构建并启动时间标签镜像
- **WHEN** 运维使用 asr_offline:v3_ 加时间标识构建镜像并启动 compose
- **THEN** compose 使用同一 tag 启动，状态和日志可关联到该构建

#### Scenario: 标签格式错误
- **WHEN** 镜像 tag 不符合 asr_offline:v3_<时间> 且未显式覆盖部署变量
- **THEN** 构建或启动说明明确提示格式错误，不把旧 v2 tag 当作默认镜像

### Requirement: 启动前校验部署前置条件
昇腾部署 MUST 在就绪前校验设备可见性、必需资产、torch_npu运行时及CONFIG_PATH可读性，失败明确报告并阻止就绪。三个必需模型资产与Qwen权重策略分开：preload=false时Qwen缺失在转写触发按4011处理，不为预检加载；正式默认preload=true时全部N份权重及processor须顺序加载成功且健康后才就绪，任何部分启动失败统一回收。前置校验本身不得加载模型或绕过功能开关；校验通过后，HTTP就绪前须完成已开启且非空数组的增强池预加载尝试，失败按增强池降级契约处理。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用NPU、必需模型目录缺失或配置不可读
- **THEN** 启动指出具体前置条件并失败，不接受请求；默认Qwen预加载失败同样阻止就绪并回收已启动owner，显式懒加载的Qwen缺失仍按原4011处理

#### Scenario: 前置条件通过
- **WHEN** 设备、目录和配置检查通过，且按当前preload策略完成必要加载
- **THEN** 服务就绪并报告镜像tag、实际配置路径、后端和N份实际健康加载状态；不把预检、READY或模型名当真实语音forward暖身

### Requirement: 配置挂载和镜像模型路径可观测
运维文档和状态输出 MUST 区分挂载配置路径与镜像内模型路径，说明配置挂载只覆盖运行参数；默认三个本地模型仍来自 /app/model/，不得要求将这三个模型目录作为宿主机卷挂载。Qwen 权重目录可另行挂载，MUST 与三个必需模型的镜像内置策略区分。

#### Scenario: 挂载自定义配置
- **WHEN** compose 把宿主机配置挂载到容器并设置 CONFIG_PATH
- **THEN** 日志或状态可确认实际配置路径，三个内置本地模型加载路径仍为镜像内 /app/model/，Qwen 路径由独立配置说明

### Requirement: 性能报告保留业务计时字段并区分内部测量
后端迁移 MUST 保留 v3 既有 `gpu_time_ms` 的字段名、计时边界和统计语义，不因 NPU 迁移重命名或改写其含义。设备同步执行、排队、预处理、编译、批等待和端到端测量 MUST 在内部遥测中分开说明口径，不能静默替代既有业务计时。

#### Scenario: 比较后端延迟
- **WHEN** 生成原生或候选后端性能报告
- **THEN** 同时保留原业务计时契约，并明确新增内部指标边界、设备同步方法、冷/稳态条件及样本和配置，避免以不同计时口径计算加速比

### Requirement: 交付文件准备与部署执行分阶段跟踪
本 change MUST 准备可复现的昇腾构建、环境、启动配置及说明；依赖安装和环境校验 MUST 在后续实施阶段先行完成，实际镜像构建和上线 MUST 在完整服务稳定后按阶段执行。文件存在 MUST NOT 被表述为镜像已构建或已上线。

#### Scenario: 规划或交付文件完成
- **WHEN** 已修改 Dockerfile、compose、start.sh 或依赖清单但尚未执行构建上线
- **THEN** 报告明确仅完成文件准备，构建、运行时检查和上线验收仍待执行，不勾选对应执行结果

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
