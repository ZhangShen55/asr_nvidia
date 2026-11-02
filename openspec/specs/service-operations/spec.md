# service-operations Specification

## Purpose
定义离线语音服务在运行期的可观测性与配置契约：服务日志的保留策略、状态查询接口的字段集合、实例数量的语义，以及配置文件中实际生效的配置项清单，避免无效配置长期沉积。

## Requirements

### Requirement: 限制服务日志的保留规模
系统 MUST 将运行日志写入配置的日志文件路径，并 MUST 对日志做保留控制：日志文件按天切分，单个日志文件大小 MUST NOT 超过配置上限（默认 100 MB），超过保留天数的日志文件（默认 7 天）MUST 被清理。清理 MUST 在服务启动时与写入过程中均生效。

#### Scenario: 日志文件超过大小上限
- **当** 当前日志文件增长到配置的大小上限
- **则** 系统切分出新的日志文件继续写入，旧文件按保留策略处理

#### Scenario: 清理过期日志
- **当** 存在超过保留天数的日志文件
- **则** 系统在启动时与写入过程中删除这些过期文件

#### Scenario: 配置缺省
- **当** 配置文件中缺少日志保留相关配置项
- **则** 系统按保留 7 天、单文件上限 100 MB 运行

### Requirement: 提供稳定的状态查询契约
`GET /get_status` MUST 返回服务标识、状态、版本、运行时长与引擎标识等运维字段，其中版本 MUST 固定为 `latest`；引擎标识字段 MUST 保留在响应中并返回固定占位值，MUST NOT 依赖部署方单独配置。

状态响应 MUST 将 VAD 状态集中放在 `vadPool`，`localModels` 只返回 CAM++ 与 emotion2vec；固定设备数组不应返回无意义的 `pool_size` 或 `pool_max` 字段。

#### Scenario: 查询服务状态
- **当** 客户端调用 `GET /get_status`
- **则** 响应包含 `status`、`appVersion`（取值 `latest`）、运行时长字段与引擎标识字段（返回固定占位值）

#### Scenario: 部署配置缺少引擎标识项
- **当** 部署配置中不再提供引擎标识配置项
- **则** 状态接口仍返回引擎标识字段，取值为固定占位值，运维平台的字段结构保持不变

### Requirement: 明确实例数量的语义
系统 MUST 支持通过配置设定同一容器内并行运行的服务实例数量（`instance_count`），该数量 MUST 影响实例进程数量，而每个实例 MUST 独立持有请求处理槽位与本地模型加载能力；该配置项的默认值 MUST 为 1。

#### Scenario: 配置多个实例
- **当** 配置中 `instance_count` 大于 1
- **则** 容器内并行运行对应数量的服务实例，请求按负载均衡方式分发到各实例

#### Scenario: 配置缺省
- **当** 配置文件中缺少实例数量配置
- **则** 系统按默认实例数量 1 运行

### Requirement: 维护配置项清单
系统 MUST 只消费配置文件中实际生效的配置项；无业务含义或不再被任何功能读取的配置项 MUST NOT 保留在配置文件中。配置项的增删 MUST 与功能开关保持一致，功能下线时其专属配置项 MUST 一并移除。VAD 动态扩缩容字段和旧模型设备结构属于本次下线配置，必须从正式默认配置中删除。

#### Scenario: 配置文件包含已下线功能的配置项
- **WHEN** 部署配置中残留 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s` 或 `vad_call_timeout_s`
- **THEN** 服务不使用这些字段改变固定 VAD 数组，并记录迁移提示或配置校验告警

#### Scenario: 配置文件包含旧模型设备结构
- **WHEN** 部署配置中残留 `[model_device]` 或 `[model_device.local_model_pool]`
- **THEN** 服务不使用旧结构覆盖 `[model_device_pool]`，并提示运维迁移到新的设备数组结构

#### Scenario: 缺少可选配置项
- **WHEN** 配置文件缺少某个仍有效的可选配置项
- **THEN** 服务使用该项声明的默认值运行，不因删除无效字段而改变其它能力

### Requirement: 启动时固定预热 VAD 实例
服务 MUST 在启动阶段按 `qwen_asr.vad_device` 数组预热全部 VAD worker，并在启动完成前报告每个 worker 的设备、成功或失败状态；运行期间 MUST 不因等待队列自动增加或回收 VAD worker。

#### Scenario: 默认混合设备配置
- **WHEN** 使用默认 8 元素 VAD 设备数组启动服务
- **THEN** 服务预热 4 个 CPU worker 和 4 个 `cuda:0` worker，状态接口显示 8 个配置实例

#### Scenario: 启动时设备不可用
- **WHEN** VAD 设备数组中的 CUDA 设备不可用
- **THEN** 服务记录对应 worker 的失败原因；若没有任何成功 worker，则拒绝完成服务启动

### Requirement: 默认配置体现有效并发来源
默认配置 MUST 使用 `vad_device` 数组表达 VAD 实例设备，使用 `[model_device_pool]` 表达 CAM++/emotion2vec 实例设备，使用 `[qwen_asr].audio_concurrency` 表达总音频准入上限、`audio_spk_emo_concurrency` 表达增强音频准入上限，并 MUST 不包含已删除的 VAD 动态池字段和旧模型设备结构。音频准入配置 MUST NOT 被解释为模型实例数量。

#### Scenario: 检查默认配置
- **WHEN** 运维读取 `config.toml` 或 `config.docker.toml`
- **THEN** 能直接通过 VAD 数组长度、模型设备数组长度计算配置实例数，并单独查看总音频和增强音频准入上限；无需额外配置模型推理槽位

#### Scenario: 功能关闭时模型池不生效
- **WHEN** `open_v2_role=false` 或 `open_v2_emotion=false`
- **THEN** 对应模型设备数组不会触发模型加载或推理，但配置仍可保留用于重启后开启功能

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
服务运维工具 MUST 支持停止新请求、等待在途请求回收并优雅停止 SeaCraftASR；测试收尾 MUST 不自动停止或重启独立运行的 Qwen-ASR 服务。

#### Scenario: 收到测试终止信号
- **WHEN** 资源安全闸门触发或运维结束测试
- **THEN** 工具停止提交新请求，等待可回收任务并记录后端、VAD worker、缓存和端口最终状态

#### Scenario: Qwen 服务保持运行
- **WHEN** SeaCraftASR 测试收尾完成
- **THEN** Qwen-ASR 的 8806 端口和服务进程保持运行，诊断报告记录其健康检查结果

### Requirement: 增强音频并发配置具有确定默认值
系统 MUST 从 `[qwen_asr].audio_spk_emo_concurrency` 读取 B 类整音频并发上限，缺省值 MUST 为 `4`，并在本地及容器配置模板展示该项。该值 MUST 是正整数，非法值 MUST 在服务就绪前明确报错；关闭能力 MUST 使用功能开关。配置修改 MUST 重启生效，且每个服务实例独立执行两类音频限制。

#### Scenario: 缺少新增配置项
- **WHEN** 旧配置未声明 `audio_spk_emo_concurrency`
- **THEN** 服务使用增强上限 4，不自动回退为总音频上限；文档说明升级后增强并发受到新增默认上限约束

#### Scenario: 显式配置其他上限
- **WHEN** 运维配置 `audio_concurrency=8`、`audio_spk_emo_concurrency=2` 并重启
- **THEN** 每个实例最多同时处理 8 个音频，其中最多 2 个为 B

#### Scenario: 非法增强上限
- **WHEN** 新增配置值为 0、负数、布尔值、浮点数或字符串
- **THEN** 服务在就绪前报告字段名称和正整数约束，不静默纠正为 1 或把 0 当作关闭功能

#### Scenario: 增强上限大于总上限
- **WHEN** 总上限为 2、增强上限为 4
- **THEN** 配置有效，两个限制同时生效使 B 实际至多为 2；状态仍如实展示配置值 2 和 4，不静默改写增强配置

#### Scenario: 多服务实例
- **WHEN** 同一部署启动多个服务实例
- **THEN** 每个实例独立维护总音频及增强音频容量，文档说明该限制不是跨实例的全局限制

### Requirement: 暴露音频分类准入状态
`GET /get_status` MUST 以附加字段 `audioConcurrency` 展示 `configured_total`、`active_total`、`waiting_total`，以 `spkEmoConcurrency` 展示 `configured`、`active`、`waiting`。两组值 MUST 来自同一次准入状态快照，可由总数减 B 数得到 A 数；查询 MUST NOT 触发本地模型加载，既有业务、模型池及诊断字段语义 MUST 保持兼容。

#### Scenario: 查询混合活动及等待状态
- **WHEN** 总上限 8、增强上限 4、正在处理 `5A+3B`、等待 1 个 A 和 2 个 B
- **THEN** `audioConcurrency` 返回 `{"configured_total":8,"active_total":8,"waiting_total":3}`，`spkEmoConcurrency` 返回 `{"configured":4,"active":3,"waiting":2}`，可计算活动 A 为 5、等待 A 为 1

#### Scenario: 首次业务请求前查询
- **WHEN** 服务已启动但尚无音频请求，运维查询状态
- **THEN** 两组配置值如实返回，活动和等待计数均为 0，CAM++ 和 emotion2vec 不因查询而加载

#### Scenario: 功能关闭并拒绝增强请求
- **WHEN** 两种增强能力均关闭且增强请求被准入前校验拒绝
- **THEN** 状态中的增强活动数和等待数保持为 0，增强配置值仍正常展示

### Requirement: 分类准入记录有界且可区分错误原因
系统 MUST 为 v2 请求记录 A/B 分类、音频准入等待耗时、准入时的总容量与 B 容量快照及有限集合的失败原因。日志和遥测 MUST 能分别统计总容量满、增强容量满、共享等待队列满、等待超时及能力关闭；遥测 MUST 保持有界存储，不保存音频、转写正文或模型对象，且保留既有阶段字段的语义。

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
README 和部署文档 MUST 说明 A/B 分类、两类容量关系、默认值、缺省迁移、等待调度和按进程生效的范围；错误码清单 MUST 区分 `4013`、`4015`、`4016`、`4017`、`4018`。文档 MUST 区分配置关闭的请求级错误、模型池无可用实例和已开启模型的段级推理失败。

#### Scenario: 查阅默认并发示例
- **WHEN** 用户查阅总上限 8、增强上限 4 的说明
- **THEN** 文档包含 `3A+4B` 时新 B 返回 `4017` 且新 A 可处理，以及 `5A+3B`、`8A` 时新 A/B 均返回 `4013` 的示例

#### Scenario: 查阅能力关闭行为
- **WHEN** 用户查询两个功能开关为 `false`、请求增强参数为 `true` 的处理规则
- **THEN** 文档说明请求失败及单项或组合错误码，不再承诺在配置关闭时以 `null` 字段正常转写；普通 A 请求仍可正常处理

### Requirement: 四个 TensorRT 参数使用正式模型池配置表
系统 MUST 在 `[model_device_pool]` 提供 `speech_campplus_use_tensorrt`、`speech_campplus_tensorrt_precision`、`emotion2vec_use_tensorrt`、`emotion2vec_tensorrt_precision`。两个开关默认 MUST 为 false，两个精度默认 MUST 为 fp16；两个模型可以分别启用和回滚，不得以旧设备配置表作为这些参数的正式入口。

#### Scenario: 缺省 TensorRT 配置
- **WHEN** `[model_device_pool]` 未声明四个新参数
- **THEN** 两个模型均不使用 TensorRT，配置的 engine 精度缺省为 fp16，现有设备数组仍决定各自实例数

#### Scenario: 仅启用一个模型
- **WHEN** speech_campplus_use_tensorrt=true 且 emotion2vec_use_tensorrt=false
- **THEN** 仅 CAM++ 的 CUDA 实例加载 TensorRT，emotion2vec 按其设备使用 PyTorch 或 torch_npu，反向配置亦独立生效

#### Scenario: 本地与容器配置
- **WHEN** 运维读取本地或容器配置模板
- **THEN** 四个参数与两个设备数组处于同一 `[model_device_pool]` 表，开关为 false、精度为 fp16，文档说明重启后生效

### Requirement: 配置类型和精度必须严格校验
四个新参数 MUST 在服务就绪前进行类型及枚举校验：开关只接受 TOML 布尔值，精度只接受字符串 fp16 或 fp32。非法配置 MUST 报告完整字段路径及合法取值，不得隐式转换或静默纠正。

#### Scenario: 非法开关类型
- **WHEN** use_tensorrt 字段为字符串、整数或其他非布尔类型
- **THEN** 启动校验失败并报告对应 `[model_device_pool]` 字段，不将字符串 false 作为启用

#### Scenario: 非法 engine 精度
- **WHEN** tensorrt_precision 为 fp32/fp16 之外的值或非字符串
- **THEN** 启动校验失败，报告对应字段及支持值，不启用 INT8 或静默改用 fp16

#### Scenario: 显式选择 FP32
- **WHEN** emotion2vec_tensorrt_precision=fp32 且其 CUDA TensorRT 开关为 true
- **THEN** 服务选择该模型 FP32 engine，不因历史固定半精度策略改成 FP16

### Requirement: 明确实例设备与 TensorRT 配置的优先级
模型设备数组 MUST 接受 cpu、cuda:N、npu:N；TensorRT 开关和 engine 精度只影响对应模型的 CUDA 实例。NPU MUST 固定走 torch_npu，CPU MUST 使用 PyTorch CPU；功能开关、请求能力与设备数组容量的既有含义 MUST 保留。

#### Scenario: NPU 配置同时打开开关
- **WHEN** `[model_device_pool]` 中某实例为 npu:0 且对应 use_tensorrt=true
- **THEN** 该实例实际后端为 torch_npu，不要求 NVIDIA engine，状态明确 TensorRT 配置被忽略

#### Scenario: 重复 CUDA 设备
- **WHEN** CAM++ 数组为三个 cuda:0，且其 TensorRT 开关为 true
- **THEN** 配置容量仍为 3 个独立实例，不因共用同一 engine 文件变成单实例

#### Scenario: 功能关闭时的配置保留
- **WHEN** 对应 features 功能开关关闭但 TensorRT 开关为 true
- **THEN** 保留配置供后续开启使用，当前不加载或调用该模型，请求关闭能力时沿用准入前错误

### Requirement: 运维状态必须区分配置值和实际后端
`GET /get_status` 或等价输出 MUST 为 CAM++ 和 emotion2vec 按实例展示配置设备、实际后端、配置 engine 精度、实际精度、engine 标识/兼容状态和加载/失败原因。新增信息 MUST 保留既有音频准入、模型池及 VAD 状态语义，查询 MUST 不触发模型加载。

#### Scenario: 查看 CUDA TensorRT 实例
- **WHEN** CAM++ 的某 CUDA 实例已成功加载 engine
- **THEN** 可查询 configured_device、effective_backend=tensorrt、实际精度、engine 校验信息及健康状态

#### Scenario: 查看未生效的 TensorRT 配置
- **WHEN** emotion2vec 配置 use_tensorrt=true，但某实例设备为 npu:0
- **THEN** 该实例显示 effective_backend=torch_npu，不把 engine 的配置精度当作 NPU 实际精度

#### Scenario: 区分整音频等待和模型池等待
- **WHEN** B 类音频等待准入，同时已准入音频的 CAM++ 段请求等待模型实例
- **THEN** spkEmoConcurrency.waiting 与 CAM++ 模型池 waiting 分别表示整音频和段请求，不混合计数或改变容量

#### Scenario: 后端与资产失败
- **WHEN** TensorRT engine 缺失或不兼容
- **THEN** 状态包含对应实例失败原因，日志可关联模型、设备和资产，不显示 TensorRT 已成功生效，也不自动回退 PyTorch

### Requirement: 后端文档和性能报告必须反映实际配置
模型说明及运维文档 MUST 说明四个参数、设备优先级、资产位置、兼容范围和显式回滚。性能报告 MUST 记录总/B/段/Qwen 并发、设备数组和 A/B 请求比例，区分模型推理、模型池等待、音频准入、Qwen 与端到端指标。

#### Scenario: 默认 B 上限的性能报告
- **WHEN** 使用 audio_concurrency=8、audio_spk_emo_concurrency=4 比较后端
- **THEN** 报告包含准入/拒绝/等待结果，不能以两组不同 B 上限的结果计算 TensorRT 加速倍数

#### Scenario: 非 PyTorch 分配内存
- **WHEN** TensorRT 使用未被 PyTorch 分配器统计的显存
- **THEN** 报告明确统计口径和可获得的设备/分配指标；无法获取时说明原因，不以 PyTorch 计数为 0 宣称无占用

### Requirement: 八路完整音频报告必须提供可复算的 P50 和 P95
八门不同课程的完整音频验收 MUST 使用设计中固定样本及 CAM++ 3 实例、emotion2vec 1 实例、`showSpk=true`、`showEmotion=true` 的配置；两个能力 MUST 开启，测试总音频/B 上限 MUST 均为 8。报告 MUST 分别列出 PyTorch、TensorRT FP32、TensorRT FP16 的测量，保持相同样本、VAD、段/Qwen 并发、设备映射与外部负载条件，保留 PyTorch 原有精度策略。冷加载/预热 MUST 与测量批次区分，不能仅由某组承担首次加载。

#### Scenario: 八个成功完整请求的分位数
- **WHEN** 一轮八路测量全部成功完成
- **THEN** 使用逐请求单调时钟记录的客户端端到端耗时计算 P50/P95，起点为发起该请求上传/HTTP 调用，终点为完整读取最终响应；报告秒单位、N=8、线性插值方法及小样本限制，不以单段耗时、音频原始时长或整批墙钟耗时替代

#### Scenario: 测量清单与复算证据
- **WHEN** 交付任一后端/精度的八路测量报告
- **THEN** 保存样本路径/时长/大小/SHA256、配置与环境及原始逐请求测量；逐音频列出请求标识、开始/结束时间、耗时、RTF、VAD 段数和状态，记录实际模型调用、跳过/失败/null、活动 B 峰值，另列整批墙钟耗时、成功数/8、吞吐与资源指标；重复轮次分别报告，合并统计单列轮数和 N

#### Scenario: 八路批次存在失败或拒绝
- **WHEN** 八个请求中任意一个被拒绝、超时、异常或未完成
- **THEN** 报告完整列出错误并标记该轮验收不完整，不自动重试或丢弃失败后仍宣称得到八个完整音频的 P50/P95；成功子集若另列统计必须明确其样本数
