# Spec Delta

## MODIFIED Requirements

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
昇腾 NPU 部署 MUST 在服务就绪前校验 NPU 设备可见性、模型目录完整性、torch_npu 运行时可用性和 CONFIG_PATH 可读性；任一必需条件失败 MUST 明确报告并阻止就绪。三个必需本地模型资产与懒加载 Qwen 权重 MUST 区分：缺少 Qwen 权重在触发转写时按 4011 处理，不能为预检强制加载 Qwen 或改变懒加载契约。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用 NPU、必需模型目录缺失或配置文件不可读
- **THEN** 启动日志指出具体前置条件并以失败状态结束，不接受请求

#### Scenario: 前置条件通过
- **WHEN** NPU 设备、模型目录和配置检查均通过
- **THEN** 服务启动 v3 路由并报告镜像 tag、实际配置路径和 torch_npu 后端状态

### Requirement: 配置挂载和镜像模型路径可观测
运维文档和状态输出 MUST 区分挂载配置路径与镜像内模型路径，说明配置挂载只覆盖运行参数；默认三个本地模型仍来自 /app/model/，不得要求将这三个模型目录作为宿主机卷挂载。Qwen 权重目录可另行挂载，MUST 与三个必需模型的镜像内置策略区分。

#### Scenario: 挂载自定义配置
- **WHEN** compose 把宿主机配置挂载到容器并设置 CONFIG_PATH
- **THEN** 日志或状态可确认实际配置路径，三个内置本地模型加载路径仍为镜像内 /app/model/，Qwen 路径由独立配置说明

### Requirement: 启动时固定预热 VAD 实例
服务 MUST 仅按 `[vad].vad_device` 数组在启动阶段以 spawn 创建固定 VAD worker 池，数组每个元素对应一个独立 worker，按 FIFO 分派任务；旧 `[qwen_asr].vad_device` MUST NOT 作为配置回退。启动完成前 MUST 记录每个 worker 的设备及预热成功或失败状态，无可用 worker 时 MUST 拒绝就绪。运行期 MUST NOT 因等待队列自动扩缩容；NPU 预热失败 MUST NOT 静默改走 CPU。

#### Scenario: 默认混合设备配置
- **WHEN** 使用本轮宿主默认 `[vad].vad_device=["cpu","cpu","npu:7","npu:7"]` 启动服务
- **THEN** 固定池含两个 CPU 和两个物理7号卡 NPU worker；仅映射物理7号卡的容器模板按实际可见映射使用 `["cpu","cpu","npu:0","npu:0"]`，状态按四个配置实例和各自实际预热结果报告

#### Scenario: 启动时设备不可用
- **WHEN** VAD 数组中的某目标设备不可用或对应 worker 预热失败
- **THEN** 服务记录该 worker 的设备和失败原因，不将其计入健康容量；无成功 worker 时拒绝完成服务启动，不以动态扩容或 CPU 回退掩盖失败

### Requirement: 支持安全的测试收尾
运维工具 MUST 支持停止接收新请求、等待已启动设备工作及结果读取完成，并关闭本次独立启动且持有生命周期的服务、worker、代理和测试容器。进程内 Qwen MUST 随所属服务统一收尾，MUST NOT 要求存在外部 Qwen 服务或保留固定外部转写端口。资源身份 MUST 在启动时登记；共享卡既有服务和其它非测试资源 MUST 保留，不能按名称或端口批量终止。收尾失败或超时后即使强制回收成功也 MUST 保留失败结论，不把接口解除监听单独作为全部收尾成功的证明。

#### Scenario: 收到测试终止信号
- **WHEN** 资源安全闸门触发或运维结束测试
- **THEN** 工具停止提交新请求，等待 owned 在途任务及清理完成，关闭所属本地 Qwen、增强模型和固定 VAD worker，记录各自退出/失败/强制处置信息与进程树、缓存和监听端口最终状态

#### Scenario: Qwen 服务保持运行
- **WHEN** 服务器存在不属于本次测试生命周期的 Qwen 或其它共享服务
- **THEN** 这些既有资源保持原状；本次测试不要求独立 Qwen 服务存在，也不以某个固定外部端口仍在监听作为成功条件。所属测试服务内嵌的 Qwen 已随其服务关闭，测试接口不继续后台占用

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

### Requirement: 默认配置体现有效并发来源
默认配置 MUST 使用 `vad_device` 数组表达 VAD 实例设备，使用 `[model_device_pool]` 表达 CAM++/emotion2vec 实例设备，使用 `[qwen_asr].audio_concurrency` 表达总音频准入上限、`audio_spk_emo_concurrency` 表达增强音频准入上限，并 MUST 不包含已删除的 VAD 动态池字段和旧模型设备结构。音频准入配置 MUST NOT 被解释为模型实例数量。

#### Scenario: 检查默认配置
- **WHEN** 运维读取 `config.toml` 或 `docker/config.toml`
- **THEN** 能直接通过 VAD 数组长度、模型设备数组长度计算配置实例数，并单独查看总音频和增强音频准入上限；无需额外配置模型推理槽位

#### Scenario: 功能关闭时模型池不生效
- **WHEN** `open_v2_role=false` 或 `open_v2_emotion=false`
- **THEN** 对应模型设备数组不会触发模型加载或推理，但配置仍可保留用于重启后开启功能

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

## REMOVED Requirements

### Requirement: 四个 TensorRT 参数使用正式模型池配置表
**Reason**: `speech_campplus_use_tensorrt`、`speech_campplus_tensorrt_precision`、`emotion2vec_use_tensorrt`、`emotion2vec_tensorrt_precision` 四个参数只对 NVIDIA CUDA 实例生效，在昇腾 910B 上无对应加速路径，本分支只负责 NPU，因此整体移除。

**Migration**: 不适用。CAM++ 与 emotion2vec 的实际后端完全由 `[model_device_pool]` 的设备数组元素决定，`npu:N` 固定为 torch_npu、`cpu` 固定为 PyTorch CPU。旧配置中的四个键将被忽略并记录迁移提示，不作为兼容回退来源。

### Requirement: 配置类型和精度必须严格校验
**Reason**: 该需求校验的是被移除的四个 TensorRT 参数的类型与枚举取值，其校验对象已不存在。

**Migration**: 不适用。设备数组元素的格式校验（`cpu` / `npu:N`）继续在服务就绪前执行，非法设备标识 MUST 报告完整字段路径及合法取值。

## ADDED Requirements

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
