# Spec Delta

## ADDED Requirements

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
