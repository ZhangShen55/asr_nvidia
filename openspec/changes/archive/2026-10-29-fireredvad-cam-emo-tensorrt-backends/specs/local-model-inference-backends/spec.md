# Spec Delta

## Purpose

定义 CAM++ 和 emotion2vec 按现有模型池实例设备选择 PyTorch、TensorRT 与 torch_npu 的行为契约，使四个独立配置项能够决定真实推理路径，并保证模型资源生命周期、后端状态、兼容检查及回滚行为可以验证和追溯。

## ADDED Requirements

### Requirement: 按实例设备和模型开关解析实际后端
系统 MUST 按 `[model_device_pool]` 两个设备数组的每个元素及对应模型的 TensorRT 开关解析实际后端：CUDA+true 使用 TensorRT，CUDA+false 使用 PyTorch CUDA，NPU 固定使用 torch_npu，CPU 使用 PyTorch CPU。NPU/CPU MUST 忽略 TensorRT 开关及 engine 精度，不得以模型池首个设备决定所有实例后端。

#### Scenario: CUDA 启用或关闭 TensorRT
- **WHEN** 某实例配置为 `cuda:0`
- **THEN** 对应模型开关为 true 时实际后端为 tensorrt，为 false 时为 pytorch_cuda

#### Scenario: NPU 忽略 TensorRT 配置
- **WHEN** 实例设备为 `npu:0` 且对应 TensorRT 开关为 true
- **THEN** 实际后端为 torch_npu，该实例不要求 NVIDIA engine 或 TensorRT 依赖，状态明确该开关未生效

#### Scenario: CPU 忽略 TensorRT 配置
- **WHEN** 实例设备为 cpu 且对应 TensorRT 开关为 true
- **THEN** 实际后端为 pytorch_cpu，该实例不加载 NVIDIA engine，也不创建加速器上下文

#### Scenario: 混合设备池
- **WHEN** 同一模型设备数组包含 CUDA、NPU 或 CPU 实例
- **THEN** 各实例分别选择后端，一个实例的设备或加载失败不改变其他实例的后端选择

### Requirement: 后端选择不得改变能力开关和模型池容量
系统 MUST 在对应功能开放且请求需要该能力时才加载模型，MUST 保留按设备数组建立实例、健康实例决定容量和独立 FIFO 排队的规则。启用 TensorRT MUST NOT 创建额外模型槽位、合并数组实例或以全局推理锁使配置容量失效。

#### Scenario: 两个模型配置非对称实例
- **WHEN** CAM++ 数组含 3 个元素、emotion2vec 数组含 1 个元素且所有实例健康
- **THEN** 两个模型分别提供最多 3 个和 1 个推理容量，无论各自采用哪种后端

#### Scenario: 功能关闭但 engine 存在
- **WHEN** 对应功能开关关闭，设备数组非空且存在 TensorRT engine
- **THEN** 请求不能绕过能力关闭而加载或调用该模型，返回既有能力错误且不占用音频名额

### Requirement: TensorRT 资产失败必须显式报告
CUDA 实例启用 TensorRT 时，系统 MUST 校验模型、权重/导出版本、I/O、精度、动态 profile、SHA256 和目标环境兼容信息。资产缺失、不兼容或初始化失败 MUST 标记对应实例不可用并记录原因，MUST NOT 静默切换到 PyTorch。

#### Scenario: 所有实例加载失败
- **WHEN** 所请求模型没有健康实例，包括 engine 缺失、不兼容或设备数组为空
- **THEN** CAM++ 返回 4015 或 emotion2vec 返回 4016，普通转写及其他健康能力仍可用

#### Scenario: 部分实例加载失败
- **WHEN** 同一模型的一部分实例失败而其余实例健康
- **THEN** 请求可使用健康实例，状态展示失败原因和实际可用容量，不把失败实例替换为 PyTorch

#### Scenario: 显式回滚
- **WHEN** 运维关闭某模型 TensorRT 开关并重启服务
- **THEN** 该模型 CUDA 实例使用 PyTorch CUDA，另一个模型的后端选择不变

### Requirement: 每个实例的推理资源独立且正确收尾
系统 MUST 隔离并发实例的可写推理状态，MUST 在设备计算、结果读取及必要清理完成后才归还模型实例和整音频租约。异步提交成功或调用线程结束 MUST NOT 被当作设备已完成；请求取消及重复取消 MUST 保持资源所有权直到实际工作收尾。

#### Scenario: 同时处理不同长度输入
- **WHEN** 多个 TensorRT 实例同时推理不同长度的 VAD 段
- **THEN** 动态 shape、执行状态和可写输入输出缓冲互不覆盖，结果对应各自输入且模型池保持配置容量

#### Scenario: 提交返回但设备仍忙
- **WHEN** TensorRT 异步工作已提交而对应设备尚未完成
- **THEN** 模型实例仍 busy，相关音频槽位仍被持有，其他请求不能复用其可写推理资源

#### Scenario: 取消与重复取消
- **WHEN** 请求取消时已启动线程或设备工作，收尾中再次收到取消
- **THEN** 系统等待必要工作及清理完成后准确归还资源，再传播取消，最终计数归零且无重复释放

### Requirement: 输入输出边界必须保持模型和业务语义
系统 MUST 按经验证的 engine I/O 和 profile 准备输入与读取完整输出；FP16 engine 的 FP32 I/O 不得被错误转换。系统 MUST 保留不足 1 秒跳过 CAM++、emotion2vec 在 1～10 秒外返回既有“平淡”的规则，以及原聚类和情绪映射。非法或越界输入 MUST 显式记录，不得静默换后端或改变段边界。

#### Scenario: CAM++ 特征与输出
- **WHEN** CAM++ TensorRT 处理允许长度的 VAD 段
- **THEN** 使用与 FunASR 一致的 80 维、按段去均值 fbank，输出与现有契约一致的 192 维 embedding，不跨 VAD 段合并

#### Scenario: emotion2vec 完整分类
- **WHEN** emotion2vec TensorRT 处理 1～10 秒的片段
- **THEN** 完整输出含分类概率，标签顺序与现有业务映射一致，不以中间 representation 代替最终分类

#### Scenario: profile 越界
- **WHEN** 需要推理的输入超过所加载 engine 的 profile
- **THEN** 记录可定位失败，不擅自截断、补零、拆段或转入 PyTorch；普通段级失败与实例不可用按既有错误分层处理

### Requirement: 后端依赖和设备 API 必须隔离
系统 MUST 仅对需要的实例加载相应后端依赖，NPU 路径 MUST 使用独立 torch_npu 设备与精度接口而非 CUDA API；TensorRT 未启用的路径 MUST 不要求 TensorRT。无对应硬件时的桩验证 MUST 与实机验证区分报告。

#### Scenario: 未安装 TensorRT
- **WHEN** 环境未安装 TensorRT，模型使用 PyTorch CUDA、CPU 或 NPU
- **THEN** 模型不因无关 TensorRT 导入失败而不可用；NPU 实例不主动初始化 CUDA

#### Scenario: NPU 运行环境不可用
- **WHEN** 配置 npu:N 但 torch_npu 或对应设备不可用
- **THEN** 报告该实例不可用，不改走 TensorRT、CUDA 或 CPU，也不宣称已完成 NPU 实机验收

### Requirement: 生产资格必须经过模型与同卡业务验证
系统 MUST 在将 engine 标记为生产可用前完成数值、业务输出及同卡负载验证，结果按模型、精度、输入长度、并发配置和环境可追溯。已有独立模型实验 MUST NOT 被当作完整课堂或服务端到端验收。

#### Scenario: FP32 与 FP16 验证顺序
- **WHEN** 验证 CAM++ 或 emotion2vec TensorRT
- **THEN** 先复核 FP32，再验证 FP16；CAM++ 比较 embedding 与完整课堂聚类，emotion2vec 比较完整概率及业务标签

#### Scenario: 默认 B 上限下的八路提交
- **WHEN** B 上限为 4，同时提交 8 个增强音频请求
- **THEN** 报告区分准入、拒绝或等待，不将其描述为 8 个同时活动的 B；后端对比使用一致的容量、实例数及 Qwen 负载

#### Scenario: 八门不同课程的完整音频并发验收
- **WHEN** 从 `test/audio_file/音频文件/` 按设计中固定清单选择 8 门不同课程各一份完整 WAV，并发发起 `showSpk=true`、`showEmotion=true` 的请求
- **THEN** 必须在两个功能均开启、CAM++ 三个同卡 CUDA 实例、emotion2vec 一个同卡 CUDA 实例、总音频/B 上限均为 8 的相同测试配置下，分别完成 PyTorch、TensorRT FP32、TensorRT FP16 的完整服务端到端验证；记录八个请求的准入及处理重叠、实际活动 B 峰值、实际增强模型调用及质量结果，不以裁剪音频、同课不同片段、准入排队或仅模型推理替代该验收

### Requirement: 暴露每个实例的实际后端和健康状态
系统 MUST 暴露各实例的配置设备、实际后端、配置 engine 精度、实际计算精度、engine 兼容/校验信息及健康/失败原因。内存统计 MUST 明确口径，非 PyTorch 分配不得因 PyTorch 计数为零而被报告为没有资源占用。

#### Scenario: 配置精度与实际精度分开
- **WHEN** CPU 或 NPU 实例忽略配置的 TensorRT fp16 精度
- **THEN** 状态显示实际后端及实际精度，不把未使用的 engine 精度当作实际计算精度

#### Scenario: 实例故障
- **WHEN** 推理故障使实例执行状态无法继续安全使用
- **THEN** 状态反映实例不可用及可用容量变化，系统不把损坏实例继续作为健康实例出租
