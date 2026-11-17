# Spec Delta

## MODIFIED Requirements

### Requirement: 按实例设备和模型开关解析实际后端
系统 MUST 按 `[model_device_pool]` 两个设备数组的每个元素解析实际后端：`npu:N` 使用 torch_npu，CPU 使用 PyTorch CPU。NPU MUST NOT 依赖任何 NVIDIA engine、TensorRT 运行时或 CUDA API，MUST NOT 以模型池首个设备决定所有实例后端。

#### Scenario: CUDA 启用或关闭 TensorRT
- **WHEN** 某实例配置为 `cuda:0`
- **THEN** 该配置在本分支不属于有效设备形态，服务 MUST 在就绪前报告设备标识非法，MUST NOT 静默改走 CPU 或 NPU

#### Scenario: NPU 忽略 TensorRT 配置
- **WHEN** 某实例配置为 `npu:0`
- **THEN** 该实例实际后端为 torch_npu，不加载 NVIDIA engine 或 TensorRT 依赖

#### Scenario: CPU 忽略 TensorRT 配置
- **WHEN** 实例设备为 cpu
- **THEN** 实际后端为 pytorch_cpu，该实例不加载 NVIDIA engine，也不创建加速器上下文

#### Scenario: 混合设备池
- **WHEN** 同一模型设备数组包含 NPU 与 CPU 实例
- **THEN** 各实例分别选择后端，一个实例的设备或加载失败不改变其他实例的后端选择

### Requirement: 后端选择不得改变能力开关和模型池容量
系统 MUST 在对应功能开放且请求需要该能力时才加载模型，MUST 保留按设备数组建立实例、健康实例决定容量和独立 FIFO 排队的规则。后端选择 MUST NOT 创建额外模型槽位、合并数组实例或以全局推理锁使配置容量失效。

#### Scenario: 两个模型配置非对称实例
- **WHEN** CAM++ 数组含 3 个元素、emotion2vec 数组含 1 个元素且所有实例健康
- **THEN** 两个模型分别提供最多 3 个和 1 个推理容量，无论各自采用哪种后端

#### Scenario: 功能关闭但 engine 存在
- **WHEN** 对应功能开关关闭且设备数组非空
- **THEN** 请求不能绕过能力关闭而加载或调用该模型，返回既有能力错误且不占用音频名额

### Requirement: 每个实例的推理资源独立且正确收尾
系统 MUST 隔离并发实例的可写推理状态，MUST 在设备计算、结果读取及必要清理完成后才归还模型实例和整音频租约。异步提交成功或调用线程结束 MUST NOT 被当作设备已完成；请求取消及重复取消 MUST 保持资源所有权直到实际工作收尾。

#### Scenario: 同时处理不同长度输入
- **WHEN** 多个 NPU 实例同时推理不同长度的 VAD 段
- **THEN** 执行状态和可写输入输出缓冲互不覆盖，结果对应各自输入且模型池保持配置容量

#### Scenario: 提交返回但设备仍忙
- **WHEN** 推理工作已提交而对应设备尚未完成
- **THEN** 模型实例仍 busy，相关音频槽位仍被持有，其他请求不能复用其可写推理资源

#### Scenario: 取消与重复取消
- **WHEN** 请求取消时已启动线程或设备工作，收尾中再次收到取消
- **THEN** 系统等待必要工作及清理完成后准确归还资源，再传播取消，最终计数归零且无重复释放

### Requirement: 输入输出边界必须保持模型和业务语义
系统 MUST 按经验证的模型输入契约准备输入与读取完整输出。系统 MUST 保留不足 1 秒跳过 CAM++、emotion2vec 在 1～10 秒外返回既有"平淡"的规则，以及原聚类和情绪映射。非法或越界输入 MUST 显式记录，不得静默改变段边界。

#### Scenario: CAM++ 特征与输出
- **WHEN** CAM++ 处理允许长度的 VAD 段
- **THEN** 使用与 FunASR 一致的 80 维、按段去均值 fbank，输出与现有契约一致的 192 维 embedding，不跨 VAD 段合并

#### Scenario: emotion2vec 完整分类
- **WHEN** emotion2vec 处理 1～10 秒的片段
- **THEN** 完整输出含分类概率，标签顺序与现有业务映射一致，不以中间 representation 代替最终分类

#### Scenario: profile 越界
- **WHEN** 需要推理的输入超出模型允许的长度范围
- **THEN** 记录可定位失败，不擅自截断、补零、拆段；普通段级失败与实例不可用按既有错误分层处理

### Requirement: 后端依赖和设备 API 必须隔离
系统 MUST 仅对需要的实例加载相应后端依赖，NPU 路径 MUST 使用独立 torch_npu 设备与精度接口而非 CUDA API。无对应硬件时的桩验证 MUST 与实机验证区分报告。

#### Scenario: 未安装 TensorRT
- **WHEN** 运行环境不含 TensorRT
- **THEN** 模型不因无关 TensorRT 导入失败而不可用；NPU 实例不主动初始化 CUDA

#### Scenario: NPU 运行环境不可用
- **WHEN** 配置 npu:N 但 torch_npu 或对应设备不可用
- **THEN** 报告该实例不可用，不改走 CUDA 或 CPU，也不宣称已完成 NPU 实机验收

### Requirement: 暴露每个实例的实际后端和健康状态
系统 MUST 暴露各实例的配置设备、实际后端、实际计算精度及健康/失败原因。内存统计 MUST 明确口径，非 PyTorch 分配不得因 PyTorch 计数为零而被报告为没有资源占用。

#### Scenario: 配置精度与实际精度分开
- **WHEN** NPU 实例不适用任何 NVIDIA engine 精度配置
- **THEN** 状态显示实际后端及实际精度，不把未使用的 engine 精度当作实际计算精度

#### Scenario: 实例故障
- **WHEN** 推理故障使实例执行状态无法继续安全使用
- **THEN** 状态反映实例不可用及可用容量变化，系统不把损坏实例继续作为健康实例出租

## REMOVED Requirements

### Requirement: TensorRT 资产失败必须显式报告
**Reason**: 本分支只负责昇腾 910B，TensorRT 是 NVIDIA 专属加速路径，在 NPU 910B 上不可用，随本地模型后端收敛为 torch_npu 一并移除。

**Migration**: 原生基线使用 torch_npu 加载 PyTorch 权重，不再要求 NVIDIA engine manifest；模型来源、revision 和文件完整性仍需追溯。实例加载失败按既有模型不可用契约上报。

### Requirement: 生产资格必须经过模型与同卡业务验证
**Reason**: 该需求针对 TensorRT engine 的 FP32/FP16 生产准入与同卡八路对比验收，是 NVIDIA 加速路径专属流程；NPU 910B 上的生产资格验证由本 change 的实机验证报告承载。

**Migration**: 不适用。迁移后在 NPU 910B 上完成 CAM++ 与 emotion2vec 相对可追溯参考的实机数值比对与业务回归，CUDA golden 缺失时明确 CPU FP32 参考及证据限制，结论写入 `docs/` 下的 NPU 验证报告，并在 `tasks.md` 中作为验收项跟踪。

### Requirement: CUDA Docker profile 强制使用 fp16 TensorRT
**Reason**: CUDA Docker profile 与 fp16 TensorRT engine 在昇腾 910B 环境中不存在，对应 profile 与资产交付要求不再适用。

**Migration**: 不适用。NPU 部署镜像统一使用 torch_npu 后端，不存在 engine 缺失导致的回退问题。

### Requirement: Docker profile 与通用后端开关隔离
**Reason**: 该需求用于隔离 CUDA Docker profile 的强制策略与通用后端开关；移除 CUDA profile 后，Docker profile 不再强制任何加速后端，后端完全由设备标识决定。

**Migration**: 不适用。昇腾部署镜像下的 NPU 实例按设备标识固定走 torch_npu，CPU 实例走 PyTorch CPU。

## ADDED Requirements

### Requirement: NPU 不可用算子必须以等价序列替换
当目标模型依赖的算子在 NPU 运行时不被支持时，系统 MUST 以数值等价的算子序列替换，并 MUST 在替换前后保持输出一致。替换 MUST NOT 通过改变模型结构、裁剪输入或降低精度来规避算子限制。

#### Scenario: CAM++ 池化算子在 NPU 上不可用
- **WHEN** CAM++ 的段池化算子在 NPU 运行时无法直接执行
- **THEN** 系统以 reshape→sum/max→expand→slice 的等价序列替换，输出仍为 192 维，数值与已记录的参考实现保持约定容差内一致

#### Scenario: 替换后数值比对
- **WHEN** 用同段音频比较 NPU 输出与可追溯 CUDA golden；若没有该基准则使用同权重 CPU FP32 参考
- **THEN** 按预先约定容差验证算子、192 维 embedding 与聚类，并记录权重、特征参数、参考设备和误差；CPU 参考可作为原生迁移数值准入依据，但 MUST 明确未验证 CUDA/NPU 对比，不得宣称该对比通过

#### Scenario: CAM++ 尾段池化数值等价
- **WHEN** 池化输入长度为 T=99/100/101/199/200/201、短输入或含负值，末段不足 100 个元素
- **THEN** 平均池化按有效元素数计数，最大池化不以补零改变负值结果；保持原 ceil_mode、扩展与截取语义，按预先明确容差比较算子输出、embedding 和聚类结果

#### Scenario: 算子适配隔离安装环境
- **WHEN** 安装环境的原始 CAM++ 算子不受 NPU 支持
- **THEN** 适配实现在项目源码内并由模型加载路径应用，不修改 conda site-packages 或原始权重

### Requirement: 后端优化须经同条件测量与业务验证
系统 MUST 以原始 checkpoint 经 torch_npu eager FP32 运行作为数值基线，MUST NOT 在没有同条件实测时宣称其为最佳性能方案。匹配目标运行时的 TorchAir 图编译、质量合格的 BF16/FP16、保持独立段语义的批处理以及 ONNX→ATC→OM MAY 作为候选。原始权重不是 ONNX 格式 MUST NOT 被用作永久排除格式转换的理由；候选上线前 MUST 明确算子支持、动态输入、数值容差、版本和可重建性，并先同步选定后端的相关规划与依赖契约。

#### Scenario: 比较图编译候选
- **WHEN** 评估 TorchAir 或其他图执行方式
- **THEN** 验证当前 torch/torch_npu/CANN 组合及动态长度，分别记录冷编译、重编译和预热后性能，并以 embedding、聚类、情绪分类和同卡业务回归作为准入条件

#### Scenario: 评估格式转换
- **WHEN** ONNX→ATC→OM 成为候选
- **THEN** 补充导出、算子与动态长度兼容、精度、交付重建及运行器资源收尾验证；转换成功本身不代表业务正确或性能最优，未选定前不替换原生交付路径
