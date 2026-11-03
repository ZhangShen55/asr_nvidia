# Spec Delta

## ADDED Requirements

### Requirement: CUDA Docker profile 强制使用 fp16 TensorRT
当服务以本次 CUDA Docker profile 启动时，CAM++ 和 emotion2vec 的 CUDA 实例 MUST 使用各自模型目录中已校验的 fp16 TensorRT engine；该 profile MUST NOT 因 engine 缺失、不兼容或初始化失败而自动改用 PyTorch。

#### Scenario: CUDA profile 后端状态
- **WHEN** 容器使用目标 NVIDIA GPU、兼容 CUDA/TensorRT 运行时和完整 fp16 资产启动
- **THEN** CAM++ 与 emotion2vec 的 CUDA 实例实际后端均为 TensorRT，状态中显示 fp16 及资产校验结果

#### Scenario: CUDA profile 资产错误
- **WHEN** 任一模型的 fp16 engine、manifest、校验或兼容检查失败
- **THEN** 对应实例标记为不可用并报告原因，服务就绪检查失败，不能回退 PyTorch

### Requirement: Docker profile 与通用后端开关隔离
CUDA Docker profile 的强制 fp16 策略 MUST 由部署 profile 明确启用并可在状态中识别；CPU、NPU 或非该 profile 的运行方式 MUST 继续遵循既有设备后端规则，不得因 Docker profile 的策略改变其后端。

#### Scenario: 非 CUDA 设备
- **WHEN** 模型实例设备为 CPU 或 NPU，即使使用同一配置模板
- **THEN** 实例按既有 PyTorch CPU 或 torch_npu 后端运行，不加载 NVIDIA TensorRT engine

#### Scenario: 非 Docker 运行
- **WHEN** 服务不以本次 CUDA Docker profile 启动
- **THEN** 既有 TensorRT 开关、精度选择、显式回滚和资产校验规则继续生效
