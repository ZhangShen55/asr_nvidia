# Spec Delta

## ADDED Requirements

### Requirement: TensorRT engine 使用可校验的模型资产结构
CAM++ 与 emotion2vec 的可选 TensorRT 资产 MUST 相对各自配置模型目录存放于 `tensorrt/<precision>/engine.plan` 和 `manifest.json`，precision 为 fp32 或 fp16。资产 MUST 保留模型来源、权重/导出代码版本、I/O、profile、精度、构建时间、运行时/GPU 兼容信息及 SHA256，且 MUST 保留原始 PyTorch 权重。

#### Scenario: 两种精度同时交付
- **WHEN** 某模型同时提供 fp32 与 fp16 engine
- **THEN** 两套资产分别位于其模型目录的 tensorrt/fp32 与 tensorrt/fp16，按对应精度参数选择，不额外要求 engine 路径配置

#### Scenario: 模型目录被覆盖
- **WHEN** 运维通过 model_paths 配置或挂载替换原始模型目录
- **THEN** engine 相对替换后的模型目录查找，manifest 能核验其与当前模型权重的对应关系

#### Scenario: 实验产物转为正式交付
- **WHEN** 使用 2026-11-02 的独立实验候选 engine
- **THEN** 校验原始 SHA256 与目标兼容环境，补齐正式 manifest 所需的来源和接口信息，不把初步实验记录当作完整生产验收

### Requirement: engine 和运行依赖只对需要的后端生效
系统 MUST 仅在对应能力开启、请求触发且 CUDA 实例选择 TensorRT 时要求其 engine 和 TensorRT 运行依赖；其他实例 MUST 不因缺少无关 CUDA 资产而失败。实例加载失败 MUST 可定位且不得阻止普通转写使用其他健康能力，默认配置 MUST 保持两个 TensorRT 开关为 false。

#### Scenario: 只使用 PyTorch
- **WHEN** 两个 TensorRT 开关均为 false
- **THEN** 项目不要求 TensorRT engine 或 TensorRT 包，现有 PyTorch CUDA/CPU 路径仍可使用原始权重

#### Scenario: NPU 或 CPU 配置
- **WHEN** 实例设备为 npu:N 或 cpu，即使模型 TensorRT 开关为 true
- **THEN** 该实例不要求 NVIDIA engine，也不因缺少 TensorRT 依赖而失败

#### Scenario: 所需 engine 缺失
- **WHEN** 请求触发的 CUDA TensorRT 实例缺少 engine 或 manifest
- **THEN** 该实例加载失败并记录具体资产路径，健康实例仍可用；全部失败时返回对应模型不可用，不静默使用 PyTorch

### Requirement: 目标环境兼容信息必须可追溯
模型交付 MUST 说明每套 engine 已验证的 GPU 架构、TensorRT/CUDA 环境及输入范围；加载时 MUST 针对实际实例设备校验兼容性，不能仅凭 cuda:N 数字索引推断架构，未验证环境 MUST 不被宣称可直接复用。

#### Scenario: 使用当前已验证产物
- **WHEN** 交付本轮 RTX 4090 / SM89、TensorRT 10.7.0.post1、CUDA 12.4 候选
- **THEN** 说明明确其验证环境、CAM++ fbank profile 和 emotion2vec 1～10 秒 profile，engine 的 FP16 策略不改变 FP32 I/O

#### Scenario: engine 与实际设备不兼容
- **WHEN** 目标实例的架构或运行时不满足 manifest 兼容条件
- **THEN** 记录兼容检查失败，不加载该 engine；通过重建匹配产物或关闭对应开关并重启回滚
