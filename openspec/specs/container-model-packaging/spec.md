# container-model-packaging Specification

## Purpose

定义离线 ASR 模型的镜像内置交付方式，使容器无需挂载模型即可运行，同时保留通过配置文件覆盖非模型参数的能力。

## Requirements

### Requirement: 随镜像交付完整模型集合
镜像 MUST 内置 `/v2/seacraft_asr` 所需的三个模型目录：VAD、CAM++ 和 emotion2vec。镜像 MUST NOT 要求提供已下线接口专用的 SeACo-Paraformer、PUNC 与 `faster-whisper-large-v3` 目录。

#### Scenario: 不挂载模型启动容器
- **当** 运维人员使用默认镜像和默认配置启动容器且未挂载任何模型目录
- **则** 容器能够找到三个模型并完成模型加载

#### Scenario: 模型目录缺失
- **当** 构建上下文中的 `model/` 缺少任一必需模型目录
- **则** 构建或启动验证必须明确失败并指出缺失的模型目录

#### Scenario: 已下线模型目录缺失
- **当** 构建上下文中的 `model/` 不含 SeACo-Paraformer、PUNC 或 `faster-whisper-large-v3`
- **则** 构建与启动均正常完成

### Requirement: 使用固定模型目录结构
`model/` 下 MUST 使用以下目录名：`speech_fsmn_vad_zh-cn-16k-common-pytorch`、`speech_campplus_sv_zh_en_16k-common_advanced` 和 `emotion2vec_plus_large`。

#### Scenario: 使用约定目录构建
- **当** 三个模型按约定目录名放入仓库根目录的 `model/`
- **则** 镜像构建无需额外重命名即可将模型复制到 `/app/model/`

#### Scenario: 提供法语模型目录
- **当** `model/` 下仍存在历史遗留的 `faster-whisper-large-v3` 目录
- **则** 镜像构建忽略该目录，不将其复制到 `/app/model/`，法语转写能力已随 `/v1.1.8/seacraft_asr` 下线

### Requirement: 容器默认使用镜像内模型路径
默认容器配置中的五个模型路径 MUST 指向 `/app/model/` 下对应目录。

#### Scenario: 检查默认容器配置
- **当** 运维人员查看镜像内置的 `/config.toml`
- **则** 所有保留模型路径均位于 `/app/model/` 且不存在 `/var/model_zoo` 依赖

### Requirement: 允许挂载配置但不要求挂载模型
容器 MUST 继续支持通过 `CONFIG_PATH` 和配置文件挂载修改并发、设备及其他运行参数，但模型默认继续使用镜像中的副本。

#### Scenario: 挂载调整并发参数的配置
- **当** 运维人员挂载包含新并发或设备参数且仍使用 `/app/model/` 模型路径的配置
- **则** 服务使用挂载配置启动并读取镜像内模型

### Requirement: 模型文件不进入版本控制
模型目录中的权重和配置文件 MUST NOT 被 Git 跟踪，但仓库 MUST 保留模型放置说明。

#### Scenario: 放入模型后检查 Git 状态
- **当** 开发者将模型文件放入 `model/` 下五个目录
- **则** `git status` 不把模型文件列为待提交内容，`model/README.md` 仍可被跟踪

### Requirement: 模型目录进入 Docker 构建上下文
Docker 构建 MUST 包含 `model/` 下的权重文件，即使这些文件被 Git 忽略。

#### Scenario: 构建镜像
- **当** 开发者在本地准备好 `model/` 后执行 `docker build`
- **则** Docker 构建上下文包含所需权重，并将其复制到运行镜像

### Requirement: Docker 构建不依赖仓库外 PyArrow wheel
Docker 构建 MUST 能通过依赖清单获取与 CentOS 7 兼容的 `pyarrow==20.0.0`，MUST NOT 要求预先放置 `wheel/pyarrow-*.whl`。

#### Scenario: 干净构建上下文构建镜像
- **当** 仓库中不存在 `wheel/` 目录
- **则** Docker 依赖安装仍能获取 CP310 manylinux2014 PyArrow wheel 并继续构建

### Requirement: 默认容器使用容器专用配置
镜像 MUST 默认使用 `config.docker.toml` 生成 `/config.toml`，MUST NOT 将面向宿主机开发的 `config.toml` 作为默认容器配置。

#### Scenario: 启动默认镜像
- **当** 容器未设置外部配置文件
- **则** 服务使用容器专用配置中的 `cuda:0` 落卡与 `/app/model/` 模型路径

### Requirement: 镜像内置转写与规范化运行时
镜像 MUST 包含文本规范化所需的规则包运行时，使开启该能力时无需额外安装依赖；文本规范化依赖缺失时 MUST 按透传策略降级而不影响请求成功。镜像 MUST NOT 因内置小语种转写运行时而增加体积。

#### Scenario: 容器内文本规范化可用
- **当** 容器内文本规范化依赖可用
- **则** 转写请求的文本按配置执行规范化

#### Scenario: 容器内启用法语能力
- **当** 部署配置中残留多语言或法语相关开关
- **则** 服务忽略该开关，不加载 Whisper 模型，法语能力已随 `/v1.1.8/seacraft_asr` 下线

#### Scenario: 容器内文本规范化不可用
- **当** 容器内文本规范化依赖不可用
- **则** 转写请求仍返回成功结果，文本保持 ASR 原始输出

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
