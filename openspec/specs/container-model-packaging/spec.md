# container-model-packaging Specification

## Purpose

定义离线 ASR 模型的镜像内置交付方式，使容器无需挂载模型即可运行，同时保留通过配置文件覆盖非模型参数的能力。

## Requirements

### Requirement: 随镜像交付完整模型集合
镜像 MUST 内置 `/v1.1.8/seacraft_asr` 所需的五个模型目录：Paraformer、VAD、PUNC、CAM++ 和 emotion2vec。镜像 MUST 另行支持可选的 `faster-whisper-large-v3` 目录，用于法语转写能力。

#### Scenario: 不挂载模型启动容器
- **当** 运维人员使用默认镜像和默认配置启动容器且未挂载任何模型目录
- **则** 容器能够找到五个模型并完成模型加载

#### Scenario: 模型目录缺失
- **当** 构建上下文中的 `model/` 缺少任一必需模型目录
- **则** 构建或启动验证必须明确失败并指出缺失的模型目录

### Requirement: 使用固定模型目录结构
`model/` 下 MUST 使用以下目录名：`speech_fsmn_vad_zh-cn-16k-common-pytorch`、`punc_ct-transformer_cn-en-common-vocab471067-large`、`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`、`speech_campplus_sv_zh_en_16k-common_advanced` 和 `emotion2vec_plus_large`；法语能力启用时 MUST 额外提供 `faster-whisper-large-v3`。

#### Scenario: 使用约定目录构建
- **当** 五个模型按约定目录名放入仓库根目录的 `model/`
- **则** 镜像构建无需额外重命名即可将模型复制到 `/app/model/`

#### Scenario: 提供法语模型目录
- **当** `model/faster-whisper-large-v3` 按约定目录名提供
- **则** 镜像构建将其复制到 `/app/model/faster-whisper-large-v3` 供法语转写加载

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
- **则** 服务使用容器专用配置中的 `cuda:0`、单实例和 `/app/model/` 模型路径

### Requirement: 法语模型按开启条件校验
Docker 构建 MUST 仅在默认容器配置开启多语言能力时强制要求 `model/faster-whisper-large-v3` 存在且非空；未开启时 MUST NOT 因缺少该目录而失败。

#### Scenario: 开启多语言能力但缺少模型
- **当** 默认容器配置开启多语言能力且构建上下文中缺少 `faster-whisper-large-v3`
- **则** 构建失败并明确指出缺失的模型目录

#### Scenario: 未开启多语言能力
- **当** 默认容器配置未开启多语言能力
- **则** 镜像构建在只提供五个必需模型目录的情况下继续完成

### Requirement: 镜像内置转写与规范化运行时
镜像 MUST 包含法语转写所需的 Whisper 运行时与文本规范化所需的规则包运行时，使开启相应能力时无需额外安装依赖；文本规范化依赖缺失时 MUST 按透传策略降级而不影响请求成功。

#### Scenario: 容器内启用法语能力
- **当** 容器配置开启多语言能力且镜像内存在 `faster-whisper-large-v3`
- **则** 服务启动时加载 Whisper 模型并可处理法语请求

#### Scenario: 容器内文本规范化不可用
- **当** 容器内文本规范化依赖不可用
- **则** 转写请求仍返回成功结果，文本保持 ASR 原始输出
