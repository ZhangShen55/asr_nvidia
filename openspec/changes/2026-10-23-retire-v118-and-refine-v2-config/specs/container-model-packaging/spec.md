# 规格增量

## MODIFIED Requirements

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

### Requirement: 默认容器使用容器专用配置
镜像 MUST 默认使用 `config.docker.toml` 生成 `/config.toml`，MUST NOT 将面向宿主机开发的 `config.toml` 作为默认容器配置。

#### Scenario: 启动默认镜像
- **当** 容器未设置外部配置文件
- **则** 服务使用容器专用配置中的 `cuda:0` 落卡与 `/app/model/` 模型路径

## REMOVED Requirements

### Requirement: 法语模型按开启条件校验
**Reason**: 法语转写能力随 `/v1.1.8/seacraft_asr` 一并下线，构建流程不再校验 `faster-whisper-large-v3` 目录。
