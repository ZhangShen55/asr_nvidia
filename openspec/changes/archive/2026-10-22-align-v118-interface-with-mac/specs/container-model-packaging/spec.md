# Spec Delta

## MODIFIED Requirements

### Requirement: 随镜像交付完整模型集合
镜像 MUST 内置 `/v1.1.8/seacraft_asr` 所需的五个模型目录：Paraformer、VAD、PUNC、CAM++ 和 emotion2vec。镜像 MUST 另行支持可选的 `faster-whisper-large-v3` 目录，用于法语转写能力。

#### Scenario: 不挂载模型启动容器
- **WHEN** 运维人员使用默认镜像和默认配置启动容器且未挂载任何模型目录
- **THEN** 容器能够找到五个模型并完成模型加载

#### Scenario: 模型目录缺失
- **WHEN** 构建上下文中的 `model/` 缺少任一必需模型目录
- **THEN** 构建或启动验证必须明确失败并指出缺失的模型目录

### Requirement: 使用固定模型目录结构
`model/` 下 MUST 使用以下目录名：`speech_fsmn_vad_zh-cn-16k-common-pytorch`、`punc_ct-transformer_cn-en-common-vocab471067-large`、`speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`、`speech_campplus_sv_zh_en_16k-common_advanced` 和 `emotion2vec_plus_large`；法语能力启用时 MUST 额外提供 `faster-whisper-large-v3`。

#### Scenario: 使用约定目录构建
- **WHEN** 五个模型按约定目录名放入仓库根目录的 `model/`
- **THEN** 镜像构建无需额外重命名即可将模型复制到 `/app/model/`

#### Scenario: 提供法语模型目录
- **WHEN** `model/faster-whisper-large-v3` 按约定目录名提供
- **THEN** 镜像构建将其复制到 `/app/model/faster-whisper-large-v3` 供法语转写加载

## ADDED Requirements

### Requirement: 法语模型按开启条件校验
Docker 构建 MUST 仅在默认容器配置开启多语言能力时强制要求 `model/faster-whisper-large-v3` 存在且非空；未开启时 MUST NOT 因缺少该目录而失败。

#### Scenario: 开启多语言能力但缺少模型
- **WHEN** 默认容器配置开启多语言能力且构建上下文中缺少 `faster-whisper-large-v3`
- **THEN** 构建失败并明确指出缺失的模型目录

#### Scenario: 未开启多语言能力
- **WHEN** 默认容器配置未开启多语言能力
- **THEN** 镜像构建在只提供五个必需模型目录的情况下继续完成

### Requirement: 镜像内置转写与规范化运行时
镜像 MUST 包含法语转写所需的 Whisper 运行时与文本规范化所需的规则包运行时，使开启相应能力时无需额外安装依赖；文本规范化依赖缺失时 MUST 按透传策略降级而不影响请求成功。

#### Scenario: 容器内启用法语能力
- **WHEN** 容器配置开启多语言能力且镜像内存在 `faster-whisper-large-v3`
- **THEN** 服务启动时加载 Whisper 模型并可处理法语请求

#### Scenario: 容器内文本规范化不可用
- **WHEN** 容器内文本规范化依赖不可用
- **THEN** 转写请求仍返回成功结果，文本保持 ASR 原始输出
