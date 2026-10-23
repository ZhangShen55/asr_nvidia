# 规格增量

## Purpose

定义 v2 链路三个本地模型（VAD、CAM++、emotion2vec）在服务进程内的承载方式与显存归属标识约定：
三个模型由同一个服务进程承载、进程名统一为 `asr-offline-model`，运维通过 `nvidia-smi` 的进程列表
即可定位本服务占用的显存；模型仍保持懒加载，未触发的模型不占显存。

## ADDED Requirements

### Requirement: 本地模型由单一服务进程承载并统一命名
系统 MUST 在同一个服务进程内承载 VAD、CAM++、emotion2vec 三个本地模型，MUST NOT 为每个模型单独拉起独立进程；
承载本地模型的服务进程的进程名 MUST 为 `asr-offline-model`，且 MUST 通过覆写进程命令行首项的方式设置，
使 `nvidia-smi` 的进程列表直接显示该名称。

#### Scenario: 在 nvidia-smi 中定位服务显存
- **当** 运维在 `nvidia-smi` 的 GPU 进程列表中查看
- **则** 本服务以 `asr-offline-model` 进程名出现，其显存占用为该进程内已加载本地模型的合计占用

#### Scenario: 未拆分为多个模型进程
- **当** 服务同时加载了 VAD、CAM++ 与 emotion2vec
- **则** 三个模型处于同一进程内，`nvidia-smi` 中不出现按模型拆分的多个进程行

### Requirement: 本地模型按需懒加载
系统 MUST 支持本地模型的懒加载：对应功能开关关闭或请求未触发时，MUST NOT 加载该模型权重；
需要加载时，MUST 等待加载完成后才发起推理，模型加载失败 MUST NOT 导致整个服务无法启动。

#### Scenario: 未请求说话人能力
- **当** 请求未携带 `showSpk`
- **则** CAM++ 权重不被加载，显存中不出现 CAM++ 的权重占用

#### Scenario: 首次请求触发加载
- **当** 首次出现携带 `showSpk` 的请求
- **则** 系统加载 CAM++ 权重并完成本次推理，后续请求复用已加载的模型

#### Scenario: 模型加载失败
- **当** 模型权重加载失败
- **则** 服务保持可用，该次请求的对应能力返回失败或空值，且失败原因可被记录与查询

### Requirement: 暴露本地模型的加载状态与显存明细
系统 MUST 在状态接口中暴露三个本地模型各自的加载状态与显存占用明细，
使运维在 `nvidia-smi` 只能看到聚合显存的情况下仍能区分各模型的占用。

#### Scenario: 查询模型状态
- **当** 运维请求 `GET /get_status`
- **则** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态与各自显存占用明细

#### Scenario: 模型尚未加载
- **当** 某模型尚未被任何请求触发加载
- **则** 状态接口中该模型的加载状态为未加载且显存占用为 0

### Requirement: 实例数量与显存放大的关系可控
系统 MUST 保证服务实例数量与显存占用之间存在明确、可预期的关系：实例数量 MUST 由配置项 `instance_count` 决定，
其默认值 MUST 为 1；每个实例进程各自持有一份本地模型权重，因此实例数量为 N 时本地模型显存占用为单实例的 N 倍。

#### Scenario: 默认单实例
- **当** 配置中未显式设置 `instance_count`
- **则** 服务以 1 个实例运行，`nvidia-smi` 中只出现一行 `asr-offline-model`

#### Scenario: 配置多个实例
- **当** 配置中 `instance_count` 为 2
- **则** 服务以 2 个实例运行，`nvidia-smi` 中出现两行 `asr-offline-model`，本地模型显存占用为单实例的 2 倍
