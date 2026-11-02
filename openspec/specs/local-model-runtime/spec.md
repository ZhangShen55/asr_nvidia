# local-model-runtime Specification

## Purpose
定义 v2 链路三个本地模型（VAD、CAM++、emotion2vec）在服务进程内的承载方式与显存归属标识约定：
三个模型由同一个服务进程承载、进程名统一为 `asr-offline-model`，运维通过 `nvidia-smi` 的进程列表
即可定位本服务占用的显存；模型仍保持懒加载，未触发的模型不占显存。

## Requirements

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

系统 MUST 在状态接口中暴露三个本地模型各自的加载状态与显存占用明细，使运维在 `nvidia-smi` 只能看到聚合显存的情况下仍能区分各模型的占用；对于 CAM++ 与 emotion2vec，状态 MUST 同时包含设备数组和独立模型池的容量、忙碌、等待与失败信息。

#### Scenario: 查询模型状态
- **WHEN** 运维请求 `GET /get_status`
- **THEN** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态、设备、显存占用和各自模型池状态

#### Scenario: 模型尚未加载
- **WHEN** 某模型尚未被任何请求触发加载
- **THEN** 状态接口中该模型的加载状态为未加载且显存占用为 0，同时保留其配置实例数和设备数组

### Requirement: 实例数量与显存放大的关系可控

系统 MUST 保证服务实例数量与显存占用之间存在明确、可预期的关系：完整服务进程实例数量 MUST 由配置项 `instance_count` 决定；单个服务进程内 CAM++ 与 emotion2vec 的模型副本数量和并发容量 MUST 分别由各自设备数组决定，不能由 `instance_count` 或共享模型槽位隐式改变。

#### Scenario: 默认单实例
- **WHEN** 配置中未显式设置 `instance_count` 且两个模型设备数组各有一个元素
- **THEN** 服务以 1 个完整服务实例运行，两个模型池各有 1 个实例

#### Scenario: 配置多个实例
- **WHEN** 配置中 `instance_count` 为 2 且模型设备数组保持不变
- **THEN** 服务以 2 个进程运行，每个进程按各自设备数组加载模型，模型池比例不在进程间合并

### Requirement: 本地增强模型池按设备数组建立固定实例
系统 MUST 从 `[model_device_pool]` 的两个设备数组建立 CAM++ 与 emotion2vec 独立模型池；每个数组元素对应一个模型实例，数组长度决定配置实例数，实例在首次启用对应能力时按数组顺序加载到指定设备。

#### Scenario: CAM++ 与 emotion2vec 实例数量不同
- **WHEN** CAM++ 数组有 3 个元素、emotion2vec 数组有 1 个元素
- **THEN** 两个模型池分别最多提供 3 个和 1 个实例，不因数量不同而补齐或共享实例

#### Scenario: 多卡设备数组
- **WHEN** 同一模型数组包含多个 `cuda:N` 设备
- **THEN** 每个实例加载到对应数组元素指定的设备，并在状态接口中按实例顺序展示设备

### Requirement: 模型池配置只使用新配置结构
系统 MUST 将 `[model_device_pool]` 作为 CAM++ 与 emotion2vec 的唯一正式设备配置结构；旧 `[model_device]` 单设备字段和 `[model_device.local_model_pool]` 嵌套结构 MUST 不得覆盖新数组配置。

#### Scenario: 只存在新配置
- **WHEN** 配置仅提供 `[model_device_pool]` 及两个设备数组
- **THEN** 服务按数组建立模型池，功能开关关闭时不加载对应模型

#### Scenario: 旧配置与新配置同时存在
- **WHEN** 配置同时包含旧结构和新结构
- **THEN** 服务使用新结构，并记录可操作的迁移告警

#### Scenario: 空数组禁用模型
- **WHEN** 某模型设备数组为空
- **THEN** 该模型不创建实例；请求要求该能力时返回明确的模型不可用错误，其他能力仍可用

### Requirement: 暴露模型池配置容量与运行容量
状态接口 MUST 区分每个模型池的配置实例数、成功加载实例数、失败实例数、可用实例数、busy、waiting、设备数组和显存合计；模型池等待 MUST 可与 Qwen 和 VAD 等待区分。

#### Scenario: 查询非对称模型池状态
- **WHEN** CAM++ 配置 3 个实例且 emotion2vec 配置 1 个实例并已触发加载
- **THEN** `/get_status` 分别返回 `configured_instances=3/1`、成功实例数、设备数组、busy/waiting 和显存信息

#### Scenario: 部分实例加载失败
- **WHEN** 数组中的一个设备不可用而其它实例加载成功
- **THEN** 状态接口报告失败实例和原因，成功实例仍能服务请求

### Requirement: 模型池并发容量由各自设备数组决定

系统 MUST 将 `speech_campplus_devices` 数组中成功加载的实例数作为 CAM++ 模型池的最大并发容量，并将 `emotion2vec_devices` 数组中成功加载的实例数作为 emotion2vec 模型池的最大并发容量；两个模型池 MUST 独立排队，MUST NOT 通过固定比例或额外槽位配置绑定容量。

#### Scenario: 两个模型配置不同数量实例
- **WHEN** CAM++ 设备数组有 3 个元素且 emotion2vec 设备数组有 1 个元素
- **THEN** CAM++ 最多同时执行 3 个推理，emotion2vec 最多同时执行 1 个推理，两个池互不借用容量

#### Scenario: 配置数组为空
- **WHEN** 任一模型的设备数组为空
- **THEN** 对应模型池容量为 0、不加载该模型实例，要求该能力的请求得到明确不可用结果

#### Scenario: 部分实例加载失败
- **WHEN** 设备数组声明多个实例但其中部分实例加载失败
- **THEN** 模型池最大并发容量按成功加载的实例数计算，失败实例原因可查询且不占用可用槽位

### Requirement: 模型池状态区分独立容量与等待

系统 MUST 在状态接口中分别暴露 CAM++ 与 emotion2vec 的配置实例数、成功加载实例数、可用实例数、busy、waiting 和失败实例数；状态 MUST 能区分模型池等待与其它服务队列等待。

#### Scenario: 查询独立模型池状态
- **WHEN** CAM++ 配置 3 个实例且 emotion2vec 配置 1 个实例并均加载成功
- **THEN** 状态分别显示 `configured_instances=3` 和 `configured_instances=1`，并显示各自 busy、waiting、available_instances

#### Scenario: 一个模型池排队而另一个空闲
- **WHEN** emotion2vec 的可用实例全部忙碌而 CAM++ 仍有可用实例
- **THEN** emotion2vec 的 waiting 增加，CAM++ 的 waiting 不因 emotion2vec 排队而增加
