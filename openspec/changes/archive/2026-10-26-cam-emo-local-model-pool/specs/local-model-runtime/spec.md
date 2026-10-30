# Spec Delta

## MODIFIED Requirements

### Requirement: 本地模型由单一服务进程承载并统一命名

系统 MUST 在同一个服务进程内承载 VAD、CAM++ 与 emotion2vec 的全部已加载模型副本，MUST NOT 为每个模型或模型副本单独拉起独立进程；承载本地模型的服务进程的进程名 MUST 为 `asr-offline-model`，且 MUST 通过覆写进程命令行首项的方式设置，使 `nvidia-smi` 的进程列表直接显示该名称。

#### Scenario: 在 nvidia-smi 中定位服务显存
- **WHEN** 运维在 GPU 进程列表中查看
- **THEN** 本服务以 `asr-offline-model` 进程名出现，其显存占用为该进程内已加载本地模型副本的合计占用

#### Scenario: 未拆分为多个模型进程
- **WHEN** 服务同时加载 VAD、多个 CAM++ 副本与多个 emotion2vec 副本
- **THEN** 所有模型仍处于同一服务进程内，`nvidia-smi` 中不出现按模型或副本拆分的独立进程行

### Requirement: 本地模型按需懒加载

系统 MUST 支持本地模型池的懒加载：对应功能开关关闭或请求未触发时，MUST NOT 加载该模型的任何副本；需要加载时，MUST 等待至少一个可用副本完成加载后才发起推理，单个副本加载失败 MUST NOT 导致整个服务无法启动。

#### Scenario: 未请求说话人能力
- **WHEN** 请求未携带 `showSpk`
- **THEN** CAM++ 模型池的所有副本均不被加载，显存中不出现 CAM++ 权重占用

#### Scenario: 首次请求触发加载
- **WHEN** 首次出现携带 `showSpk` 的请求且 CAM++ 配置了多个副本
- **THEN** 系统按配置创建可用副本并完成本次推理，后续请求复用已加载副本

#### Scenario: 模型加载失败
- **WHEN** 一个或多个模型副本权重加载失败
- **THEN** 服务保持可用，失败副本被记录，该次请求的对应能力在有可用副本时继续推理，否则返回失败或空值

### Requirement: 暴露本地模型的加载状态与显存明细

系统 MUST 在状态接口中暴露 VAD、CAM++ 与 emotion2vec 的模型状态和显存明细；CAM++ 与 emotion2vec 还 MUST 暴露设备数组、配置实例数、已加载实例数、可用数、失败数及忙碌或等待计数，使运维能够区分不同设备上的模型池资源占用。

#### Scenario: 查询模型状态
- **WHEN** 运维请求 `GET /get_status`
- **THEN** 响应包含 VAD、CAM++、emotion2vec 的加载状态和显存占用，并包含两个模型池的副本与调度字段

#### Scenario: 模型尚未加载
- **WHEN** 某模型尚未被任何请求触发加载
- **THEN** 状态接口中该模型的加载状态为未加载、已加载副本数为 0 且显存占用为 0

### Requirement: 实例数量与显存放大的关系可控

系统 MUST 保证完整服务实例数量与模型池实例数量分别可配置且语义独立：`instance_count` 决定服务进程数量并默认 1，每个服务进程按 `[model_device.local_model_pool]` 中的设备数组持有对应实例；总模型显存随服务进程数与各模型数组长度的乘积增长，具体占用必须可在状态接口和测试报告中核对。

#### Scenario: 默认单实例
- **WHEN** 未显式设置 `instance_count` 和模型池设备数组
- **THEN** 服务以 1 个实例运行，每个启用模型使用 1 个副本，`nvidia-smi` 中只出现一行 `asr-offline-model`

#### Scenario: 配置多个模型副本
- **WHEN** `instance_count=1` 且 CAM++ 配置为 `["cuda:1", "cuda:2"]`
- **THEN** 服务仍只有 1 个 `asr-offline-model` 进程，但该进程可见 2 个 CAM++ 副本的聚合显存和状态

#### Scenario: 配置多个实例
- **WHEN** `instance_count=2` 且每个模型池设备数组长度为 N
- **THEN** 服务运行 2 个独立进程，每个进程各持有 N 个实例，总模型显存按两个进程分别加载的实例合计计算
