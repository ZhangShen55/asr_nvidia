# Spec Delta

## Purpose

为 CAM++ 与 emotion2vec 提供独立于完整服务进程数量的本地模型池能力，使多个 GPU 上的显式设备数组实例可以参与并发推理，同时保持现有请求结果、懒加载和失败降级语义可观测、可验证。

## ADDED Requirements

### Requirement: 在 model_device 中配置模型池

系统 MUST 在 `[model_device.local_model_pool]` 中通过 `speech_campplus_devices` 与 `emotion2vec_devices` 非空数组配置实例；数组长度即对应模型实例数，每个元素必须是合法的 `cuda:N`。缺省数组为 `["cuda:0"]`，且该配置 MUST 与 `instance_count` 相互独立。

#### Scenario: 使用默认模型池配置
- **WHEN** 配置未提供模型池字段
- **THEN** 每个启用的本地模型使用 1 个副本并绑定 `cuda:0`，服务行为与现有单实例模式一致

#### Scenario: 配置不同模型副本数
- **WHEN** 运维分别配置 `speech_campplus_devices = ["cuda:1", "cuda:2"]`、`emotion2vec_devices = ["cuda:1"]`
- **THEN** 两个模型按各自数量加载和调度，修改模型副本数不改变完整服务进程的 `instance_count`

#### Scenario: 拒绝非法设备数组
- **WHEN** 任一设备数组为空或包含非法设备字符串
- **THEN** 配置校验失败并明确指出对应数组元素，服务不得以部分配置启动

### Requirement: 模型池副本固定绑定同一设备

系统 MUST 将模型池的每个副本绑定到设备数组中对应的 CUDA 设备；系统 MUST NOT 自动重排、迁移或把未配置的副本分布到其它 GPU。

#### Scenario: 多副本按数组落卡
- **WHEN** CAM++ 配置 `["cuda:1", "cuda:2"]`，emotion2vec 配置 `["cuda:1"]`
- **THEN** CAM++ 两个副本分别加载到 GPU 1、GPU 2，emotion2vec 副本加载到 GPU 1

#### Scenario: 单卡资源竞争
- **WHEN** 多个副本同时获得推理任务并竞争 GPU 0
- **THEN** 请求可以等待 GPU 资源或由 CUDA 进行抢占，服务不得因为资源竞争改变响应结构或静默改用其它设备

### Requirement: 模型池按需加载并独立回收

系统 MUST 只为已启用且被请求触发的模型创建副本；模型池初始化失败或单个副本推理失败时，系统 MUST 隔离该副本并继续提供其它副本及转写服务，且失败原因可查询。

#### Scenario: 未触发说话人能力
- **WHEN** `showSpk=false` 或 `open_v2_role=false`
- **THEN** CAM++ 模型池不加载任何副本且角色字段保持 `null`

#### Scenario: 单个副本加载失败
- **WHEN** 模型池中的一个副本加载失败
- **THEN** 该副本标记为失败并记录原因，其余可用副本继续接收任务；若无可用副本，角色字段按现有失败降级返回 `null`

#### Scenario: 服务结束回收副本
- **WHEN** 服务进程退出或模型池被显式关闭
- **THEN** 已加载副本释放其模型对象和 GPU 资源，且不会残留后台推理任务

### Requirement: 模型池调度保持推理结果契约

系统 MUST 将每次模型推理绑定到一个可用副本并返回与单副本模式相同的数据契约；CAM++ 副本返回的 embedding 必须可以合并到同一音频的统一角色聚类流程，emotion2vec 副本返回的情绪标签必须保持现有标签格式。

#### Scenario: 多副本处理同一音频的不同片段
- **WHEN** 同一音频的多个 VAD 段被分派到不同 CAM++ 副本
- **THEN** 所有 embedding 在该音频的统一聚类阶段合并处理，响应中的角色编号保持连续且与段顺序一致

#### Scenario: 副本处理情绪片段
- **WHEN** 不同 VAD 段被分派到不同 emotion2vec 副本
- **THEN** 每段返回一个与单副本模式同构的情绪值，不依赖跨段的模型内部状态

### Requirement: 暴露模型池运行状态

系统 MUST 在本地模型状态查询中暴露每个模型的配置副本数、已加载副本数、设备、可用数、失败数以及忙碌或等待计数；未启用模型和未触发加载的模型 MUST 返回零已加载副本。

#### Scenario: 查询多副本状态
- **WHEN** 运维调用 `GET /get_status`
- **THEN** 响应包含 CAM++ 与 emotion2vec 的模型池状态，能够区分配置数量、已加载数量和当前调度压力

#### Scenario: 查询未启用模型
- **WHEN** 对应 `open_v2_role` 或 `open_v2_emotion` 为 false
- **THEN** 该模型状态标记为未启用，已加载副本数和失败数均不因请求增长
