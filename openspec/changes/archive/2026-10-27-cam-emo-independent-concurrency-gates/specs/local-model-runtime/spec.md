# Spec Delta

## ADDED Requirements

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

## MODIFIED Requirements

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
