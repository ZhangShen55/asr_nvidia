# 规格增量

## MODIFIED Requirements

### Requirement: 本地模型由单一服务进程承载并统一命名
系统 MUST 由主服务进程承载 CAM++ 与 emotion2vec 两个本地模型，主进程名 MUST 为 `asr-offline-model`，并 MUST 通过覆写进程命令行首项的方式设置，使 `nvidia-smi` 的进程列表直接显示该名称。VAD MUST NOT 由主进程承载，MUST 由一组独立的 VAD worker 进程承载，worker 进程名 MUST 为 `asr-offline-vad`。当 VAD 配置为在 CPU 上运行时，VAD worker MUST NOT 建立 CUDA 上下文，MUST NOT 出现在 `nvidia-smi` 的 GPU 进程列表中。

#### Scenario: 在 nvidia-smi 中定位服务显存
- **当** 运维在 `nvidia-smi` 的 GPU 进程列表中查看
- **则** 本服务以 `asr-offline-model` 进程名出现，其显存占用为该进程内已加载本地模型的合计占用

#### Scenario: 未拆分为多个模型进程
- **当** 服务同时加载了 CAM++ 与 emotion2vec
- **则** 这两个模型处于同一进程内，`nvidia-smi` 中不出现按模型拆分的多个进程行；承载它们的进程不同时加载 VAD 权重

#### Scenario: VAD 运行在 CPU 时不产生 GPU 进程行
- **当** VAD 配置为在 CPU 上运行且请求触发了 VAD 切分
- **则** `nvidia-smi` 的 GPU 进程列表中不出现 VAD worker 进程行

#### Scenario: VAD 配置为 GPU 时进程行可辨识
- **当** VAD 配置为在 GPU 上运行且 VAD worker 持有 CUDA 上下文
- **则** `nvidia-smi` 的 GPU 进程列表中出现名为 `asr-offline-vad` 的进程行

#### Scenario: 主进程内不加载 VAD 权重
- **当** 服务已完成启动
- **则** 主进程内不存在 VAD 模型权重，VAD 权重只存在于 VAD worker 进程中

### Requirement: 暴露本地模型的加载状态与显存明细
系统 MUST 在状态接口中暴露 VAD、CAM++、emotion2vec 三个本地模型各自的加载状态与显存占用明细，使运维在 `nvidia-smi` 只能看到聚合显存的情况下仍能区分各模型的占用。当 VAD 运行在 CPU 上时，其显存占用 MUST 为 0，状态接口 MUST 同时体现 VAD 的实际运行设备，使运维能够区分「未加载」与「加载在 CPU 上」。

#### Scenario: 查询模型状态
- **当** 运维请求 `GET /get_status`
- **则** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态与各自显存占用明细

#### Scenario: 模型尚未加载
- **当** 某模型尚未被任何请求触发加载
- **则** 状态接口中该模型的加载状态为未加载且显存占用为 0

#### Scenario: VAD 运行在 CPU 上的状态体现
- **当** VAD 已加载且运行在 CPU 上
- **则** 状态接口中 VAD 的显存占用为 0，且其运行设备标识为 CPU

### Requirement: 实例数量与显存放大的关系可控
系统 MUST 保证服务实例数量与显存占用之间存在明确、可预期的关系：实例数量 MUST 由配置项 `instance_count` 决定，其默认值 MUST 为 1；每个实例进程各自持有一份 CAM++ 与 emotion2vec 权重，因此实例数量为 N 时这两个模型的显存占用为单实例的 N 倍。VAD worker 的数量 MUST 由独立的 `vad_pool_size` 决定，MUST NOT 受 `instance_count` 影响；VAD 配置在 GPU 上时其显存占用随 worker 数量增长。

#### Scenario: 默认单实例
- **当** 配置中未显式设置 `instance_count`
- **则** 服务以 1 个实例运行，`nvidia-smi` 中只出现一行 `asr-offline-model`

#### Scenario: 配置多个实例
- **当** 配置中 `instance_count` 为 2
- **则** 服务以 2 个实例运行，`nvidia-smi` 中出现两行 `asr-offline-model`，CAM++ 与 emotion2vec 显存占用为单实例的 2 倍

#### Scenario: VAD 池规模与实例数相互独立
- **当** 配置中 `instance_count` 为 2 且 `vad_pool_size` 为 8
- **则** 每个服务实例各自维护 8 个 VAD worker，VAD worker 总数为 16
