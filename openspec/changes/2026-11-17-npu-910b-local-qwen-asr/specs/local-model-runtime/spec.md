# Spec Delta

## MODIFIED Requirements

### Requirement: 暴露本地模型的加载状态与显存明细
系统 MUST 在状态接口中暴露 VAD、CAM++、emotion2vec 各自的加载状态与明确口径的 NPU HBM 占用信息，区分主服务与 VAD worker 进程；无法按模型准确归因的资源 MUST 明确标记，不得把其他进程占用算作本模型占用；对于 CAM++ 与 emotion2vec，状态 MUST 同时包含设备数组和独立模型池的容量、忙碌、等待与失败信息。

#### Scenario: 查询模型状态
- **WHEN** 运维请求 `GET /get_status`
- **THEN** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态、设备、HBM 观测口径和各自模型池状态；转写模型的状态由本地推理契约另行提供

#### Scenario: 模型尚未加载
- **WHEN** CAM++ 或 emotion2vec 尚未被任何请求触发加载
- **THEN** 状态接口中该模型的加载状态为未加载且 HBM 占用为 0，同时保留其配置实例数和设备数组

### Requirement: 实例数量与显存放大的关系可控
系统 MUST 保证服务实例数量与 HBM 占用之间存在明确、可预期的关系：完整服务进程实例数量 MUST 由配置项 `instance_count` 决定；单个服务进程内 CAM++ 与 emotion2vec 的模型副本数量和并发容量 MUST 分别由各自设备数组决定，不能由 `instance_count` 或共享模型槽位隐式改变。

#### Scenario: 默认单实例
- **WHEN** 配置中未显式设置 `instance_count` 且两个模型设备数组各有一个元素
- **THEN** 服务以 1 个完整服务实例运行，两个模型池各有 1 个实例

#### Scenario: 配置多个实例
- **WHEN** 配置中 `instance_count` 为 2 且模型设备数组保持不变
- **THEN** 服务以 2 个进程运行，每个进程独立建立增强模型池和 VAD worker 池，并在触发转写后各自加载 Qwen 权重；HBM 容量评估包含这些副本，模型池比例不在进程间合并

### Requirement: 本地增强模型池按设备数组建立固定实例
系统 MUST 从 `[model_device_pool]` 的两个设备数组建立 CAM++ 与 emotion2vec 独立模型池；每个数组元素对应一个模型实例，数组长度决定配置实例数，实例在首次启用对应能力时按数组顺序加载到指定设备。

#### Scenario: CAM++ 与 emotion2vec 实例数量不同
- **WHEN** CAM++ 数组有 3 个元素、emotion2vec 数组有 1 个元素
- **THEN** 两个模型池分别最多提供 3 个和 1 个实例，不因数量不同而补齐或共享实例

#### Scenario: 多卡设备数组
- **WHEN** 同一模型数组包含多个 `npu:N` 设备
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

### Requirement: 模型池并发容量由各自设备数组决定
系统 MUST 将 `speech_campplus_devices` 数组中成功加载的实例数作为 CAM++ 模型池的最大并发容量，并将 `emotion2vec_devices` 数组中成功加载的实例数作为 emotion2vec 模型池的最大并发容量；两个模型池 MUST 独立排队，MUST NOT 通过固定比例或额外模型槽位配置绑定容量。整音频 B 类准入上限 MUST NOT 改变这两个模型池的实例数或段级推理容量。

#### Scenario: 两个模型配置不同数量实例
- **WHEN** CAM++ 设备数组有 3 个元素且 emotion2vec 设备数组有 1 个元素
- **THEN** CAM++ 最多同时执行 3 个推理，emotion2vec 最多同时执行 1 个推理，两个池互不借用容量

#### Scenario: 配置数组为空
- **WHEN** 任一模型的设备数组为空
- **THEN** 对应模型池容量为 0、不加载该模型实例，要求该能力的请求得到明确不可用结果

#### Scenario: 部分实例加载失败
- **WHEN** 设备数组声明多个实例但其中部分实例加载失败
- **THEN** 模型池最大并发容量按成功加载的实例数计算，失败实例原因可查询且不占用可用槽位

## REMOVED Requirements

### Requirement: 本地模型由单一服务进程承载并统一命名
**Reason**: 现行实现已使用独立 spawn VAD worker 池；要求 VAD、CAM++、emotion2vec 同进程与实际固定池架构冲突，本次 NPU 迁移不重构该进程边界。

**Migration**: 由「主服务与固定 VAD worker 分工承载模型」承接进程命名、生命周期与资源观测职责。

## ADDED Requirements

### Requirement: 主服务与固定 VAD worker 分工承载模型
系统 MUST 在主服务进程中承载 CAM++、emotion2vec 与本地 Qwen 推理；CAM++、emotion2vec 和 Qwen 按各自能力契约懒加载。VAD MUST 保留 spawn 创建的固定独立 worker 池，启动时预热，按 FIFO 调度，设备数组长度决定 worker 数，MUST NOT 为本次迁移改为主进程内 VAD。主服务进程 MUST 命名为 `asr-offline-model`，VAD worker MUST 使用 `asr-offline-vad`；运维 MUST 能区分两类进程的资源，MUST NOT 把主服务 HBM 称为含独立 VAD worker 的全部占用。

#### Scenario: 服务启动并预热 VAD
- **WHEN** 服务按固定 VAD 设备数组启动
- **THEN** spawn worker 完成 VAD 预热后才报告就绪，主服务不因此加载尚未请求的增强模型或 Qwen 权重

#### Scenario: 查询进程资源
- **WHEN** 运维查看 NPU 进程及服务状态
- **THEN** 主服务与 VAD worker 可区分，资源统计覆盖各自进程，无法归因的 HBM 明确标记；查询不触发懒加载

#### Scenario: 本轮宿主开发使用指定设备
- **WHEN** 在 xjtu910b7 的 conda asr 环境开展本轮开发
- **THEN** 使用 `/home/zhangs/conda-envs/asr/bin/python` 和宿主 `npu:7`，不调度其他卡或终止既有服务；容器内逻辑编号按实际设备映射核对，不照搬宿主卡号
