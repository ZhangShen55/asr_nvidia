# Spec Delta

## ADDED Requirements

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
