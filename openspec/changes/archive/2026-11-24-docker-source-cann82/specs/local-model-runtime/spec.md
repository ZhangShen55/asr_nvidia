# Spec Delta

## MODIFIED Requirements

### Requirement: 实例数量与显存放大的关系可控
系统 MUST 分开表达完整HTTP服务数instance_count、每服务Qwen owner数N及各增强数组副本数。本版无代理源码容器 MUST 限定instance_count=1，在worker/模型创建前拒绝其它值；N默认4，合法1～8保留。增加N不得复制HTTP、VAD或增强池。一般多HTTP架构的资源评估仍计入每服务全部副本、容量分别生效，但不属于本版容器支持范围。

#### Scenario: 默认单实例
- **WHEN** 未声明instance_count且两个增强数组各有一个元素
- **THEN** 一个HTTP内四Qwen owner、各一个增强实例，不按Qwen数复制增强池；此通用数组示例不改变最终交付CAM4/emo1

#### Scenario: 配置多个实例
- **WHEN** 本版源码容器配置instance_count=2
- **THEN** 在模型及worker创建前明确拒绝并说明单HTTP限制，不静默启动一份或恢复旧Nginx多HTTP模式；其它显式多服务架构仍须分别计入各自模型资源，不能合称全局容量

#### Scenario: 只增加Qwen实例
- **WHEN** 单HTTP内修改合法N并重启，增强及VAD数组未改
- **THEN** 只改变Qwen数量及其派生容量，不倍增VAD、增强实例或HTTP进程
