# Spec Delta

## ADDED Requirements

### Requirement: 日志路径归入日志配置段
系统 MUST 从 `[log].log_path` 读取日志文件路径，默认值 MUST 为 `"./asr_service.log"`；配置文件 MUST NOT 同时保留生效的顶层 `log_path`。

#### Scenario: 使用日志段中的路径
- **WHEN** 配置包含 `[log].log_path = "./logs/service.log"`
- **THEN** 文件日志写入该路径，并继续使用 `[log]` 中的保留天数和大小限制

#### Scenario: 日志路径缺省
- **WHEN** `[log]` 中未提供 `log_path`
- **THEN** 系统使用 `./asr_service.log`

#### Scenario: 顶层旧路径迁移
- **WHEN** 部署配置仍使用顶层 `log_path` 而没有 `[log].log_path`
- **THEN** 系统按约定的兼容策略处理并记录迁移提示；新发布的标准配置不再包含顶层键

### Requirement: 本地与容器配置结构一致
系统 MUST 保证 `config.toml` 与 `config.docker.toml` 对日志路径、VAD 阈值、模型设备和文本后处理相关配置使用相同的 section 与 key 结构；两份配置的环境差异仅限路径、Qwen 地址等部署值。

#### Scenario: 校验两份标准配置
- **WHEN** 测试读取本地配置和容器配置
- **THEN** 两份配置均包含 `[log].log_path`，且不包含顶层 `log_path`

#### Scenario: 配置项可被实际读取
- **WHEN** 标准配置中的任一新增配置被修改并重启服务
- **THEN** 对应运行时行为使用修改后的值，不能只存在于示例文件而未被消费

