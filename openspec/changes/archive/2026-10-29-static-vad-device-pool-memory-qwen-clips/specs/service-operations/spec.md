# Spec Delta

## ADDED Requirements

### Requirement: 启动时固定预热 VAD 实例
服务 MUST 在启动阶段按 `qwen_asr.vad_device` 数组预热全部 VAD worker，并在启动完成前报告每个 worker 的设备、成功或失败状态；运行期间 MUST 不因等待队列自动增加或回收 VAD worker。

#### Scenario: 默认混合设备配置
- **WHEN** 使用默认 8 元素 VAD 设备数组启动服务
- **THEN** 服务预热 4 个 CPU worker 和 4 个 `cuda:0` worker，状态接口显示 8 个配置实例

#### Scenario: 启动时设备不可用
- **WHEN** VAD 设备数组中的 CUDA 设备不可用
- **THEN** 服务记录对应 worker 的失败原因；若没有任何成功 worker，则拒绝完成服务启动

### Requirement: 默认配置体现有效并发来源
默认 `config.toml` MUST 使用 `vad_device` 数组表达 VAD 并发，使用 `[model_device_pool]` 表达 CAM++/emotion2vec 实例设备，并 MUST 不包含已删除的 VAD 动态池字段和旧模型设备结构。

#### Scenario: 检查默认配置
- **WHEN** 运维读取默认 `config.toml`
- **THEN** 能直接通过 VAD 数组长度、模型设备数组长度计算有效实例数，不需要查找额外槽位配置

#### Scenario: 功能关闭时模型池不生效
- **WHEN** `open_v2_role=false` 或 `open_v2_emotion=false`
- **THEN** 对应模型设备数组不会触发模型加载，但配置仍可保留用于后续开启功能

### Requirement: 降低控制台 HTTP 访问日志噪声
服务 MUST 仅在控制台处理 `httpx`/`httpcore` 成功请求的 `INFO` 访问日志，使其不再逐段打印 HTTP 200 行；`WARNING`、`ERROR` 和业务处理日志 MUST 继续输出。文件日志是否保留完整 HTTP 访问记录 MUST 与控制台过滤解耦，并按现有文件日志策略保留。

#### Scenario: 成功请求不刷屏
- **WHEN** Qwen 客户端收到 HTTP 2xx 成功响应
- **THEN** 控制台不打印对应的 `HTTP Request ... 200 OK` `INFO` 行，进度和业务耗时日志仍正常显示

#### Scenario: HTTP 警告和错误保留
- **WHEN** Qwen 请求产生连接警告、非 2xx 响应或异常
- **THEN** 控制台保留对应的 warning/error 记录，并包含可定位请求失败的上下文

#### Scenario: 非 TTY 日志输出
- **WHEN** 服务输出被重定向到文件、管道或测试捕获器
- **THEN** 不启用 tqdm 的 ANSI 控制序列；普通日志可被稳定解析，不能因为进度显示改变日志级别或丢失错误

## MODIFIED Requirements

### Requirement: 维护配置项清单
系统 MUST 只消费配置文件中实际生效的配置项；无业务含义或不再被任何功能读取的配置项 MUST NOT 保留在配置文件中。配置项的增删 MUST 与功能开关保持一致，功能下线时其专属配置项 MUST 一并移除。VAD 动态扩缩容字段和旧模型设备结构属于本次下线配置，必须从正式默认配置中删除。

#### Scenario: 配置文件包含已下线功能的配置项
- **WHEN** 部署配置中残留 `vad_pool_size`、`vad_pool_max`、`vad_queue_watermark`、`vad_collapse_timeout_s`、`vad_queue_timeout_s` 或 `vad_call_timeout_s`
- **THEN** 服务不使用这些字段改变固定 VAD 数组，并记录迁移提示或配置校验告警

#### Scenario: 配置文件包含旧模型设备结构
- **WHEN** 部署配置中残留 `[model_device]` 或 `[model_device.local_model_pool]`
- **THEN** 服务不使用旧结构覆盖 `[model_device_pool]`，并提示运维迁移到新的设备数组结构

#### Scenario: 缺少可选配置项
- **WHEN** 配置文件缺少某个仍有效的可选配置项
- **THEN** 服务使用该项声明的默认值运行，不因删除无效字段而改变其它能力
