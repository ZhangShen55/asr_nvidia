# Spec Delta

## MODIFIED Requirements

### Requirement: 默认配置体现有效并发来源
默认配置 MUST 使用 `vad_device` 数组表达 VAD 实例设备，使用 `[model_device_pool]` 表达 CAM++/emotion2vec 实例设备，使用 `[qwen_asr].audio_concurrency` 表达总音频准入上限、`audio_spk_emo_concurrency` 表达增强音频准入上限，并 MUST 不包含已删除的 VAD 动态池字段和旧模型设备结构。音频准入配置 MUST NOT 被解释为模型实例数量。

#### Scenario: 检查默认配置
- **WHEN** 运维读取 `config.toml` 或 `config.docker.toml`
- **THEN** 能直接通过 VAD 数组长度、模型设备数组长度计算配置实例数，并单独查看总音频和增强音频准入上限；无需额外配置模型推理槽位

#### Scenario: 功能关闭时模型池不生效
- **WHEN** `open_v2_role=false` 或 `open_v2_emotion=false`
- **THEN** 对应模型设备数组不会触发模型加载或推理，但配置仍可保留用于重启后开启功能

## ADDED Requirements

### Requirement: 增强音频并发配置具有确定默认值
系统 MUST 从 `[qwen_asr].audio_spk_emo_concurrency` 读取 B 类整音频并发上限，缺省值 MUST 为 `4`，并在本地及容器配置模板展示该项。该值 MUST 是正整数，非法值 MUST 在服务就绪前明确报错；关闭能力 MUST 使用功能开关。配置修改 MUST 重启生效，且每个服务实例独立执行两类音频限制。

#### Scenario: 缺少新增配置项
- **WHEN** 旧配置未声明 `audio_spk_emo_concurrency`
- **THEN** 服务使用增强上限 4，不自动回退为总音频上限；文档说明升级后增强并发受到新增默认上限约束

#### Scenario: 显式配置其他上限
- **WHEN** 运维配置 `audio_concurrency=8`、`audio_spk_emo_concurrency=2` 并重启
- **THEN** 每个实例最多同时处理 8 个音频，其中最多 2 个为 B

#### Scenario: 非法增强上限
- **WHEN** 新增配置值为 0、负数、布尔值、浮点数或字符串
- **THEN** 服务在就绪前报告字段名称和正整数约束，不静默纠正为 1 或把 0 当作关闭功能

#### Scenario: 增强上限大于总上限
- **WHEN** 总上限为 2、增强上限为 4
- **THEN** 配置有效，两个限制同时生效使 B 实际至多为 2；状态仍如实展示配置值 2 和 4，不静默改写增强配置

#### Scenario: 多服务实例
- **WHEN** 同一部署启动多个服务实例
- **THEN** 每个实例独立维护总音频及增强音频容量，文档说明该限制不是跨实例的全局限制

### Requirement: 暴露音频分类准入状态
`GET /get_status` MUST 以附加字段 `audioConcurrency` 展示 `configured_total`、`active_total`、`waiting_total`，以 `spkEmoConcurrency` 展示 `configured`、`active`、`waiting`。两组值 MUST 来自同一次准入状态快照，可由总数减 B 数得到 A 数；查询 MUST NOT 触发本地模型加载，既有业务、模型池及诊断字段语义 MUST 保持兼容。

#### Scenario: 查询混合活动及等待状态
- **WHEN** 总上限 8、增强上限 4、正在处理 `5A+3B`、等待 1 个 A 和 2 个 B
- **THEN** `audioConcurrency` 返回 `{"configured_total":8,"active_total":8,"waiting_total":3}`，`spkEmoConcurrency` 返回 `{"configured":4,"active":3,"waiting":2}`，可计算活动 A 为 5、等待 A 为 1

#### Scenario: 首次业务请求前查询
- **WHEN** 服务已启动但尚无音频请求，运维查询状态
- **THEN** 两组配置值如实返回，活动和等待计数均为 0，CAM++ 和 emotion2vec 不因查询而加载

#### Scenario: 功能关闭并拒绝增强请求
- **WHEN** 两种增强能力均关闭且增强请求被准入前校验拒绝
- **THEN** 状态中的增强活动数和等待数保持为 0，增强配置值仍正常展示

### Requirement: 分类准入记录有界且可区分错误原因
系统 MUST 为 v2 请求记录 A/B 分类、音频准入等待耗时、准入时的总容量与 B 容量快照及有限集合的失败原因。日志和遥测 MUST 能分别统计总容量满、增强容量满、共享等待队列满、等待超时及能力关闭；遥测 MUST 保持有界存储，不保存音频、转写正文或模型对象，且保留既有阶段字段的语义。

#### Scenario: 区分两个满额错误
- **WHEN** 请求分别因总容量满返回 `4013`、因增强容量满返回 `4017`
- **THEN** 日志与最近请求遥测能够区分类别、错误码及满额原因，包含对应容量快照，汇总分别计数

#### Scenario: 同码不同拒绝原因
- **WHEN** 请求因共享等待队列满返回 `4013`
- **THEN** 记录等待队列满原因及等待使用量，与总活动容量满的 `4013` 分开统计

#### Scenario: 组合关闭错误和提前返回
- **WHEN** 请求因两项所需能力都关闭返回 `4018`
- **THEN** 日志和遥测记录 B 类及组合关闭原因，不虚构模型推理耗时，也不继承其他请求的阶段数据

#### Scenario: 等待和处理取消
- **WHEN** 请求等待超时或被取消
- **THEN** 遥测区分超时和取消，不将取消误记为处理成功，已回收的名额与状态快照一致

#### Scenario: 长期运行的遥测存储
- **WHEN** 请求数量超过既有遥测事件上限
- **THEN** 继续按有界历史保留策略更新事件，分类及拒绝原因统计明确使用该保留窗口，不创建按请求无限增长的记录或计数键

### Requirement: 文档明确分类限制与能力错误
README 和部署文档 MUST 说明 A/B 分类、两类容量关系、默认值、缺省迁移、等待调度和按进程生效的范围；错误码清单 MUST 区分 `4013`、`4015`、`4016`、`4017`、`4018`。文档 MUST 区分配置关闭的请求级错误、模型池无可用实例和已开启模型的段级推理失败。

#### Scenario: 查阅默认并发示例
- **WHEN** 用户查阅总上限 8、增强上限 4 的说明
- **THEN** 文档包含 `3A+4B` 时新 B 返回 `4017` 且新 A 可处理，以及 `5A+3B`、`8A` 时新 A/B 均返回 `4013` 的示例

#### Scenario: 查阅能力关闭行为
- **WHEN** 用户查询两个功能开关为 `false`、请求增强参数为 `true` 的处理规则
- **THEN** 文档说明请求失败及单项或组合错误码，不再承诺在配置关闭时以 `null` 字段正常转写；普通 A 请求仍可正常处理
