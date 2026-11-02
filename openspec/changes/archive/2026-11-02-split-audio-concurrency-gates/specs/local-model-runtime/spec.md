# Spec Delta

## MODIFIED Requirements

### Requirement: 本地模型按需懒加载
系统 MUST 支持本地模型的懒加载：对应功能开关关闭或请求未触发时，MUST NOT 加载或调用该模型；需要加载时，MUST 等待加载完成后才发起推理，模型加载失败 MUST NOT 导致整个服务无法启动。请求要求被关闭的能力时 MUST 在音频准入前返回明确错误，不得以空值掩盖配置关闭。

#### Scenario: 未请求说话人能力
- **WHEN** `showSpk` 和 `showRoleIdentify` 均为 `false`
- **THEN** CAM++ 权重不因该请求加载，该请求不增加 CAM++ 权重显存占用，也不调用该模型

#### Scenario: 首次请求触发加载
- **WHEN** `open_v2_role=true` 且首次出现需要 `showSpk` 或 `showRoleIdentify` 的已准入请求
- **THEN** 系统加载 CAM++ 权重并完成本次推理，后续请求复用已加载的模型

#### Scenario: 模型加载失败
- **WHEN** 功能开关已开启，但所请求模型的全部实例加载失败或设备数组为空
- **THEN** 服务保持可用，该请求得到对应模型不可用错误（CAM++ 为 `4015`、emotion2vec 为 `4016`），且失败原因可被记录与查询

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

## ADDED Requirements

### Requirement: 功能开关不能被设备配置或推理入口绕过
`open_v2_role=false` 或 `open_v2_emotion=false` 时，系统 MUST 阻止对应模型的按需加载和推理；非空 `[model_device_pool]` 数组、请求参数或直接进入增强运行链路 MUST NOT 覆盖功能开关。模型设备配置仅在对应能力开启且确有请求时生效；关闭能力的请求级错误遵循 v2 准入规格。

#### Scenario: 保留实例配置但关闭两种能力
- **WHEN** 两种功能开关均为 `false`，CAM++ 和 emotion2vec 均配置非空设备数组，请求同时需要两种能力
- **THEN** 返回 `4018`，两个模型均不加载、不推理，音频活动及等待计数保持不变

#### Scenario: 关闭说话人能力但请求角色识别
- **WHEN** `open_v2_role=false` 且请求 `showRoleIdentify=true`、`showEmotion=false`
- **THEN** 返回 `4015`，不因角色识别参数进入 CAM++ 模型加载或推理

#### Scenario: 关闭能力的运行入口防护
- **WHEN** 对应功能开关为 `false`，增强运行链路被直接调用而未经过路由校验
- **THEN** 系统仍明确拒绝使用该能力，不加载、不租用、不调用该模型，也不把配置关闭当成普通段级推理失败

#### Scenario: 仅使用已开启的另一种能力
- **WHEN** 一种增强能力关闭，另一种开启，请求只需要已开启的能力
- **THEN** 请求按 B 类正常准入，仅加载和调用所需模型，关闭的模型不因设备配置而生效
