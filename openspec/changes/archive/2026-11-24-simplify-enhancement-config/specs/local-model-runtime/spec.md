# Spec Delta

## MODIFIED Requirements

### Requirement: 本地模型按需懒加载
系统 MUST 仅由各自设备数组非空决定CAM++、emotion2vec能力启用；空数组禁用，不再读取features开关。非空数组在HTTP就绪前完成全部配置实例加载尝试并常驻复用；部分失败使用健康实例，全部失败不阻止其它能力服务。配置关闭与加载故障 MUST 区分，所需关闭能力在音频准入前按4015/4016/4018拒绝；Qwen沿自身预加载契约。

#### Scenario: 未请求说话人能力
- **WHEN** showSpk和showRoleIdentify均为false
- **THEN** 请求不调用CAM++，已按启动策略加载的权重保持驻留，不因该请求重复加载

#### Scenario: 首次请求触发加载
- **WHEN** CAM++设备数组非空且首次出现需要showSpk或showRoleIdentify的已准入请求
- **THEN** 请求复用启动阶段已加载的健康CAM++实例，不再由该请求首次构造模型池

#### Scenario: 模型加载失败
- **WHEN** 所请求模型设备数组非空，但其全部实例加载失败
- **THEN** 服务保持可用，该请求得到对应模型不可用错误（CAM++为4015、emotion2vec为4016），且失败原因可记录与查询；其它健康能力不受影响

#### Scenario: 两个能力独立预加载
- **WHEN** 服务启动且两个设备数组均非空
- **THEN** 在HTTP就绪前依数组完成两池全部实例加载尝试，一个池全部加载失败不跳过另一个池

#### Scenario: 关闭能力不加载
- **WHEN** 任一增强能力设备数组为空
- **THEN** 启动不创建该能力模型实例，能力校验和不可用错误沿用原契约

### Requirement: 暴露本地模型的加载状态与显存明细
系统 MUST 在状态接口中暴露 VAD、CAM++、emotion2vec 各自的加载状态与明确口径的 NPU HBM 占用信息，区分主服务与 VAD worker 进程；无法按模型准确归因的资源 MUST 明确标记，不得把其他进程占用算作本模型占用；对于 CAM++ 与 emotion2vec，状态 MUST 同时包含设备数组和独立模型池的容量、忙碌、等待与失败信息。

#### Scenario: 查询模型状态
- **WHEN** 运维请求 `GET /get_status`
- **THEN** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态、设备、HBM 观测口径和各自模型池状态；转写模型的状态由本地推理契约另行提供

#### Scenario: 模型尚未加载
- **WHEN** CAM++ 或 emotion2vec 因设备数组为空而未加载
- **THEN** 状态接口中该模型的加载状态为未加载且 HBM 占用为 0，同时保留其配置实例数和设备数组

### Requirement: 本地增强模型池按设备数组建立固定实例
系统 MUST 从model_device_pool的两个设备数组建立每服务共享的CAM++与emotion2vec独立模型池，每个数组元素对应一个模型实例，数组长度决定配置实例数；在数组非空时于HTTP就绪前按数组顺序完成加载尝试。各Qwen owner及不同请求 MUST 复用这两个池，不各自复制池；最终交付配置按用户选择CAM4/emo1，在四Qwen真实共驻B4下标定并冻结，不从N、S或B4推导；未验收不得宣称已通过。

#### Scenario: CAM++ 与 emotion2vec 实例数量不同
- **WHEN** CAM++数组有3个元素、emotion2vec数组有1个元素
- **THEN** 两个模型池分别最多提供3个和1个健康执行实例，不因数量不同而补齐、共享或为每个Qwen owner复制一组

#### Scenario: 多卡设备数组
- **WHEN** 同一模型数组包含多个npu:N设备
- **THEN** 每个实例加载到对应数组元素指定的设备，并在状态接口按实例顺序展示设备；本次宿主验证仍仅使用物理7号卡

### Requirement: 模型池配置只使用新配置结构
系统 MUST 将[model_device_pool]的speech_campplus_devices和emotion2vec_devices作为两增强能力的唯一正式启用及设备来源，每数组元素一实例，空数组禁用。旧[model_device]结构不得覆盖新数组；已删除的features开关 MUST 在创建任何worker或模型前拒绝并给出迁移办法，不能静默忽略旧false而启用非空数组。

#### Scenario: 只存在新配置
- **WHEN** 配置仅提供 `[model_device_pool]` 及两个设备数组
- **THEN** 非空数组按配置建立并启动预加载模型池，空数组不加载对应模型

#### Scenario: 旧配置与新配置同时存在
- **WHEN** 配置同时包含旧[model_device]或[model_device.local_model_pool]结构和新[model_device_pool]数组，此处旧结构不含已删除的features.open_v2_*开关
- **THEN** 服务使用新结构，并记录可操作的迁移告警

#### Scenario: 空数组禁用模型
- **WHEN** 某模型设备数组为空
- **THEN** 该模型不创建实例；请求要求该能力时返回明确的模型不可用错误，其他能力仍可用

#### Scenario: 旧增强开关残留
- **WHEN** 配置仍声明features.open_v2_role或features.open_v2_emotion，无论值为true还是false
- **THEN** 在任何worker或模型创建前报告具体旧字段和删除办法；原false通过清空对应设备数组迁移，不能把它无声变为开启


### Requirement: 功能开关不能被设备配置或推理入口绕过
系统 MUST 在配置加载、启动预加载、路由能力校验及直接增强运行入口一致使用设备数组派生的启用状态。空数组能力不加载、不租用、不推理，非空数组容量和请求参数不能覆盖空数组关闭。请求只决定是否执行已启用能力，不能扩容或自动开启；所需关闭能力按既有单项/组合4015、4016、4018在音频准入前拒绝，实际加载失败仍按模型故障契约处理。

#### Scenario: 保留实例配置但关闭两种能力
- **WHEN** CAM++和emotion2vec设备数组均为空，请求同时需要两种能力
- **THEN** 返回 `4018`，两个模型均不加载、不推理，音频活动及等待计数保持不变

#### Scenario: 关闭说话人能力但请求角色识别
- **WHEN** CAM++设备数组为空 且请求 `showRoleIdentify=true`、`showEmotion=false`
- **THEN** 返回 `4015`，不因角色识别参数进入 CAM++ 模型加载或推理

#### Scenario: 关闭能力的运行入口防护
- **WHEN** 对应设备数组为空，增强运行链路被直接调用而未经过路由校验
- **THEN** 系统仍明确拒绝使用该能力，不加载、不租用、不调用该模型，也不把配置关闭当成普通段级推理失败

#### Scenario: 仅使用已开启的另一种能力
- **WHEN** 一种增强能力关闭，另一种开启，请求只需要已开启的能力
- **THEN** 请求按 B 类正常准入，仅加载和调用所需模型，关闭的模型不因请求参数或直接调用而生效

### Requirement: 主服务与固定 VAD worker 分工承载模型
正式continuous/process MUST 由一个主HTTP进程协调N个owned spawn Qwen进程，每份仅在所属worker加载权重；CAM++按既有配置承载方式使用共享独立池，emotion2vec维持既有共享runner。VAD MUST 保留启动预热的固定spawn池，按数组长度配置并FIFO分派完整音频，不改主进程内VAD或运行期扩缩容。主进程保持asr-offline-model、VAD保持asr-offline-vad；运维 MUST 按出生身份区分各worker资源。legacy保留既有承载方式。

#### Scenario: 服务启动并预热 VAD
- **WHEN** 服务按固定VAD设备数组启动
- **THEN** spawn worker完成VAD预热并满足可用条件后才就绪，增强池另按设备数组在HTTP就绪前预加载，空数组不加载，预加载不执行语音推理；Qwen另按preload开关完成全部预加载或保持懒加载，不能跳过默认预加载就绪条件

#### Scenario: 查询进程资源
- **WHEN** 运维查看NPU进程及服务状态
- **THEN** 主服务、VAD、各Qwen及已配置进程模式CAM可按PID和出生身份区分；父与worker资源分别统计，无法归因者明确标记，查询不触发懒加载，不把父进程HBM当作全部worker总量

#### Scenario: 本轮宿主开发使用指定设备
- **WHEN** 在xjtu910b7的conda asr环境开展本轮开发
- **THEN** 使用/home/zhangs/conda-envs/asr/bin/python及宿主npu:7，不调度其他卡或终止既有服务；容器逻辑编号按实际可见映射核对，不照搬宿主卡号
