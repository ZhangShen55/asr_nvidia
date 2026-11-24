## MODIFIED Requirements

### Requirement: 本地模型按需懒加载
系统 MUST 在HTTP就绪前对功能开启且设备数组非空的CAM++、emotion2vec模型池完成全部配置实例的加载尝试，并常驻复用；关闭或空数组不加载。部分失败使用健康实例，某池全部失败不阻止其它能力服务，按原错误契约拒绝所需能力请求并记录原因。Qwen MUST 沿用自身预加载契约；请求要求被关闭的能力 MUST 在音频准入前明确拒绝。

#### Scenario: 未请求说话人能力
- **WHEN** showSpk和showRoleIdentify均为false
- **THEN** 请求不调用CAM++，已按启动策略加载的权重保持驻留，不因该请求重复加载

#### Scenario: 首次请求触发加载
- **WHEN** open_v2_role=true、数组非空且首次出现需要showSpk或showRoleIdentify的已准入请求
- **THEN** 请求复用启动阶段已加载的健康CAM++实例，不再由该请求首次构造模型池

#### Scenario: 模型加载失败
- **WHEN** 功能开关已开启，但所请求模型的全部实例加载失败或设备数组为空
- **THEN** 服务保持可用，该请求得到对应模型不可用错误（CAM++为4015、emotion2vec为4016），且失败原因可记录与查询；其它健康能力不受影响

#### Scenario: 两个能力独立预加载
- **WHEN** 服务启动且两个功能开启、设备数组非空
- **THEN** 在HTTP就绪前依数组完成两池全部实例加载尝试，一个池全部加载失败不跳过另一个池

#### Scenario: 关闭能力不加载
- **WHEN** 任一增强能力功能开关关闭或其设备数组为空
- **THEN** 启动不创建该能力模型实例，能力校验和不可用错误沿用原契约

### Requirement: 暴露本地模型的加载状态与显存明细
系统 MUST 在状态接口中暴露 VAD、CAM++、emotion2vec 各自的加载状态与明确口径的 NPU HBM 占用信息，区分主服务与 VAD worker 进程；无法按模型准确归因的资源 MUST 明确标记，不得把其他进程占用算作本模型占用；对于 CAM++ 与 emotion2vec，状态 MUST 同时包含设备数组和独立模型池的容量、忙碌、等待与失败信息。

#### Scenario: 查询模型状态
- **WHEN** 运维请求 `GET /get_status`
- **THEN** 响应中包含 VAD、CAM++、emotion2vec 三个模型的加载状态、设备、HBM 观测口径和各自模型池状态；转写模型的状态由本地推理契约另行提供

#### Scenario: 模型尚未加载
- **WHEN** CAM++ 或 emotion2vec 因功能关闭或设备数组为空而未加载
- **THEN** 状态接口中该模型的加载状态为未加载且 HBM 占用为 0，同时保留其配置实例数和设备数组

### Requirement: 本地增强模型池按设备数组建立固定实例
系统 MUST 从model_device_pool的两个设备数组建立每服务共享的CAM++与emotion2vec独立模型池，每个数组元素对应一个模型实例，数组长度决定配置实例数；在功能开启且数组非空时于HTTP就绪前按数组顺序完成加载尝试。各Qwen owner及不同请求 MUST 复用这两个池，不各自复制池；最终交付配置按用户选择CAM4/emo1，在四Qwen真实共驻B4下标定并冻结，不从N、S或B4推导；未验收不得宣称已通过。

#### Scenario: CAM++ 与 emotion2vec 实例数量不同
- **WHEN** CAM++数组有3个元素、emotion2vec数组有1个元素
- **THEN** 两个模型池分别最多提供3个和1个健康执行实例，不因数量不同而补齐、共享或为每个Qwen owner复制一组

#### Scenario: 多卡设备数组
- **WHEN** 同一模型数组包含多个npu:N设备
- **THEN** 每个实例加载到对应数组元素指定的设备，并在状态接口按实例顺序展示设备；本次宿主验证仍仅使用物理7号卡
