# Spec Delta

## MODIFIED Requirements

### Requirement: 维护配置项清单
系统 MUST 只消费实际生效配置，删除下线功能专属字段，保留既有旧VAD和设备结构迁移诊断。正式continuous MUST 以N、S分别派生audio_concurrency=N和request_concurrency=N×S，根与Docker模板不重复维护这两个值；兼容显式相等值须提示迁移，冲突或非法类型/范围须就绪前报错。legacy显式选择时保留原独立容量参数，不能通过忽略冲突配置静默切换语义；修改均重启生效。根与Docker的continuous模板 MUST 删除仅legacy生效的batch_wait_timeout_ms、batch_max_duration_ratio显式赋值；显式legacy仍支持原字段及语义。features旧开关不再接受，残留时在任何worker/模型创建前明确拒绝并说明迁移；两个增强batch_size从正式模板移除，保留实际batch1与原非1值拒绝约束。

#### Scenario: 配置文件包含已下线功能的配置项
- **WHEN** 部署配置中残留vad_pool_size、vad_pool_max、vad_queue_watermark、vad_collapse_timeout_s、vad_queue_timeout_s或vad_call_timeout_s
- **THEN** 服务不使用这些字段改变固定VAD数组，并记录迁移提示或配置校验告警

#### Scenario: 配置文件包含旧模型设备结构
- **WHEN** 部署配置中残留model_device或model_device.local_model_pool结构
- **THEN** 服务不使用旧结构覆盖model_device_pool，并提示迁移到新的设备数组

#### Scenario: 缺少可选配置项
- **WHEN** 配置文件缺少某个仍有效的可选配置项
- **THEN** 服务使用loader与交付模板一致的声明默认值，continuous的总音频及全局段容量仍从N、S派生，不因旧字段缺失回退到8或128

#### Scenario: 旧容量值与派生值一致
- **WHEN** continuous配置N=4、S=16且旧字段显式audio_concurrency=4、request_concurrency=64
- **THEN** 允许启动并给出删除重复字段的迁移提示，状态报告其派生来源，模板不继续复制这两个赋值

#### Scenario: 旧容量值冲突或类型非法
- **WHEN** continuous的旧显式容量与N或N×S不一致，或N、S、容量字段包含布尔值、非整数、越界值
- **THEN** 就绪前报具体字段及期望派生值，不静默取较小值、忽略冲突或将非法值纠正为1

#### Scenario: 当前continuous模板精简
- **WHEN** 运维读取根或Docker的正式continuous模板
- **THEN** 不再出现两项legacy组批字段、两个固定增强batch_size或features.open_v2_role/open_v2_emotion；增强能力仅由各自设备数组表达

#### Scenario: 固定增强batch模板省略
- **WHEN** 正式模板不再提供speech_campplus_batch_size和emotion2vec_batch_size
- **THEN** 实际推理和状态仍固定batch1，缺省值仍为1，显式非1值继续按原约束拒绝，不开放新批处理能力

#### Scenario: legacy实验参数仍有作用
- **WHEN** 显式选择legacy模式并提供batch_wait_timeout_ms或batch_max_duration_ratio
- **THEN** 原组批等待、长度分桶和类型/范围校验保持，不把当前模板精简当成删除legacy执行能力

#### Scenario: 保留有效参数
- **WHEN** 当前配置保留两个prefetch_queue_depth、online_repetition_*、repetition_*及retry_count
- **THEN** 仍按各自原契约加载和使用，不改变预取反压、在线检测、异常恢复或生成后修复规则


### Requirement: 默认配置体现有效并发来源
正式最终缺省 MUST 为continuous/process、N=4（合法1～8保留）、宿主npu:7、S=16、preload_on_startup=true、总音频派生N、全局段派生N×S及B配置上限4；HTTP缺省1。VAD、CAM++和emo配置实例数仍各由设备数组决定，最终增强池默认为用户选择的CAM4/emo1，须在四Qwen共驻B4及增强驻留A4中真实验证并冻结；不能由N推导或仅凭选择宣称已通过。每owner物理步及输入预算与全服务PCM预算 MUST 分开说明，不能增加队列即声称增加执行容量；loader、根/Docker模板、README及部署说明的有效值与迁移规则保持一致。

#### Scenario: 检查默认配置
- **WHEN** 运维读取config.toml或docker/config.toml
- **THEN** 能区分HTTP1、Qwen N4、每音频S16、总4/全局64、B4、独立CAM4/emo1及PCM预算；根配置设备为宿主npu:7，单卡容器按物理7实际可见映射使用逻辑npu:0，不把音频容量当模型池份数

#### Scenario: 功能关闭时模型池不生效
- **WHEN** 对应增强设备数组为空
- **THEN** 对应模型不触发加载或推理；重新填写非空设备数组并重启才能开启，Qwen默认预加载不绕过增强数组关闭

### Requirement: 增强音频并发配置具有确定默认值
系统 MUST 从audio_spk_emo_concurrency读取B配置上限，缺省4，正整数且重启生效，根与Docker模板均展示；关闭能力使用空设备数组，不以B上限或旧features开关表达。正式continuous的B新准入有效上限 MUST 不超过min(配置B上限,健康Qwen容量)，不得据此改写配置或复制增强池。健康下降时closing请求仍计原活动，安全终态前不伪释放；各HTTP服务独立维护，不是跨服务全局上限。

#### Scenario: 缺少新增配置项
- **WHEN** 旧配置未声明audio_spk_emo_concurrency
- **THEN** 服务使用B配置上限4，不自动回退为N；文档说明有效上限还受总音频与健康owner约束

#### Scenario: 显式配置其他上限
- **WHEN** 运维显式配置continuous N=8、相等旧audio_concurrency=8、audio_spk_emo_concurrency=2并重启
- **THEN** 每个HTTP服务配置总容量8、B上限2，健康时最多8音频且至多2B；旧相等容量仅兼容并给迁移提示

#### Scenario: 非法增强上限
- **WHEN** B配置值为0、负数、布尔值、浮点数或字符串
- **THEN** 就绪前报告字段和正整数约束，不静默纠正为1，不把0当作功能关闭

#### Scenario: 增强上限大于总上限
- **WHEN** continuous N=2且B配置上限4，两个owner健康
- **THEN** 配置有效，B有效至多2；状态仍分别报告配置总2和B4及有效值，不静默改写B配置

#### Scenario: 多服务实例
- **WHEN** 同一部署显式启动多个完整HTTP服务
- **THEN** 每个服务独立维护总音频、B和其健康owner容量，文档说明这些不是跨服务共享上限，不能用多服务成绩代替本次单HTTP验收

### Requirement: 明确实例设备与 TensorRT 配置的优先级
模型设备数组 MUST 接受 cpu 与 npu:N。NPU MUST 固定走 torch_npu，CPU MUST 使用 PyTorch CPU；增强能力启用 MUST 由设备数组是否非空决定，请求能力与设备数组容量含义保留。系统 MUST NOT 保留仅对 NVIDIA 实例生效的加速器开关与引擎精度配置项。

#### Scenario: NPU 配置同时打开开关
- **WHEN** `[model_device_pool]` 中某实例为 npu:0
- **THEN** 该实例实际后端为 torch_npu，不要求任何 NVIDIA engine 或 TensorRT 依赖

#### Scenario: 重复 CUDA 设备
- **WHEN** CAM++ 数组为三个 npu:0
- **THEN** 配置容量仍为 3 个独立实例，不因共用同一设备或权重文件变成单实例

#### Scenario: 功能关闭时的配置保留
- **WHEN** 对应模型设备数组为空
- **THEN** 当前不加载或调用该模型；重新配置非空数组并重启后开启，请求关闭能力时沿用准入前错误

### Requirement: 文档明确分类限制与能力错误
README和部署说明 MUST 说明最终默认N4/S16、派生总4/global64、B4及CAM4/emo1、请求owner生命周期、健康降级、等待调度及按HTTP服务生效范围。错误码仍区分4013、4015、4016、4017、4018及等待4010，保留能力关闭优先和总容量满优先于B满；配置关闭、模型池不可用与普通段级失败降级不能混淆。legacy和旧A8矩阵保留各自历史范围，不用最终N4或历史N6/N5结果改写。

#### Scenario: 查阅默认并发示例
- **WHEN** 用户查阅正式最终默认总4、B4的说明
- **THEN** 文档说明4A、4B或总计4个混合请求使总满，新A/B均4013优先；本配置不能单独触发4017。显式N6下1A+4B新B4017、新A可形成2A+4B的通用契约及原实测保留为历史，不伪装N4可达状态；B双开仍只占一个B槽

#### Scenario: 查阅能力关闭行为
- **WHEN** 用户查询对应设备数组为空、增强请求参数为true的处理规则
- **THEN** 文档说明准入前请求失败及4015/4016/4018单项组合错误，不承诺配置关闭时以null成功；普通A仍按自身容量处理，已开启模型的普通段级失败另列null降级

### Requirement: 启动前校验部署前置条件
昇腾部署 MUST 在就绪前校验设备可见性、必需资产、torch_npu运行时及CONFIG_PATH可读性，失败明确报告并阻止就绪。三个必需模型资产与Qwen权重策略分开：preload=false时Qwen缺失在转写触发按4011处理，不为预检加载；正式默认preload=true时全部N份权重及processor须顺序加载成功且健康后才就绪，任何部分启动失败统一回收。旧features开关MUST在任何worker/模型创建前给出迁移错误；前置校验本身不得加载模型或绕过空数组关闭；校验通过后，HTTP就绪前须完成非空设备数组增强池的预加载尝试，失败按增强池降级契约处理。

#### Scenario: GPU 或资产不可用
- **WHEN** 容器无可用NPU、必需模型目录缺失或配置不可读
- **THEN** 启动指出具体前置条件并失败，不接受请求；默认Qwen预加载失败同样阻止就绪并回收已启动owner，显式懒加载的Qwen缺失仍按原4011处理

#### Scenario: 前置条件通过
- **WHEN** 设备、目录和配置检查通过，且按当前preload策略完成必要加载
- **THEN** 服务就绪并报告镜像tag、实际配置路径、后端和N份实际健康加载状态；不把预检、READY或模型名当真实语音forward暖身
