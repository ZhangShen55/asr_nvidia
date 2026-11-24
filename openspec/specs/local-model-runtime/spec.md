# local-model-runtime Specification

## Purpose

定义现行 v3 链路的模型承载与资源归属：CAM++、emotion2vec 按启动预加载策略承载，本地 Qwen 按自身加载策略承载，VAD 由启动时以 spawn 创建并预热的固定独立 worker 池承载并按 FIFO 调度。状态与运维观测须区分主服务和 VAD worker、配置与健康容量，以及各进程的 HBM 和主机资源口径。

## Requirements

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

### Requirement: 实例数量与显存放大的关系可控
系统 MUST 分开表达完整HTTP服务数instance_count、每服务Qwen owner数N及CAM++/emotion2vec各自数组决定的副本数。正式多owner默认及本次验收 MUST 使用instance_count=1、N=4（合法N1～8保留），增加N不得复制整套HTTP、VAD或增强池。显式多HTTP部署的每服务仍独立持有其请求容量及模型资源，资源评估 MUST 计入全部副本，不将各服务局部上限称作跨进程全局上限。

#### Scenario: 默认单实例
- **WHEN** 配置中未显式设置instance_count且两个模型设备数组各有一个元素
- **THEN** 服务以1个完整HTTP实例运行，两个增强池各1份；正式continuous的四Qwen owner属于该HTTP实例，不按Qwen数复制增强池；此场景的显式各1数组是通用配置示例，最终交付数组为CAM4/emo1

#### Scenario: 配置多个实例
- **WHEN** 配置中instance_count为2且模型设备数组保持不变
- **THEN** 原显式多服务部署运行2个完整服务，各自独立建立增强池、VAD池和配置数量的Qwen owner，并按各自预加载或懒加载契约加载；HBM评估包含这些副本，池容量不在服务间合并，该配置不冒称本次单HTTP验收

#### Scenario: 只增加Qwen实例
- **WHEN** 一个HTTP服务内将N从一个合法值改为另一个合法值并重启，增强及VAD设备数组未改
- **THEN** 只改变Qwen owner数量及其派生容量，VAD和两个增强池的配置实例数不随N倍增

### Requirement: 本地增强模型池按设备数组建立固定实例
系统 MUST 从model_device_pool的两个设备数组建立每服务共享的CAM++与emotion2vec独立模型池，每个数组元素对应一个模型实例，数组长度决定配置实例数；在功能开启且数组非空时于HTTP就绪前按数组顺序完成加载尝试。各Qwen owner及不同请求 MUST 复用这两个池，不各自复制池；最终交付配置按用户选择CAM4/emo1，在四Qwen真实共驻B4下标定并冻结，不从N、S或B4推导；未验收不得宣称已通过。

#### Scenario: CAM++ 与 emotion2vec 实例数量不同
- **WHEN** CAM++数组有3个元素、emotion2vec数组有1个元素
- **THEN** 两个模型池分别最多提供3个和1个健康执行实例，不因数量不同而补齐、共享或为每个Qwen owner复制一组

#### Scenario: 多卡设备数组
- **WHEN** 同一模型数组包含多个npu:N设备
- **THEN** 每个实例加载到对应数组元素指定的设备，并在状态接口按实例顺序展示设备；本次宿主验证仍仅使用物理7号卡

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

### Requirement: 模型池并发容量由各自设备数组决定
系统 MUST 将各自成功加载的设备数组元素数作为CAM++和emotion2vec的最大执行并发，两个模型池独立排队，禁止固定比例、共享模型槽位或Qwen owner数隐式绑定容量。整音频B上限4及Qwen N、S的改变 MUST NOT 改变增强池实例数、预取深度或实际计算批大小；每项容量与实际健康数分别报告。

#### Scenario: 两个模型配置不同数量实例
- **WHEN** CAM++设备数组有3个元素且emotion2vec设备数组有1个元素
- **THEN** CAM++最多同时执行3个推理，emotion2vec最多同时执行1个推理，两个池互不借用容量

#### Scenario: 配置数组为空
- **WHEN** 任一模型设备数组为空
- **THEN** 对应模型池容量为0、不加载该模型实例，要求该能力的请求得到明确不可用结果，不能以Qwen有空闲owner为由伪造增强成功

#### Scenario: 部分实例加载失败
- **WHEN** 设备数组声明多个实例但其中部分实例加载失败
- **THEN** 模型池最大并发按成功加载的实例数计算，失败原因可查询且不占用可用槽位；不把配置数当健康容量

### Requirement: 模型池状态区分独立容量与等待

系统 MUST 在状态接口中分别暴露 CAM++ 与 emotion2vec 的配置实例数、成功加载实例数、可用实例数、busy、waiting 和失败实例数；状态 MUST 能区分模型池等待与其它服务队列等待。

#### Scenario: 查询独立模型池状态
- **WHEN** CAM++ 配置 3 个实例且 emotion2vec 配置 1 个实例并均加载成功
- **THEN** 状态分别显示 `configured_instances=3` 和 `configured_instances=1`，并显示各自 busy、waiting、available_instances

#### Scenario: 一个模型池排队而另一个空闲
- **WHEN** emotion2vec 的可用实例全部忙碌而 CAM++ 仍有可用实例
- **THEN** emotion2vec 的 waiting 增加，CAM++ 的 waiting 不因 emotion2vec 排队而增加

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

### Requirement: 主服务与固定 VAD worker 分工承载模型
正式continuous/process MUST 由一个主HTTP进程协调N个owned spawn Qwen进程，每份仅在所属worker加载权重；CAM++按既有配置承载方式使用共享独立池，emotion2vec维持既有共享runner。VAD MUST 保留启动预热的固定spawn池，按数组长度配置并FIFO分派完整音频，不改主进程内VAD或运行期扩缩容。主进程保持asr-offline-model、VAD保持asr-offline-vad；运维 MUST 按出生身份区分各worker资源。legacy保留既有承载方式。

#### Scenario: 服务启动并预热 VAD
- **WHEN** 服务按固定VAD设备数组启动
- **THEN** spawn worker完成VAD预热并满足可用条件后才就绪，不因此加载未请求的增强模型；Qwen另按preload开关完成全部预加载或保持懒加载，不能跳过默认预加载就绪条件

#### Scenario: 查询进程资源
- **WHEN** 运维查看NPU进程及服务状态
- **THEN** 主服务、VAD、各Qwen及已配置进程模式CAM可按PID和出生身份区分；父与worker资源分别统计，无法归因者明确标记，查询不触发懒加载，不把父进程HBM当作全部worker总量

#### Scenario: 本轮宿主开发使用指定设备
- **WHEN** 在xjtu910b7的conda asr环境开展本轮开发
- **THEN** 使用/home/zhangs/conda-envs/asr/bin/python及宿主npu:7，不调度其他卡或终止既有服务；容器逻辑编号按实际可见映射核对，不照搬宿主卡号

### Requirement: B类三分支独立排队且按原段汇合
请求双增强时，有效VAD段裁剪后 MUST 并发分发给Qwen、CAM++和emotion2vec，各分支仅使用所属窗口及模型池，按请求ID和原段ID汇合；不得等待Qwen结束才发增强。Qwen安全完成 MUST 立即归还自身段credit，慢增强不能在共享PCM预算有余时占住该credit。CAM的0.3秒有效段覆盖、emo仅1～10秒实际推理及其它有效时长平淡、原聚类和情绪映射 MUST 保持。

#### Scenario: 四音频同时双增强
- **WHEN** 最终四Qwen、CAM4/emo1均驻留，四请求showSpk=true、showEmotion=true且能力可用
- **THEN** 三分支独立排队并实际调用，CAM和emo实例由各自共享池提供；不为各请求或各Qwen复制增强池，结果保持原音频与原段归属

#### Scenario: 情绪队列慢于转写
- **WHEN** 某段Qwen安全结束而emo仍等待或执行，且全服务PCM预算有余
- **THEN** 该段Qwen credit已释放，同请求下一段可进入Qwen；最终响应仍等待该段所有已启动分支终态，不串行化后续增强任务

#### Scenario: 短段和情绪时长边界
- **WHEN** 实际裁剪段分别不足0.3秒、0.3至不足1秒、1至10秒闭区间或超过10秒
- **THEN** 不足0.3秒在三模型前移除；其余有效段执行Qwen及所请求CAM，只有1至10秒实际推emo，其它有效时长按既有规则平淡并明确计skip，不能将skip算作NN成功

### Requirement: PCM汇合预算保持全服务有界所有权
活动PCM段数与字节预算 MUST 为单HTTP服务共享，缺省256段/268435456字节，不按N复制，也不等同于N×S逻辑段窗口。每段PCM MUST 持有到全部已启动分支及必要读取清理终态；正常、失败、跳过和取消均只释放一次。预算耗尽 MUST 显式反压，不能无限堆积PCM、future或汇合结果；完整请求owner及音频/B租约到整请求安全收尾才归还。

#### Scenario: 先完成转写但增强未结束
- **WHEN** 一个段的Qwen已释放credit但CAM或emo仍有在途工作
- **THEN** 该段PCM继续受全服务预算持有，归还Qwen credit不提前归还PCM，也不把此PCM计作仍在Qwen执行

#### Scenario: 共享预算达到上限
- **WHEN** 多音频的未完成分支使256段或256MiB PCM任一预算达到上限
- **THEN** 暂停新增PCM工作并分别报告使用量和反压原因，释放后继续；不能每owner各使用一份同额预算或让一个模型实例等待另一池实例形成循环等待

#### Scenario: 分支取消或失败后汇合
- **WHEN** 已启动分支在设备访问期间失败、取消或遭重复取消
- **THEN** 等实际工作及必要清理终态后回收PCM；段级增强失败维持原null降级与统计，模型全部不可用维持4015/4016，错误不导致提前释放owner或其它请求资源
