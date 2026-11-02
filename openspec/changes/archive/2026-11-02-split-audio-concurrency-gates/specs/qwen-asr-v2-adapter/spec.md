# Spec Delta

## MODIFIED Requirements

### Requirement: 限制同时处理的音频数量
系统 MUST 使用 `audio_concurrency` 限制 A+B 同时处理的音频文件总数（默认 8）。每个请求 MUST 从准入成功持有槽位至处理及该请求全部临时文件清理完成，并在成功、异常或取消时准确释放。默认不等待，超过总上限 MUST 立即返回总处理队列满；B 类请求另受增强音频并发上限限制。

#### Scenario: 请求数超过音频并发上限
- **WHEN** `queue_wait_timeout_s=0` 且活动音频总数已达到 `audio_concurrency`
- **THEN** 新的 A 或 B 请求立即返回 `code=4013`，`msg` 包含当前总音频并发 `active/capacity`；即使 B 容量也已满，仍优先返回 `4013`

#### Scenario: 排队等待超时
- **WHEN** 配置了大于 0 的排队等待超时，且请求在等待中超过该时长仍未获得完整处理槽位
- **THEN** 系统返回 `{"msg": "请求排队超时，请稍后再试", "code": 4010}`，并回收等待名额

#### Scenario: 队列容量已满
- **WHEN** 等待模式下请求不能立即获准且 A+B 共享等待队列长度达到 `queue_max_size`
- **THEN** 系统返回 `{"msg": "服务繁忙，请稍后再试", "code": 4013}`，不占用活动槽位或增加等待计数

#### Scenario: 音频处理结束释放槽位
- **WHEN** 一个音频请求处理完成且其临时文件已清理
- **THEN** 系统准确释放其总音频槽位；B 同时释放增强槽位，并使最早满足容量条件的等待请求开始处理

#### Scenario: 音频处理异常时释放槽位
- **WHEN** 一个音频请求在处理过程中抛出异常
- **THEN** 系统清理该请求产生的全部临时文件后准确释放其全部活动槽位，不影响后续请求

### Requirement: 保留扩展字段
`/v2/seacraft_asr` 的每个成功响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组。`role`、`emotion` 在对应能力未被请求或已启用模型的段级推断失败时为 `null`；请求要求的能力被配置关闭时 MUST 返回请求级错误，不得作为成功转写返回 `null`。

#### Scenario: 返回扩展字段
- **WHEN** 成功响应包含分段
- **THEN** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **WHEN** `showSpk`、`showEmotion`、`showRoleIdentify` 均未提供或均为 `false`
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`

### Requirement: 通过配置外置运行参数
Qwen 适配链路的服务地址、模型名、请求并发度、总音频并发度、增强音频并发度、队列容量、超时和 VAD 参数 MUST 来自配置文件；VAD 实例数 MUST 只由 `vad_device` 数组长度决定，动态 VAD 池参数 MUST 不再作为有效配置。

#### Scenario: 修改并发参数后重启生效
- **WHEN** 运维把 `vad_device` 改为 8 个元素、`segment_concurrency` 改为 16、`audio_concurrency` 改为 8、`audio_spk_emo_concurrency` 改为 4 并重启服务
- **THEN** 服务启动 8 个固定 VAD worker，并按新的 Qwen、总音频和增强音频并发参数运行

#### Scenario: 配置缺省时使用默认值
- **WHEN** 配置文件缺少相关新配置项
- **THEN** 服务使用默认 `vad_device=["cpu","cpu","cpu","cpu","cuda:0","cuda:0","cuda:0","cuda:0"]`、`segment_concurrency=16`、`request_concurrency=128`、`audio_concurrency=8`、`audio_spk_emo_concurrency=4`，默认满额立即拒绝

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并与 Qwen 转写并行执行：每个 VAD 段的转写与所请求的 role/emotion 任务 MUST 同时发起。CAM++ 与 emotion2vec 的段级并发容量 MUST 分别由各自设备数组的成功实例数决定，不增加共享模型推理闸门；B 类整音频准入限制不改变模型池实例容量。两个模型 MUST 分别由独立配置开关控制，emotion2vec MUST 以半精度加载；请求关闭的能力 MUST 在音频准入前报错。

#### Scenario: 开启角色与情绪能力
- **WHEN** 两个功能开关均为 `true`，请求 `showSpk=true`、`showEmotion=true` 且模型推断成功
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行并分别使用对应模型池

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk`、`showEmotion`、`showRoleIdentify`（均默认 `false`）
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`，两个模型池均不因该请求加载

#### Scenario: 配置未开启说话人能力
- **WHEN** `open_v2_role=false` 且请求需要说话人能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4015`，不开始该请求的模型加载或转写；未请求说话人能力的请求不因该开关报错

#### Scenario: 配置未开启情绪能力
- **WHEN** `open_v2_emotion=false` 且请求需要情绪能力，没有同时请求另一项已关闭的能力
- **THEN** 系统在音频准入前返回 `4016`，不开始该请求的模型加载或转写；未请求情绪能力的请求不因该开关报错

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** `open_v2_role` 与 `open_v2_emotion` 中至少一项为 `false`
- **THEN** 仅请求已开启能力时正常进入 B 处理；请求任何关闭的能力时整个请求失败；同时请求两项且两项都关闭时返回 `4018`

#### Scenario: 角色或情绪推断失败
- **WHEN** 请求的增强能力均已开启且存在可用模型实例，但某个分段的说话人或情绪推理失败
- **THEN** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **WHEN** emotion2vec 模型被加载
- **THEN** 模型以半精度（fp16）加载，情绪精度不做额外调优

## ADDED Requirements

### Requirement: 按请求的增强能力分类音频
系统 MUST 将 `showSpk`、`showEmotion`、`showRoleIdentify` 均为 `false` 的请求归为 A，任一为 `true` 归为 B；三个参数默认值均为 `false`。`showSpk` 或 `showRoleIdentify` 为 `true` MUST 视为需要说话人能力，`showEmotion=true` MUST 视为需要情绪能力；每个音频至多占用一个增强槽位。

#### Scenario: 默认普通转写
- **WHEN** 请求未提供三个增强参数
- **THEN** 请求归为 A，只消耗一个总音频槽位，且参数文档展示三个默认值均为 `false`

#### Scenario: 单项增强请求
- **WHEN** 请求仅设置 `showSpk=true` 或仅设置 `showEmotion=true`，对应能力已开启
- **THEN** 请求归为 B，同时消耗一个总音频槽位和一个增强槽位

#### Scenario: 角色识别请求
- **WHEN** 请求仅设置 `showRoleIdentify=true` 且说话人能力已开启
- **THEN** 请求归为 B，需要说话人能力并同时消耗一个总音频槽位和一个增强槽位

#### Scenario: 同时请求多项增强
- **WHEN** 请求同时开启两个或三个增强参数且所需能力均已开启
- **THEN** 请求归为 B，仅占用一个总音频槽位及一个增强槽位

### Requirement: 增强音频容量是总容量的子集
系统 MUST 始终满足 `active_A + active_B <= audio_concurrency` 且 `active_B <= audio_spk_emo_concurrency`。A 仅占总容量，B 原子取得总容量和增强容量；等待或拒绝的请求 MUST NOT 预占任何活动槽位。默认不等待时，总容量满返回 `4013`；仅 B 容量满返回 `4017`，消息 MUST 标识 `spk_emo` 和增强容量使用比。

#### Scenario: 三个 A 和四个 B
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `3A+4B`
- **THEN** 新 B 返回 `4017` 和 `spk_emo队列已满，请稍后再试（当前增强音频并发：4/4）`；新 A 可获准形成 `4A+4B`

#### Scenario: 五个 A 和三个 B
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `5A+3B`
- **THEN** 新 A 和新 B 均返回 `4013`，B 不因还有一个增强名额而获准

#### Scenario: 八个普通音频占满总容量
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `8A`
- **THEN** 新 A 和新 B 均返回 `4013`，系统不为 B 预留总容量

#### Scenario: 总容量和增强容量同时已满
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `4A+4B`
- **THEN** 新 A 和新 B 均返回 `4013`，总容量错误优先于增强容量错误

#### Scenario: 两类容量均有余量
- **WHEN** 总上限为 8、增强上限为 4，正在处理 `4A+3B` 且无等待请求
- **THEN** 下一请求无论为 A 或 B 均可获准，所有容量约束仍成立

#### Scenario: 增强容量满不占满总容量
- **WHEN** 总上限为 8、增强上限为 4、默认不等待，正在处理 `0A+4B`
- **THEN** 新 B 返回 `4017`，新 A 可获准且增强活动数保持 4

### Requirement: 关闭能力在准入之前拒绝
系统 MUST 在音频准入、等待排队、音频预处理和模型调用之前校验请求所需的功能开关。仅所请求的说话人能力关闭返回 `4015`，仅所请求的情绪能力关闭返回 `4016`，同时请求两项且两项都关闭返回 `4018`。业务错误 MUST 保持 HTTP 200 和 `{"msg": ..., "code": ...}` 格式，不返回成功转写分段。

#### Scenario: 两种能力关闭但只做普通转写
- **WHEN** 两个功能开关均为 `false` 且三个增强参数均为 `false`
- **THEN** 请求按 A 处理，只受总音频容量限制，增强活动数不增加

#### Scenario: 两种能力关闭且同时请求两种增强
- **WHEN** 两个功能开关均为 `false`，请求需要说话人及情绪能力
- **THEN** 请求返回 `4018`，消息说明两种能力均未启用；即使总容量已满也不返回 `4013`，且所有活动、等待计数不变

#### Scenario: 同时请求两种能力但只有一种关闭
- **WHEN** 请求需要说话人及情绪能力，恰好只有一个功能开关为 `false`
- **THEN** 说话人关闭返回 `4015`，情绪关闭返回 `4016`，请求整体失败且不进行部分转写

#### Scenario: 两种能力关闭但仅请求一种
- **WHEN** 两个功能开关均为 `false`，但请求仅需要说话人能力或仅需要情绪能力
- **THEN** 分别返回 `4015` 或 `4016`，不使用组合错误 `4018`

### Requirement: 等待队列按可用容量调度
在 `queue_wait_timeout_s>0` 时，系统 MUST 以 `queue_max_size` 限制 A+B 的共享等待总数，保持同类请求的到达顺序，并优先准入最早满足容量条件的请求。B 被增强容量阻塞但总容量有余量时，A MUST 能绕过该 B；当较早 B 已满足两类容量条件时，较晚 A MUST NOT 抢占其名额。

#### Scenario: A 绕过被增强容量阻塞的 B
- **WHEN** `3A+4B`、总上限 8、增强上限 4，等待队列已有 B，此时新 A 到达
- **THEN** B 继续等待且不预占总槽位，A 能使用空闲总槽位；B 不因本次 A 获准而失去其同类顺序

#### Scenario: 较早 B 变为可处理
- **WHEN** 较早等待的 B 和较晚等待的 A 同时满足容量条件
- **THEN** B 先获准，不被后到 A 持续抢占；同类请求按到达顺序处理

#### Scenario: 两类共享等待上限
- **WHEN** A 等待数与 B 等待数之和达到 `queue_max_size`，新请求也不能立即获准
- **THEN** 新请求返回 `4013`，不会因分成两类而把等待容量扩大为两倍

#### Scenario: 等待 B 不阻塞空闲总容量
- **WHEN** B 仅因增强容量不足而等待，其他容量仍有余量
- **THEN** 该 B 不占用总音频活动名额或增强活动名额，等待超时仍返回 `4010`

### Requirement: 取消和并发竞争不泄漏名额
系统 MUST 在等待取消、准入与取消竞争、准入与超时竞争、处理取消和清理异常时保持计数一致，每个已授予的活动名额至多释放一次且最终必定释放；未授予的请求只移除等待名额。B 的两个活动名额 MUST 作为同一个处理周期取得和释放。

#### Scenario: 等待中取消
- **WHEN** 等待中的 A 或 B 请求被取消
- **THEN** 对应等待数回收且活动计数不增加，后续满足容量条件的请求可继续获得名额

#### Scenario: 获准与取消同时发生
- **WHEN** 槽位刚授予请求时该请求同时取消，调用方尚未开始处理
- **THEN** 系统回收已授予的全部槽位，不泄漏、不重复释放，也不产生负数计数

#### Scenario: 获准与等待超时竞争
- **WHEN** 等待截止时刻与槽位可用同时发生
- **THEN** 请求仅有一种结果：成功取得完整名额并负责释放，或返回 `4010` 且不保留名额

#### Scenario: 处理取消或清理异常
- **WHEN** 已获准的 B 请求处理取消，或其临时文件清理发生异常
- **THEN** 系统完成必要的清理尝试并释放一个总音频槽位及一个增强槽位，保留原有错误或取消语义
