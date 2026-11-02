# Spec Delta

## ADDED Requirements

### Requirement: v2 本地增强能力按模型池独立排队

系统 MUST 让每个 VAD 段的 CAM++ 请求进入 CAM++ 独立模型池、emotion2vec 请求进入 emotion2vec 独立模型池；本地增强能力的等待 MUST 不改变 Qwen-ASR 的请求并发配置，也 MUST NOT 让一个模型池借用另一个模型池的实例。

#### Scenario: CAM++ 槽位少于 emotion2vec 槽位
- **WHEN** CAM++ 设备数组配置 3 个实例且 emotion2vec 设备数组配置 1 个实例
- **THEN** CAM++ 和 emotion2vec 分别按自身容量排队，空闲的 CAM++ 实例不能被 emotion2vec 请求占用

#### Scenario: 本地模型池等待期间 Qwen 仍可调度
- **WHEN** 某个 VAD 段等待 CAM++ 或 emotion2vec 实例
- **THEN** 其它已获得本地模型实例的段仍可并行调用 Qwen，且 Qwen 的 `segment_concurrency` 与 `request_concurrency` 上限保持生效

### Requirement: 记录本地模型池排队对端到端耗时的影响

系统 MUST 为 v2 请求记录 CAM++ 和 emotion2vec 的调用次数、推理耗时和模型池等待耗时，并 MUST 能在测试报告中分别比较本地模型等待、VAD、Qwen 分段处理和其它后处理阶段。

#### Scenario: 8 路完整音频压测
- **WHEN** 使用同一份完整音频并发提交 8 个 `showSpk=true`、`showEmotion=true` 请求
- **THEN** 结果包含两个模型池各自的 busy/waiting、等待耗时、P50/P95、错误数和音频吞吐

#### Scenario: 模型池实例比例变化
- **WHEN** 仅改变两个设备数组的元素数量而不新增槽位配置
- **THEN** 测试可以比较不同模型实例组合对 CAM++ 等待、emotion2vec 等待和端到端效率的影响

## MODIFIED Requirements

### Requirement: 异步并行计算真实角色与情绪

系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并 MUST 与 Qwen 转写并行执行：每个 VAD 段的转写任务与该段的 role/emotion 任务 MUST 在同一时刻发起，MUST NOT 等待全部转写完成后再计算 role/emotion。CAM++ 与 emotion2vec 的并发容量 MUST 分别由各自设备数组的成功实例数决定，MUST NOT 受共享模型闸门限制；两个模型 MUST 分别由独立配置开关控制，emotion2vec MUST 以半精度加载。

#### Scenario: 开启角色与情绪能力
- **WHEN** 请求参数 `showSpk` 与 `showEmotion` 均为 true
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行并分别使用对应模型池

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk` / `showEmotion`（默认 false）
- **THEN** 每个分段的 `role` 与 `emotion` 均为 `null`，两个模型池均不因该请求加载

#### Scenario: 配置未开启说话人能力
- **WHEN** 配置中 `open_v2_role` 为 false
- **THEN** 系统不加载 CAM++ 模型，响应分段的 `role` 保持 `null`

#### Scenario: 配置未开启情绪能力
- **WHEN** 配置中 `open_v2_emotion` 为 false
- **THEN** 系统不加载 emotion2vec 模型，响应分段的 `emotion` 保持 `null`

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** 配置中 `open_v2_role` 与 `open_v2_emotion` 中任一为 false
- **THEN** 系统只跳过该开关对应的模型池加载，另一个模型仍可按请求参数独立启用

#### Scenario: 角色或情绪推断失败
- **WHEN** 说话人或情绪模型推理失败
- **THEN** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **WHEN** emotion2vec 模型被加载
- **THEN** 模型以半精度（fp16）加载，情绪精度不做额外调优

### Requirement: 验证角色与情绪的效果与性能

系统 MUST 在 `test/audio_file/` 的音频上验证 role 与 emotion 的效果与性能：与 `/v1.1.8` 链路的逐段结果比对一致率，并测量开启后的时延、吞吐与显存增量；验证 MUST 覆盖由设备数组决定的非对称模型实例组合，并记录模型池各自的等待情况。

#### Scenario: 角色与情绪效果比对
- **WHEN** 在同一音频、同一 VAD 参数下分别调用 `/v2/seacraft_asr`（`showSpk=true`、`showEmotion=true`）与 `/v1.1.8/seacraft_asr`
- **THEN** 逐段比较 role 的角色语义与 emotion 的业务标签，记录一致率与不一致样本

#### Scenario: 8 路非对称模型池压测
- **WHEN** CAM++ 与 emotion2vec 均部署在 `cuda:0`，设备数组分别配置 6 个和 2 个元素，并发提交 8 个完整音频请求
- **THEN** 记录端到端 P50/P95、CAM++/emotion2vec 各自 busy/waiting、GPU 利用率、显存、错误和吞吐；6:2 仅作为测试组合，不成为固定默认值

#### Scenario: 并行开销可接受
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **THEN** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **WHEN** 在非对称模型实例组合下运行 role/emotion 全开
- **THEN** 记录本机 GPU 的 SM 利用率与显存占用增量、吞吐（音频/分钟、段/秒）以及两个模型池的等待耗时，并写入中文测试报告
