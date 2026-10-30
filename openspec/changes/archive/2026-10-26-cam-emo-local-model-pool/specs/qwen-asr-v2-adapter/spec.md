# Spec Delta

## MODIFIED Requirements

### Requirement: 配置 v2 链路模型的落卡

系统 MUST 支持通过 `[model_device]` 指定 v2 链路三个本地模型的设备；VAD 使用单实例，CAM++ 与 emotion2vec 的模型池实例 MUST 按各自 `*_devices` 数组逐实例绑定 GPU，默认数组为 `["cuda:0"]`。本变更 MUST NOT 自动迁移实例。

#### Scenario: 修改落卡配置
- **WHEN** 运维人员把三个模型的设备配置改为 `cuda:1` 且每个模型池只声明一张卡
- **THEN** VAD、CAM++ 全部副本与 emotion2vec 全部副本均加载到 GPU 1

#### Scenario: 多副本按设备数组加载
- **WHEN** CAM++ 配置 `speech_campplus_devices = ["cuda:1", "cuda:2"]`
- **THEN** 两个对应副本分别加载到 GPU 1 和 GPU 2，不会被自动迁移

#### Scenario: 配置缺省
- **WHEN** 配置文件中缺少落卡配置
- **THEN** 三个模型及其已启用副本均加载到 GPU 0

#### Scenario: 拒绝非法设备数组
- **WHEN** 同一模型池声明空数组或非法 GPU 设备字符串
- **THEN** 服务在配置校验阶段拒绝启动并报告对应数组字段

### Requirement: 异步并行计算真实角色与情绪

系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并 MUST 与 Qwen 转写并行执行；每个 VAD 段的转写任务与该段的 role/emotion 任务 MUST 在同一时刻发起，MUST NOT 等待全部转写完成后再计算 role/emotion。两个模型 MUST 分别由独立配置开关控制（`open_v2_role` 与 `open_v2_emotion`，默认均为 true）；模型池实例由 `[model_device.local_model_pool]` 的设备数组配置，实际计算仍由请求参数 `showSpk` / `showEmotion` 触发，emotion2vec MUST 以半精度加载。

#### Scenario: 开启角色与情绪能力
- **WHEN** 请求参数 `showSpk` 与 `showEmotion` 均为 true，且两个模型池均有可用副本
- **THEN** 响应分段的 `role` 与 `emotion` 为真实推断值，且每段与转写同时并发执行

#### Scenario: 多副本并行处理
- **WHEN** 同一请求包含多个 VAD 段且模型池配置多个副本
- **THEN** 不同段可以分派到不同副本，所有段仍在各自转写任务完成后返回同构的 role/emotion 字段

#### Scenario: 角色与情绪能力未开启
- **WHEN** 请求未携带 `showSpk` / `showEmotion`（默认 false）
- **THEN** 每个分段的 `role` 与 `emotion` 保持 `null`，对应模型池不被请求触发加载

#### Scenario: 配置未开启说话人能力
- **WHEN** 配置中 `open_v2_role` 为 false
- **THEN** 系统不加载 CAM++ 模型池，响应分段的 `role` 保持 `null`

#### Scenario: 配置未开启情绪能力
- **WHEN** 配置中 `open_v2_emotion` 为 false
- **THEN** 系统不加载 emotion2vec 模型池，响应分段的 `emotion` 保持 `null`

#### Scenario: 配置未同时开启说话人与情绪能力
- **WHEN** 配置中 `open_v2_role` 与 `open_v2_emotion` 中任一为 false
- **THEN** 系统只跳过对应模型池，另一个模型仍可按请求参数独立启用

#### Scenario: 角色或情绪推断失败
- **WHEN** 某个模型池副本推理失败或所有副本暂时不可用
- **THEN** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **WHEN** emotion2vec 模型池的任一副本被加载
- **THEN** 每个副本均以半精度（fp16）加载，情绪标签格式与单副本模式一致

### Requirement: 验证效果与性能边界

系统 MUST 使用 `test/audio_file/` 中的完整音频完成 role/emotion 全开压测，覆盖 2、4、8、16、32 路音频并发，并以单实例模型池为 baseline 记录端到端时延、吞吐、GPU 算力占用、显存、排队、失败率、转写准确度和时间戳准度；测试必须形成中文可归档报告并逐档说明是否提升。

#### Scenario: 完成并发压测
- **WHEN** 开发人员在目标服务器上以 `showSpk=true`、`showEmotion=true` 对完整音频发起 2、4、8、16、32 路并发
- **THEN** 每个档位均记录单实例 baseline 与模型池配置的样本数、成功失败数、P50/P95/max 时延、音频吞吐和段吞吐

#### Scenario: 采集同卡资源指标
- **WHEN** 多副本全部运行在 `cuda:0`
- **THEN** 报告记录 GPU 利用率、显存峰值与增量、模型池 busy/waiting、Qwen waiting/KV 等指标，并标注同卡抢占现象

#### Scenario: 判定性能提升
- **WHEN** 比较模型池与单实例 baseline
- **THEN** 报告逐档给出吞吐和端到端效率变化；至少一个主要并发档位必须有可重复的吞吐提升，若未达到则结论标记为未满足并保留完整数据

#### Scenario: 记录质量边界
- **WHEN** 并发压测完成
- **THEN** 报告记录转写准确度、时间戳准度以及幻觉、重复、超时和资源耗尽的首次出现并发边界

#### Scenario: 产出测试报告
- **WHEN** 验证完成
- **THEN** 仓库中留下中文可归档测试报告，说明各并发档位的效率变化、质量边界、同卡抢占代价和推荐副本数

### Requirement: 验证角色与情绪的效果与性能

系统 MUST 在相同完整音频和 VAD 参数下比较单实例与模型池的 role/emotion 结果，并测量开启后的时延、吞吐与显存增量；模型池扩容 MUST NOT 改变转写文本、分段数量或时间轴，且报告必须说明一致率与性能变化。

#### Scenario: 角色与情绪效果比对
- **WHEN** 在同一音频、同一 VAD 参数下分别调用单实例和多实例 `/v2/seacraft_asr`（`showSpk=true`、`showEmotion=true`）
- **THEN** 逐段比较 role 的角色语义与 emotion 的业务标签，记录一致率与不一致样本

#### Scenario: 并行开销可接受
- **WHEN** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **THEN** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅并给出可选处置手段

#### Scenario: 开启增强能力不破坏分段结构
- **WHEN** 比较单实例与模型池的分段结果
- **THEN** 两组的分段数量、每段起止时间和转写文本保持一致，角色编号只允许因聚类标签命名规则产生等价变化

#### Scenario: 显存与吞吐可记录
- **WHEN** 在 2 / 4 / 8 / 16 / 32 音频并发下运行 role/emotion 全开
- **THEN** 记录 GPU 0 的 SM 利用率与显存占用增量、吞吐（音频/分钟、段/秒）和各模型池副本数，并写入测试报告
