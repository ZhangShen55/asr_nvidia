# Spec Delta

## Purpose

定义基于 Qwen3-ASR 的 `/v2/seacraft_asr` 离线转写能力：本地 VAD 切分与时间轴映射、对 Qwen 服务的并发调用与音频排队、模型自动发现、与 `/v1.1.8` 同构的响应封装，以及后续真实说话人角色与情绪的异步并行计算。

## ADDED Requirements

### Requirement: 提供 Qwen 版转写接口
系统 MUST 提供 `POST /v2/seacraft_asr`，其 multipart 表单参数与 `POST /v1.1.8/seacraft_asr` 一致（音频文件、语言、词级时间戳、热词、说话人、情绪、语速、角色识别），且 `/v1.1.8/seacraft_asr` 的行为 MUST NOT 因本能力而改变。

#### Scenario: 提交音频到 v2 接口
- **当** 客户端向 `/v2/seacraft_asr` 上传有效音频
- **则** 系统返回与 `/v1.1.8/seacraft_asr` 同构的成功响应，顶层包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms`

#### Scenario: 未提交可选参数
- **当** 客户端只提交音频文件
- **则** 系统按 `language=auto`、`wordTimestamps=false`、`showSpk=false`、`showEmotion=false`、`showRoleIdentify=false` 处理

#### Scenario: v1.1.8 行为保持不变
- **当** 客户端向 `/v1.1.8/seacraft_asr` 提交请求
- **则** 系统仍按原有 Paraformer/Whisper 分流逻辑处理，响应结构与错误码不变

### Requirement: 以 VAD 段作为转写与时间轴单元
系统 MUST 在服务本地对音频做 VAD 切分，并以「1 个 VAD 段对应 1 次 Qwen 请求、对应 1 个响应分段」的方式组织结果，响应分段的 `bg` 与 `ed` MUST 取该 VAD 段的起止时间；系统 MUST NOT 合并相邻 VAD 段。

#### Scenario: 音频被切分为多段
- **当** 音频包含多段人声
- **则** 系统按 VAD 段逐段请求 Qwen 服务，响应分段的起止时间与对应 VAD 段边界一致

#### Scenario: 音频无人声
- **当** VAD 未检测到任何有效人声
- **则** 系统返回 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`

#### Scenario: 过滤过短语音段
- **当** 某 VAD 段时长小于配置的最小语音段时长
- **则** 系统不向 Qwen 服务发起该段的请求，且该段不产出响应分段

### Requirement: 自动发现 Qwen 服务模型
系统 MUST 在未显式配置模型名时，通过调用 Qwen 服务的 `/v1/models` 接口获取可用模型标识并使用之，且请求 MUST NOT 携带 API Key。

#### Scenario: 未配置模型名
- **当** 配置中的模型名为空
- **则** 系统从 `/v1/models` 返回的列表中选取模型标识用于转写请求

#### Scenario: 显式配置模型名
- **当** 配置中填写了模型名
- **则** 系统直接使用该模型名，不调用模型发现

#### Scenario: 模型发现失败
- **当** `/v1/models` 不可达或返回空列表且未显式配置模型名
- **则** 系统返回 `{"msg": "Qwen 服务不可用或未发现可用模型", "code": 4011}`

### Requirement: 解析 Qwen 转写结果
系统 MUST 解析 Qwen 服务返回的原始文本，剥离 `language <语种><asr_text>` 前缀，取出转写文本与识别语种；返回文本为空或语种为空的段 MUST NOT 产出响应分段。

#### Scenario: 返回带语种前缀的文本
- **当** Qwen 服务返回 `language Chinese<asr_text>识别文本`
- **则** 系统产出文本为 `识别文本`、语种为 `Chinese` 的分段

#### Scenario: 段内无有效语音
- **当** Qwen 服务返回 `language None<asr_text>`
- **则** 该段不产出响应分段

#### Scenario: 服务返回多段拼接文本
- **当** Qwen 服务返回包含多个 `language <语种><asr_text>` 片段的文本
- **则** 系统拼接各片段文本并取第一个可识别的语种

#### Scenario: Qwen 服务调用失败
- **当** Qwen 服务不可达、超时或返回错误状态
- **则** 系统按配置的失败策略重试，重试耗尽后返回 `{"msg": "Qwen 服务调用失败，请稍后再试", "code": 4012}`

### Requirement: 按规则透传语言参数
系统 MUST 保留 `/v2/seacraft_asr` 的 `language` 请求参数（默认 `auto`），但当取值为 `auto`、空串或未提交时 MUST NOT 向 Qwen 服务发送 `language` 字段；提交具体语种时 MUST 原样透传。响应的 `language` MUST 取模型实际识别出的语种，而非请求参数回显。

#### Scenario: 未提交语言参数
- **当** 客户端未提交 `language`，或提交 `auto`
- **则** 系统不向 Qwen 服务发送 `language` 字段，由模型自动识别语种，响应 `language` 为识别结果

#### Scenario: 提交具体语种
- **当** 客户端提交 `language=en`
- **则** 系统把 `en` 透传给 Qwen 服务

#### Scenario: Qwen 服务不支持该语种
- **当** 提交的语种不在 Qwen 服务支持的清单内
- **则** 系统返回 `{"msg": "不支持的语言", "code": 4009}`

### Requirement: 返回语种占比字典
系统 MUST 在 `/v2/seacraft_asr` 的响应顶层返回 `language_ratio` 字典，键为模型识别出的语种名，值为该语种文本字符数占总字符数的百分比（保留一位小数，各值之和为 100.0）；无有效文本时返回空字典。

#### Scenario: 单语种音频
- **当** 音频全部识别为中文
- **则** 响应 `language_ratio` 为 `{"Chinese": 100.0}`

#### Scenario: 中英混合音频
- **当** 音频中部分段落识别为英文、其余为中文
- **则** 响应 `language_ratio` 同时包含 `Chinese` 与 `English` 两个键，值为各自文本字符占比

#### Scenario: 无有效文本
- **当** 没有任何有效转写文本
- **则** 响应 `language_ratio` 为空字典

### Requirement: 限制同时处理的音频数量
系统 MUST 限制同时处理的音频文件数量（默认 8），并 MUST 在处理结束（含异常）时立即释放槽位、删除该请求产生的全部临时文件。默认配置下超出上限的请求 MUST 立即返回队列已满错误，不得静默排队。

#### Scenario: 请求数超过音频并发上限
- **当** 并发到达的音频请求数超过配置的上限
- **则** 超出的请求立即返回 `{"msg": "处理队列已满，请稍后再试", "code": 4013}`

#### Scenario: 排队等待超时
- **当** 配置了大于 0 的排队等待超时，且请求在等待中超过该时长仍未获得处理槽位
- **则** 系统返回 `{"msg": "请求排队超时，请稍后再试", "code": 4010}`

#### Scenario: 队列容量已满
- **当** 等待队列长度达到配置的上限且无法再接纳
- **则** 系统返回 `{"msg": "服务繁忙，请稍后再试", "code": 4013}`

#### Scenario: 音频处理结束释放槽位
- **当** 一个音频请求处理完成
- **则** 系统释放其占用槽位并让队列中的下一个请求开始处理

#### Scenario: 音频处理异常时释放槽位
- **当** 一个音频请求在处理过程中抛出异常
- **则** 系统同样释放其占用槽位并删除全部临时文件，不影响后续请求

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时发往 Qwen 服务的 VAD 段请求数量（`segment_concurrency`，默认 4），并 MUST 限制所有音频合计发往 Qwen 服务的请求数量（`request_concurrency`，默认 32），超出部分 MUST 等待可用名额后再发出。

#### Scenario: 单个音频产生大量分段
- **当** 一个音频切分出超过并发上限的 VAD 段
- **则** 系统按 `segment_concurrency` 分批向 Qwen 服务发起请求，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **当** 多个音频同时进入处理
- **则** 全部音频合计发往 Qwen 服务的并发请求数不超过 `request_concurrency`

### Requirement: 保留扩展字段
`/v2/seacraft_asr` 的每个响应分段 MUST 保留 `segment_words`、`role`、`emotion` 三个字段，其中 `segment_words` 为空数组（Qwen 服务不返回词级时间戳）；`role` 与 `emotion` 在对应能力未开启或推断失败时为 `null`。

#### Scenario: 返回扩展字段
- **当** 响应包含分段
- **则** 每个分段都包含 `segment_words`、`role`、`emotion` 三个键，其中 `segment_words` 为空数组

#### Scenario: 未开启说话人与情绪能力
- **当** 请求未携带 `showSpk` / `showEmotion`
- **则** 每个分段的 `role` 与 `emotion` 均为 `null`

### Requirement: 通过配置外置运行参数
Qwen 适配链路的参数 MUST 全部来自配置文件，服务地址、模型名、请求并发度、音频并发度、队列容量与超时、VAD 参数 MUST NOT 硬编码在业务代码中。

#### Scenario: 修改并发参数后重启生效
- **当** 运维人员修改配置中的 Qwen 请求并发度与音频并发度并重启服务
- **则** 服务按新的并发参数运行

#### Scenario: 配置缺省时使用默认值
- **当** 配置文件中缺少相关配置项
- **则** 服务使用默认值：单音频内同时发往 Qwen 的段请求数 4、全部音频合计请求数 32、同时处理的音频数 8、队列满立即拒绝

### Requirement: 配置 v2 链路模型的落卡
系统 MUST 支持通过配置指定 v2 链路三个本地模型（VAD、emotion2vec、speech_campplus）的 GPU 索引，三者绑定同一张卡，默认 GPU 0；`/v1.1.8` 链路 MUST NOT 受该配置影响。

#### Scenario: 修改落卡配置
- **当** 运维人员把三个模型的 GPU 索引改为 1 并重启服务
- **则** v2 链路的三个模型均加载到 GPU 1，v1.1.8 链路仍按原有 `device` / `ngpu` 配置运行

#### Scenario: 配置缺省
- **当** 配置文件中缺少落卡配置
- **则** 三个模型均加载到 GPU 0

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并 MUST 与 Qwen 转写并行执行：每个 VAD 段的转写任务与该段的 role/emotion 任务 MUST 在同一时刻发起，MUST NOT 等待全部转写完成后再计算 role/emotion。两个模型 MUST 仅在配置 `open_spk` 与 `open_emotion` 同时为 true 时加载；实际计算 MUST 由请求参数 `showSpk` / `showEmotion` 触发（默认均为 false）；emotion2vec MUST 以半精度加载。

#### Scenario: 开启角色与情绪能力
- **当** 请求参数 `showSpk` 与 `showEmotion` 均为 true
- **则** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行

#### Scenario: 角色与情绪能力未开启
- **当** 请求未携带 `showSpk` / `showEmotion`（默认 false）
- **则** 响应分段的 `role` 与 `emotion` 保持 `null`

#### Scenario: 配置未同时开启说话人与情绪能力
- **当** 配置中 `open_spk` 与 `open_emotion` 未同时为 true
- **则** 系统不加载 CAM++ 与 emotion2vec 模型

#### Scenario: 角色或情绪推断失败
- **当** 说话人或情绪模型推理失败
- **则** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **当** emotion2vec 模型被加载
- **则** 模型以半精度（fp16）加载，显存占用约 1 GB 级别，情绪精度不做额外调优

### Requirement: 验证效果与性能边界
系统 MUST 在 `test/audio_file/` 提供的音频上完成效果与性能验证，覆盖 1、2、4、8 路并发下的 GPU 算力占用、端到端时延、转写准确度、时间戳准度，并记录出现幻觉或重复的并发边界。

#### Scenario: 完成并发压测
- **当** 开发人员在目标服务器上按 1、2、4、8 路并发发起请求
- **则** 测试记录包含各并发档位的时延、GPU 算力占用与转写质量观测

#### Scenario: 产出测试报告
- **当** 验证完成
- **则** 仓库中留下一份可归档的测试报告，说明准确度、时间戳准度与幻觉/重复出现的边界

### Requirement: 验证角色与情绪的效果与性能
系统 MUST 在 `test/audio_file/` 的音频上验证 role 与 emotion 的效果与性能：与 `/v1.1.8` 链路的逐段结果比对一致率，并测量开启后的时延、吞吐与显存增量；开启 role/emotion MUST NOT 改变转写文本。

#### Scenario: 角色与情绪效果比对
- **当** 在同一音频、同一 VAD 参数下分别调用 `/v2/seacraft_asr`（`showSpk=true`、`showEmotion=true`）与 `/v1.1.8/seacraft_asr`
- **则** 逐段比较 role 的角色语义与 emotion 的业务标签，记录一致率与不一致样本

#### Scenario: 并行开销可接受
- **当** 比较「仅转写」与「说话人与情绪全开」两组在同一并发档位下的端到端时延
- **则** 单路并发下全开组的 P50 时延增幅不超过 30%；更高并发档位下如实记录增幅幅度并给出可选处置手段供决策

#### Scenario: 开启增强能力不破坏分段结构
- **当** 比较「仅转写」与「说话人与情绪全开」两组的分段结果
- **则** 两组的分段数量与每段起止时间一致，段内文本差异幅度不高于「仅转写」组自身在两次独立请求之间的波动幅度

#### Scenario: 显存与吞吐可记录
- **当** 在 1 / 4 / 8 音频并发下运行 role/emotion 全开
- **则** 记录本机 GPU 的 SM 利用率与显存占用增量、吞吐（音频/分钟、段/秒），并写入测试报告
