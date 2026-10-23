# 规格增量

## MODIFIED Requirements

### Requirement: 返回语种占比字典
系统 MUST 在 `/v2/seacraft_asr` 的响应顶层返回 `language_ratio` 字典，键为模型识别出的语种的**简写标识**（例如中文 `zh`、英文 `en`、粤语 `yue`），值为该语种文本字符数占总字符数的百分比（保留一位小数，各值之和为 100.0）；无有效文本时返回空字典。语种判定来源与判定逻辑 MUST NOT 改变，仅对键名做简写映射；无法映射的语种 MUST 原样透传其原始名称。

#### Scenario: 单语种音频
- **当** 音频全部识别为中文
- **则** 响应 `language_ratio` 为 `{"zh": 100.0}`

#### Scenario: 中英混合音频
- **当** 音频中部分段落识别为英文、其余为中文
- **则** 响应 `language_ratio` 同时包含 `zh` 与 `en` 两个键，值为各自文本字符占比

#### Scenario: 无法映射的语种
- **当** 模型识别出简写表中不存在的语种
- **则** 响应 `language_ratio` 中该语种以模型返回的原始名称作为键

#### Scenario: 无有效文本
- **当** 没有任何有效转写文本
- **则** 响应 `language_ratio` 为空字典

### Requirement: 异步并行计算真实角色与情绪
系统 MUST 支持使用 `speech_campplus_sv_zh_en_16k-common_advanced` 推断说话人角色、使用 `emotion2vec_plus_large` 推断情绪，并 MUST 与 Qwen 转写并行执行：每个 VAD 段的转写任务与该段的 role/emotion 任务 MUST 在同一时刻发起，MUST NOT 等待全部转写完成后再计算 role/emotion。两个模型 MUST 分别由独立配置开关控制（`open_v2_role` 与 `open_v2_emotion`，默认均为 true）；实际计算 MUST 由请求参数 `showSpk` / `showEmotion` 触发（默认均为 false）；emotion2vec MUST 以半精度加载。

#### Scenario: 开启角色与情绪能力
- **当** 请求参数 `showSpk` 与 `showEmotion` 均为 true
- **则** 响应分段的 `role` 与 `emotion` 为真实推断值，且与转写同时并发执行

#### Scenario: 角色与情绪能力未开启
- **当** 请求未携带 `showSpk` / `showEmotion`（默认 false）
- **则** 响应分段的 `role` 与 `emotion` 保持 `null`

#### Scenario: 配置未开启说话人能力
- **当** 配置中 `open_v2_role` 为 false
- **则** 系统不加载 CAM++ 模型，响应分段的 `role` 保持 `null`

#### Scenario: 配置未开启情绪能力
- **当** 配置中 `open_v2_emotion` 为 false
- **则** 系统不加载 emotion2vec 模型，响应分段的 `emotion` 保持 `null`

#### Scenario: 配置未同时开启说话人与情绪能力
- **当** 配置中 `open_v2_role` 与 `open_v2_emotion` 中任一为 false
- **则** 系统只跳过该开关对应的模型加载，另一个模型仍可按请求参数独立启用

#### Scenario: 角色或情绪推断失败
- **当** 说话人或情绪模型推理失败
- **则** 该分段的 `role` 或 `emotion` 回退为 `null`，转写结果仍正常返回

#### Scenario: 情绪模型精度
- **当** emotion2vec 模型被加载
- **则** 模型以半精度（fp16）加载，显存占用约 1 GB 级别，情绪精度不做额外调优

### Requirement: 配置 v2 链路模型的落卡
系统 MUST 支持通过配置指定 v2 链路三个本地模型（VAD、emotion2vec、speech_campplus）的 GPU 索引，三者绑定同一张卡，默认 GPU 0。

#### Scenario: 修改落卡配置
- **当** 运维人员把三个模型的 GPU 索引改为 1 并重启服务
- **则** v2 链路的三个模型均加载到 GPU 1

#### Scenario: 配置缺省
- **当** 配置文件中缺少落卡配置
- **则** 三个模型均加载到 GPU 0

### Requirement: 限制对 Qwen 服务的并发请求数
系统 MUST 限制单个音频内同时发往 Qwen 服务的 VAD 段请求数量（`segment_concurrency`，默认 4），并 MUST 限制所有音频合计发往 Qwen 服务的请求数量（`request_concurrency`，默认 32），超出部分 MUST 等待可用名额后再发出。两个默认值 MUST 与 Qwen 服务的实际承载能力匹配：当 Qwen 服务按多前端进程扩容后，本服务侧默认值 MUST NOT 主动放大，扩容收益由 Qwen 服务侧承担。

#### Scenario: 单个音频产生大量分段
- **当** 一个音频切分出超过并发上限的 VAD 段
- **则** 系统按 `segment_concurrency` 分批向 Qwen 服务发起请求，单个音频不超过该上限

#### Scenario: 多个音频同时处理
- **当** 多个音频同时进入处理
- **则** 全部音频合计发往 Qwen 服务的并发请求数不超过 `request_concurrency`

## ADDED Requirements

### Requirement: 规范化输出分段文本的标点
系统 MUST 在文本规范化之后对 `/v2/seacraft_asr` 的分段文本做标点卫生处理：当一个分段的文本去除空白后只剩标点符号时，该分段 MUST 从响应中移除；当一个分段的文本首个字符是标点符号时，MUST 移除该开头标点。该规则 MUST 独立于文本规范化开关生效，即规范化关闭时同样执行。

#### Scenario: 分段只有标点
- **当** 某个分段的文本只有标点符号（如 `。`、`！`）
- **则** 该分段不出现在响应的 `segments` 数组中

#### Scenario: 分段以标点开头
- **当** 某个分段的文本以标点符号开头（如 `，就。`）
- **则** 响应中该分段的文本不再以标点开头

#### Scenario: 规范化关闭时规则仍生效
- **当** 文本规范化被关闭
- **则** 上述两条标点处理规则仍然作用于响应分段

### Requirement: 提供文本规范化开关
系统 MUST 支持通过配置控制文本规范化的启用与行为：启用开关默认开启（`enabled` 为 true），并支持独立控制独立数字转换（`enable_standalone_number`）、数字逐位读出（`enable_0_to_9`）与语气词剔除（`remove_interjections`）。语气词剔除开关 MUST 可在配置中显式设置，默认行为 MUST 为剔除「呃」「啊」等语气词；剔除产生的标点副作用 MUST 由标点卫生规则清理。

#### Scenario: 默认开启文本规范化
- **当** 部署配置未显式覆盖文本规范化开关
- **则** 系统对转写文本执行规范化，并按配置剔除语气词

#### Scenario: 显式保留语气词
- **当** 部署配置把语气词剔除开关设为 false
- **则** 系统保留转写文本中的语气词

#### Scenario: 关闭文本规范化
- **当** 部署配置关闭文本规范化
- **则** 系统直接返回模型原始转写文本，不做数字与标点规范化

#### Scenario: 规范化依赖不可用
- **当** 文本规范化依赖缺失或初始化失败且失败策略为 `passthrough`
- **则** 请求仍成功返回，文本保持 ASR 原始输出
