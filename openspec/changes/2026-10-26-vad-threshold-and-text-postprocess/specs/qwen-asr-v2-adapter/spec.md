# Spec Delta

## MODIFIED Requirements

### Requirement: 配置 v2 链路模型的落卡
系统 MUST 支持通过配置指定 v2 链路本地模型的运行设备：CAM++（`speech_campplus`）与 emotion2vec 的设备值 MUST 使用规范设备串 `cuda:N`，并由 `[model_device]` 配置；默认值 MUST 为 `cuda:0`。VAD 的运行设备 MUST 由独立的 `[qwen_asr].vad_device` 指定，默认为 `"cpu"`，MUST NOT 随 `[model_device]` 的设备配置迁移。

#### Scenario: 修改落卡配置
- **WHEN** 运维人员把 `[model_device].emotion2vec` 与 `[model_device].speech_campplus` 改为 `"cuda:1"` 并重启服务
- **THEN** CAM++ 与 emotion2vec 均加载到 `cuda:1`

#### Scenario: 配置缺省
- **WHEN** 配置文件缺少 `[model_device]` 中的本地模型设备
- **THEN** CAM++ 与 emotion2vec 使用 `cuda:0`，VAD 使用 CPU

#### Scenario: 设备串非法
- **WHEN** 本地模型设备不是 `cuda:N` 格式、CUDA 不可用或 N 超出可见设备范围
- **THEN** 服务拒绝加载对应模型并返回可读的配置或设备错误，不得静默回退到另一张卡

#### Scenario: VAD 单独切换设备
- **WHEN** 运维人员把 `[qwen_asr].vad_device` 设为 `"cuda:1"`
- **THEN** VAD 使用 `cuda:1`，CAM++ 与 emotion2vec 仍使用 `[model_device]` 中各自的设备

### Requirement: 按规则透传语言参数
系统 MUST 保留 `/v2/seacraft_asr` 的 `language` 请求参数（默认 `auto`），当取值为 `auto`、空串或未提交时 MUST NOT 向 Qwen 服务发送 `language` 字段。响应的 `language` MUST 基于删除无效分段后的语言占比确定：单一语种取该语种，混合语种取占比最高者；占比相同取最早出现的有效语种。

#### Scenario: 未提交语言参数
- **WHEN** 客户端未提交 `language` 或提交 `auto`
- **THEN** 请求不向 Qwen 发送 `language` 字段，响应语言按有效分段的语言占比确定

#### Scenario: 提交具体语种
- **WHEN** 客户端提交 `language=en`
- **THEN** 系统把 `en` 原样透传给 Qwen，响应仍使用模型返回的有效语种结果

#### Scenario: 中英文混合且英文占比最高
- **WHEN** 删除无效分段后英文文本占比高于其他语种
- **THEN** 响应顶层 `language` 为英文语种，英文后处理规则可以生效

#### Scenario: Qwen 服务不支持该语种
- **WHEN** 提交的语种不在 Qwen 服务支持清单内
- **THEN** 系统返回 `{"msg": "不支持的语言", "code": 4009}`

## ADDED Requirements

### Requirement: 配置 VAD 语音噪声判定阈值
系统 MUST 支持在 `[qwen_asr]` 中配置 VAD 语音/噪声判定阈值，默认值 MUST 为 `0.6`，并在 VAD worker 初始化时生效。数值越小表示检测越灵敏，数值越大表示更容易过滤小声语音。

#### Scenario: 使用默认阈值
- **WHEN** 配置中未提供 VAD 语音噪声判定阈值
- **THEN** 系统使用 `0.6`，且行为保持 FunASR 默认阈值附近

#### Scenario: 修改阈值后重启生效
- **WHEN** 运维人员修改阈值并重启服务
- **THEN** 所有新建 VAD worker 使用该阈值，VAD 输出随阈值变化，不需要修改代码

#### Scenario: 阈值方向可解释
- **WHEN** 使用较低阈值与较高阈值处理同一份音频
- **THEN** 较低阈值允许更多低能量语音进入候选，较高阈值过滤更多低能量语音；实现和文档 MUST 明确该方向

### Requirement: 删除单句单字语气词分段
系统 MUST 在生成最终响应前删除整个 `segment_text` 去除外围空白和标点后仅为一个指定中文语气词的分段，默认集合包含「啊」「哦」「嗯」「呃」。被删除分段的时间、速度、词、角色和情绪字段 MUST 一并不出现在响应中。

#### Scenario: 删除纯单字语气词
- **WHEN** 分段文本为 `"嗯。"`、`"啊！"`、`"哦"` 或 `"呃"`
- **THEN** 整个分段从 `segments` 中移除

#### Scenario: 保留包含有效内容的分段
- **WHEN** 分段文本为 `"嗯，我认为"`、`"可以啊，可以"` 或其他包含有效词语的文本
- **THEN** 分段保留，不能因为包含语气词而整体删除

#### Scenario: 删除后没有有效文本
- **WHEN** 音频的所有有效候选分段都属于单字语气词分段
- **THEN** 接口返回现有的 `{"msg": "音频文件为空或未检测到任何人声", "code": 4008}`

### Requirement: 英文主导结果替换中文口气词
当最终语言为英文时，系统 MUST 把由分段边界、空白或标点界定的中文口气词替换为固定英文形式：`啊`→`ah`、`哦`→`oh`、`嗯`→`um`、`呃`→`uh`。纯单字语气词分段优先按分段移除规则处理，不得先替换后保留。

#### Scenario: 替换英文主导分段中的口气词
- **WHEN** 有效分段的最终语言为英文，文本为 `"I，嗯，think so"`
- **THEN** 分段文本输出为 `"I，um，think so"` 或等价的标点规范化形式

#### Scenario: 不替换普通中文内容中的同字
- **WHEN** 最终语言不是英文，或语气词字符不是由边界、空白或标点界定
- **THEN** 系统保留原字符，不执行英文口气词替换

#### Scenario: 混合语言英文占比不足
- **WHEN** 英文不是删除无效分段后占比最高的语种
- **THEN** 系统不执行英文口气词替换

### Requirement: 后处理后响应统计保持一致
系统 MUST 在单字语气词删除、英文口气词替换、ITN 和标点清理完成后，使用最终保留的分段重新生成顶层 `text`、`language_ratio` 和 `speed_info`；这些字段 MUST NOT 引用已删除分段。

#### Scenario: 删除分段后重算统计
- **WHEN** 一次响应删除一个或多个单字语气词分段
- **THEN** `text` 不包含被删除文本，`language_ratio` 不统计被删除文本，`speed_info` 不包含被删除分段的速度

#### Scenario: 替换文本后保持分段字段
- **WHEN** 一个分段发生英文口气词替换
- **THEN** 该分段的 `bg`、`ed`、`segment_words`、`role`、`emotion` 字段仍保持原有契约，文本和基于文本的速度值与最终文本一致

#### Scenario: 标点清理与后处理组合
- **WHEN** ITN 将语气词转换为空或标点，随后标点卫生规则删除该分段
- **THEN** 顶层统计与最终 `segments` 数组一致，不出现只存在于统计字段中的幽灵分段

### Requirement: 提供细小声音参考音频测试出口
系统 MUST 提供针对 `test/audio_file/理论计算机科学导论_时间am9.55_191803.wav` 的可选集成回归测试，并使用对应人工标注结果识别“细小声音”分段。测试 MUST 记录不同 VAD 阈值下的误检、漏检、分段数量和发送给 Qwen 的请求数量。

#### Scenario: 解析人工标注结果
- **WHEN** 测试读取 `test/test_result/理论计算机科学导论_时间am9.55_191803.json`
- **THEN** 测试能处理其中的裸 `细小声音` 标记及说明文字，并得到 46 个标注分段，不修改原始文件

#### Scenario: 阈值对比
- **WHEN** 使用至少三个阈值档位处理同一参考音频
- **THEN** 测试报告每档的标注段命中率、非标注段误检率、过滤后段数、Qwen 请求数和端到端耗时

#### Scenario: 低能量样本回归
- **WHEN** VAD 阈值调整后运行参考音频测试
- **THEN** 标注段的抑制效果相对默认阈值可量化比较，正常有效语音的漏检变化单独记录，不以“段数减少”作为唯一通过标准

