# offline-asr-service Specification

## Purpose

定义精简后的离线语音服务对外能力，确保常用 ASR 转写、状态查询和任务列表清理继续可用，同时让不再维护的接口和能力明确退出。

## Requirements

### Requirement: 保留最小业务接口集
系统 MUST 只暴露离线 ASR 转写、服务状态查询和任务列表清理三类业务接口，其中离线 ASR 转写接口包含 `POST /v1.1.8/seacraft_asr` 与 `POST /v2/seacraft_asr`。

#### Scenario: 查询可用业务接口
- **当** 客户端读取 OpenAPI 定义或调用业务接口
- **则** 系统提供 `POST /v1.1.8/seacraft_asr`、`POST /v2/seacraft_asr`、`GET /get_status` 和 `DELETE /clear_tasks_list`

#### Scenario: 访问已移除接口
- **当** 客户端请求 `/v1.1.7/seacraft_asr`、`/audio/db_snr`、`/audio/detect_mandarin` 或 `/text/question`
- **则** 系统不得执行对应业务处理，并返回未找到该路由的结果

### Requirement: 保持离线 ASR 请求契约
`POST /v1.1.8/seacraft_asr` MUST 继续接受现有 multipart 表单参数，包括音频文件、语言、词级时间戳、热词、说话人、情绪、语速和角色识别选项。未显式提交时，`language` MUST 默认为 `auto`，`wordTimestamps` 默认为 `false`，`showSpk` 与 `showEmotion` 在 12 服务器默认为 `false`，`showRoleIdentify` 默认为 `false`。

#### Scenario: 提交标准转写请求
- **当** 客户端上传有效音频并提交现有可选参数
- **则** 系统接受请求，并按当前业务规则返回转写结果

#### Scenario: 不提交可选参数
- **当** 客户端只提交音频文件
- **则** 系统按 `language=auto`、`showSpk=false`、`showEmotion=false`、`showRoleIdentify=false` 处理请求

### Requirement: 保持离线 ASR 响应契约
成功的 `POST /v1.1.8/seacraft_asr` 响应 MUST 继续包含语言、分段文本、整段文本、语速分时段统计、音频加载耗时和 GPU 处理耗时等既有字段。整段文本 MUST 由分段文本按空格拼接并经过文本规范化处理，与 Mac 侧同源实现保持一致。

#### Scenario: 返回转写结果
- **当** 音频中检测到有效人声并完成转写
- **则** 响应包含非空分段及与请求选项一致的说话人、角色和情绪字段

#### Scenario: 规范化依赖不可用
- **当** 文本规范化依赖缺失或初始化失败且失败策略为 `passthrough`
- **则** 请求仍成功返回，文本保持 ASR 原始输出

### Requirement: 保留现有离线 ASR 能力
`POST /v1.1.8/seacraft_asr` MUST 继续支持中文转写、VAD、标点恢复、可选词级时间戳、说话人区分、教师/学生角色推断、情绪识别、语速统计和长音频分块处理。

#### Scenario: 处理启用说话人识别的长音频
- **当** 请求启用说话人分离且音频超过配置的长音频阈值
- **则** 系统按现有分块、重叠裁剪和重试规则完成转写，并返回合并且时间偏移正确的分段

### Requirement: 移除低频能力依赖
服务启动和 `/v1.1.8/seacraft_asr` 运行 MUST NOT 依赖 Pyannote 或 BERT 模型。Whisper 模型 MUST 仅在部署配置开启多语言能力时按需加载，未开启或模型缺失 MUST NOT 影响中英文转写与其他功能开关的加载。

#### Scenario: 低频模型文件缺失
- **当** 镜像中没有 Whisper、Pyannote 或 BERT 模型文件且未开启多语言能力
- **则** 服务仍能正常启动并完成 `/v1.1.8/seacraft_asr` 中英文转写请求

#### Scenario: 未配置低频功能开关
- **当** 部署配置中不存在多语言、多语言说话人分离或文本分类相关开关
- **则** 服务仍能正常加载保留功能所需的模型与配置

### Requirement: 按语言分流转写引擎
系统 MUST 在读取音频前对 `language` 做去空格与小写归一化，并将 `auto`、`zh`、`en` 分流到 Paraformer，将 `fr` 分流到 Whisper，其余语言一律拒绝。

#### Scenario: 中英文请求
- **当** 客户端提交 `auto`、`zh` 或 `en`（含 `" ZH "`、`"EN"` 等大小写与空格变体）
- **则** 请求由 Paraformer 处理，响应中的 `language` 为归一化后的取值

#### Scenario: 法语请求
- **当** 客户端提交 `fr`（含 `" FR "` 等变体）
- **则** 请求由 Whisper 处理，响应中的 `language` 为 `fr`

#### Scenario: 不支持的语言
- **当** 客户端提交 `auto`、`zh`、`en`、`fr` 以外的语言，或显式提交空的 `language`
- **则** 系统返回 `{"msg": "不支持的语言: <归一化后的语言>", "code": 4009}`，且不读取音频

### Requirement: 保留显式空语言的表单语义
系统 MUST 区分“未提交 `language`”与“显式提交空 `language`”两种情形，前者使用默认 `auto`，后者按不支持的语言处理。

#### Scenario: 未提交语言参数
- **当** 客户端提交的表单中不含 `language` 字段
- **则** 系统按 `language=auto` 处理请求

#### Scenario: 显式提交空语言参数
- **当** 客户端提交的表单中 `language` 为空字符串
- **则** 系统返回不支持语言的错误结果，且不进入音频预处理

### Requirement: 法语转写使用 Whisper
多语言能力开启且 Whisper 模型就绪时，法语请求 MUST 使用 Whisper 完成转写，响应结构 MUST 与 Paraformer 成功响应保持一致的顶层字段集合。

#### Scenario: 法语转写成功
- **当** 客户端提交法语音频且多语言能力开启、Whisper 模型加载成功
- **则** 响应包含 `language`、`segments`、`text`、`speed_info`、`load_audio_time_ms`、`gpu_time_ms` 字段，且不包含错误码字段

#### Scenario: 请求角色与情绪但法语链路无增强模型
- **当** 客户端在法语请求中提交 `showSpk`、`showRoleIdentify` 或 `showEmotion`
- **则** 对应分段包含取值为 `null` 的 `role` 或 `emotion` 字段，不返回错误

#### Scenario: 多语言能力未开启
- **当** 部署配置未开启多语言能力或 Whisper 模型未就绪，且客户端提交法语请求
- **则** 系统返回 `{"msg": "未开启小语种识别或模型未就绪", "code": 4003}`，且不读取音频

### Requirement: 语速统计覆盖非 ASCII 文本
语速统计 MUST 对 Unicode 文本按实际内容单元计数，使包含重音字母等非 ASCII 字符的文本在两侧实现中得到相同的语速结果。

#### Scenario: 统计法语语速
- **当** 转写结果包含带重音符号的法语词汇
- **则** 分段语速与整段语速统计按实际词数计算，而非按零计数
