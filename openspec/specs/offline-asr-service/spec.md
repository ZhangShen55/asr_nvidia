# offline-asr-service Specification

## Purpose

定义精简后的离线语音服务对外能力，确保常用 ASR 转写、状态查询和任务列表清理继续可用，同时让不再维护的接口和能力明确退出。

## Requirements

### Requirement: 保留最小业务接口集
系统 MUST 只暴露离线 ASR 转写、服务状态查询和任务列表清理三项业务接口。

#### Scenario: 查询可用业务接口
- **当** 客户端读取 OpenAPI 定义或调用业务接口
- **则** 系统提供 `POST /v1.1.8/seacraft_asr`、`GET /get_status` 和 `DELETE /clear_tasks_list`

#### Scenario: 访问已移除接口
- **当** 客户端请求 `/v1.1.7/seacraft_asr`、`/audio/db_snr`、`/audio/detect_mandarin` 或 `/text/question`
- **则** 系统不得执行对应业务处理，并返回未找到该路由的结果

### Requirement: 保持离线 ASR 请求契约
`POST /v1.1.8/seacraft_asr` MUST 继续接受现有 multipart 表单参数，包括音频文件、语言、词级时间戳、热词、说话人、情绪、语速和角色识别选项。

#### Scenario: 提交标准转写请求
- **当** 客户端上传有效音频并提交现有可选参数
- **则** 系统接受请求，并按当前业务规则返回转写结果

### Requirement: 保持离线 ASR 响应契约
成功的 `POST /v1.1.8/seacraft_asr` 响应 MUST 继续包含语言、分段文本、整段文本、语速分时段统计、音频加载耗时和 GPU 处理耗时等既有字段。

#### Scenario: 返回转写结果
- **当** 音频中检测到有效人声并完成转写
- **则** 响应包含非空分段及与请求选项一致的说话人、角色和情绪字段

### Requirement: 保留现有离线 ASR 能力
`POST /v1.1.8/seacraft_asr` MUST 继续支持中文转写、VAD、标点恢复、可选词级时间戳、说话人区分、教师/学生角色推断、情绪识别、语速统计和长音频分块处理。

#### Scenario: 处理启用说话人识别的长音频
- **当** 请求启用说话人分离且音频超过配置的长音频阈值
- **则** 系统按现有分块、重叠裁剪和重试规则完成转写，并返回合并且时间偏移正确的分段

### Requirement: 移除低频能力依赖
服务启动和 `/v1.1.8/seacraft_asr` 运行 MUST NOT 依赖 Whisper、Pyannote 或 BERT 模型。

#### Scenario: 低频模型文件缺失
- **当** 镜像中没有 Whisper、Pyannote 或 BERT 模型文件
- **则** 服务仍能正常启动并完成 `/v1.1.8/seacraft_asr` 转写请求

#### Scenario: 未配置低频功能开关
- **当** 部署配置中不存在多语言、多语言说话人分离或文本分类相关开关
- **则** 服务仍能正常加载保留功能所需的模型与配置
