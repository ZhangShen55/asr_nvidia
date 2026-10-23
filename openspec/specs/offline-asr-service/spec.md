# offline-asr-service Specification

## Purpose

定义精简后的离线语音服务对外能力，确保常用 ASR 转写、状态查询和任务列表清理继续可用，同时让不再维护的接口和能力明确退出。

## Requirements

### Requirement: 保留最小业务接口集
系统 MUST 只暴露离线 ASR 转写、服务状态查询和任务列表清理三类业务接口，其中离线 ASR 转写接口只包含 `POST /v2/seacraft_asr`。

#### Scenario: 查询可用业务接口
- **当** 客户端读取 OpenAPI 定义或调用业务接口
- **则** 系统提供 `POST /v2/seacraft_asr`、`GET /get_status` 和 `DELETE /clear_tasks_list`

#### Scenario: 访问已移除接口
- **当** 客户端请求 `/v1.1.7/seacraft_asr`、`/v1.1.8/seacraft_asr`、`/audio/db_snr`、`/audio/detect_mandarin` 或 `/text/question`
- **则** 系统不得执行对应业务处理，并返回未找到该路由的结果

### Requirement: 移除低频能力依赖
服务启动和 `/v2/seacraft_asr` 运行 MUST NOT 依赖 Whisper、Pyannote、BERT、SeACo-Paraformer 或 PUNC 模型。系统 MUST NOT 因缺少这些已下线模型的目录而无法启动或拒绝请求。

#### Scenario: 低频模型文件缺失
- **当** 镜像中不存在 Whisper、Pyannote、BERT、SeACo-Paraformer 或 PUNC 模型文件
- **则** 服务仍能正常启动并完成 `/v2/seacraft_asr` 转写请求

#### Scenario: 未配置低频功能开关
- **当** 部署配置中不存在多语言、多语言说话人分离或文本分类相关开关
- **则** 服务仍能正常加载保留功能所需的模型与配置

### Requirement: 语速统计覆盖非 ASCII 文本
语速统计 MUST 对 Unicode 文本按实际内容单元计数，使包含重音字母等非 ASCII 字符的文本在转写结果中得到正确的语速统计。

#### Scenario: 统计含非 ASCII 字符的语速
- **当** 转写结果包含带重音符号或其它非 ASCII 字符的文本
- **则** 分段语速与整段语速统计按实际词数计算，而非按零计数

#### Scenario: 统计法语语速
- **当** 转写结果包含带重音符号的西语、德语等非 ASCII 文本
- **则** 分段语速与整段语速统计按实际词数计算，而非按零计数
