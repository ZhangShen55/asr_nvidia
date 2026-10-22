# Spec Delta

## MODIFIED Requirements

### Requirement: 保留最小业务接口集
系统 MUST 只暴露离线 ASR 转写、服务状态查询和任务列表清理三类业务接口，其中离线 ASR 转写接口包含 `POST /v1.1.8/seacraft_asr` 与 `POST /v2/seacraft_asr`。

#### Scenario: 查询可用业务接口
- **当** 客户端读取 OpenAPI 定义或调用业务接口
- **则** 系统提供 `POST /v1.1.8/seacraft_asr`、`POST /v2/seacraft_asr`、`GET /get_status` 和 `DELETE /clear_tasks_list`

#### Scenario: 访问已移除接口
- **当** 客户端请求 `/v1.1.7/seacraft_asr`、`/audio/db_snr`、`/audio/detect_mandarin` 或 `/text/question`
- **则** 系统不得执行对应业务处理，并返回未找到该路由的结果
