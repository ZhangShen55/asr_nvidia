# Spec Delta

## MODIFIED Requirements

### Requirement: 提供 Qwen 版转写接口
系统 MUST 提供 POST /v3/seacraft_asr，其 multipart 表单参数与 POST /v1.1.8/seacraft_asr 一致（音频文件、语言、词级时间戳、热词、说话人、情绪、语速、角色识别），且 /v1.1.8/seacraft_asr 的行为 MUST NOT 因本能力而改变。

#### Scenario: 提交音频到 v2 接口
- **WHEN** 客户端向 /v3/seacraft_asr 上传有效音频
- **THEN** 系统返回与 /v1.1.8/seacraft_asr 同构的成功响应，顶层包含 language、segments、text、speed_info、load_audio_time_ms、gpu_time_ms

#### Scenario: 未提交可选参数
- **WHEN** 客户端只提交音频文件
- **THEN** 系统按 language=auto、wordTimestamps=false、showSpk=false、showEmotion=false、showRoleIdentify=false 处理

#### Scenario: v1.1.8 行为保持不变
- **WHEN** 客户端向 /v1.1.8/seacraft_asr 提交请求
- **THEN** 系统仍按原有 Paraformer/Whisper 分流逻辑处理，响应结构与错误码不变

## ADDED Requirements

### Requirement: 移除 v2 Qwen 路由
服务 MUST NOT 再注册或文档化 POST /v2/seacraft_asr；客户端访问该路径 MUST 得到未找到路由的结果，v3 是本链路唯一公开版本。

#### Scenario: 访问已移除的 v2 路径
- **WHEN** 客户端向 /v2/seacraft_asr 发起请求
- **THEN** 服务不执行转写处理并返回未找到路由的结果

#### Scenario: v3 语义迁移
- **WHEN** 客户端把原 v2 请求路径替换为 /v3/seacraft_asr
- **THEN** VAD 切分、Qwen 调用、A/B 音频准入、role/emotion 错误码和响应字段继续遵循本能力其余要求
