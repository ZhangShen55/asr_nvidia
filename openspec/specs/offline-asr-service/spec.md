# offline-asr-service Specification

## Purpose

定义精简后的离线语音服务对外能力，确保常用 ASR 转写、状态查询和任务列表清理继续可用，同时让不再维护的接口和能力明确退出。

## Requirements

### Requirement: 保留最小业务接口集
系统 MUST 只暴露离线 ASR 转写、服务状态查询和任务列表清理三类业务接口，其中离线 ASR 转写接口只包含 POST /v3/seacraft_asr。

#### Scenario: 查询可用业务接口
- **WHEN** 客户端读取 OpenAPI 定义或调用业务接口
- **THEN** 系统提供 POST /v3/seacraft_asr、GET /get_status 和 DELETE /clear_tasks_list

#### Scenario: 访问已移除接口
- **WHEN** 客户端请求 POST /v2/seacraft_asr、/v1.1.7/seacraft_asr、/v1.1.8/seacraft_asr、/audio/db_snr、/audio/detect_mandarin 或 /text/question
- **THEN** 系统不得执行对应业务处理，并返回未找到该路由的结果

### Requirement: 移除低频能力依赖
服务启动和 /v3/seacraft_asr 运行 MUST NOT 依赖 Whisper、Pyannote、BERT、SeACo-Paraformer 或 PUNC 模型。系统 MUST NOT 因缺少这些已下线模型的目录而无法启动或拒绝请求。

#### Scenario: 低频模型文件缺失
- **WHEN** 镜像中不存在 Whisper、Pyannote、BERT、SeACo-Paraformer 或 PUNC 模型文件
- **THEN** 服务仍能正常启动并完成 /v3/seacraft_asr 转写请求

#### Scenario: 未配置低频功能开关
- **WHEN** 部署配置中不存在多语言、多语言说话人分离或文本分类相关开关
- **THEN** 服务仍能正常加载保留功能所需的模型与配置

### Requirement: 语速统计覆盖非 ASCII 文本
语速统计 MUST 对 Unicode 文本按实际内容单元计数，使包含重音字母等非 ASCII 字符的文本在转写结果中得到正确的语速统计。

#### Scenario: 统计含非 ASCII 字符的语速
- **WHEN** 转写结果包含带重音符号或其它非 ASCII 字符的文本
- **THEN** 分段语速与整段语速统计按实际词数计算，而非按零计数

#### Scenario: 统计法语语速
- **WHEN** 转写结果包含带重音符号的西语、德语等非 ASCII 文本
- **THEN** 分段语速与整段语速统计按实际词数计算，而非按零计数

### Requirement: NPU 迁移保持 v3 接口序列化与计时契约
系统 MUST 保持 `/v3/seacraft_asr` 的请求字段、默认行为、响应字段、数据类型、错误码和分段排序。`bg`、`ed` MUST 继续返回两位小数字符串；`load_audio_time_ms`、`gpu_time_ms` MUST 继续返回字符串。`gpu_time_ms` MUST 保留原字段名、主链路计时起止和统计口径，MUST NOT 因设备迁移而改名为 `npu_time_ms` 或替换为纯设备计算耗时。内部 NPU、排队、编译和推理遥测 MUST 单独定义，不改变既有返回契约。

#### Scenario: NPU 转写请求返回结果
- **WHEN** 客户端按既有 v3 表单提交音频
- **THEN** 返回保留既有字段及类型，VAD 起止边界和分段排序不因后端改变而变化，`gpu_time_ms` 仍采用迁移前的主链路计时口径

#### Scenario: 未请求增强能力
- **WHEN** 请求未显式启用 `showSpk`、`showEmotion`、`showRoleIdentify`
- **THEN** 这三个开关保持默认 false，系统不因迁移而自动加载或调用未请求的增强能力

#### Scenario: 模型加载或推理失败
- **WHEN** NPU 权重加载、设备可用性、语种或推理执行发生失败
- **THEN** 系统按既有错误分层返回，不新增对外错误码，不把推理错误伪装为无人声 `4008`

### Requirement: 批执行保持独立段的业务归属
若启用批执行，系统 MUST 保持一个 VAD 段对应一个独立逻辑转写任务及至多一个有效响应分段；批次仅作为物理执行方式，MUST NOT 拼接段音频、跨段混用语言参数或改变时间轴。允许必要的 mask/padding 仅在有效样本、长度及数值语义经过验证时使用。取消、超时与失败 MUST 保持段和实例资源的正确归属，设备工作结束前 MUST NOT 将忙实例出租给新任务。

#### Scenario: 多个独立段组成一批
- **WHEN** 调度器将多个独立段放入同一执行批次
- **THEN** 每个结果可追溯到原段、原请求和语种参数，输出排序仍与原 VAD 顺序一致，无效段仍按既有规则过滤

#### Scenario: 批次执行期间请求取消
- **WHEN** 批内某请求取消但设备工作仍未完成
- **THEN** 系统保持在途资源所有权直至收尾完成，其他段的输出不被错误归属，实例和名额不提前或重复释放
