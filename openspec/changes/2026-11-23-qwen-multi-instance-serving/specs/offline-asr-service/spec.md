# Spec Delta

## MODIFIED Requirements

### Requirement: NPU 迁移保持 v3 接口序列化与计时契约
系统 MUST 保持v3请求字段及其默认值、既有响应字段/类型、错误码、分段排序和原VAD边界；bg/ed仍为两位小数字符串，load_audio_time_ms/gpu_time_ms仍为字符串，gpu_time_ms保留原名和主链路计时口径，不能替换成纯NPU时间。正式continuous仅按本能力新增段警告状态；owner、设备、排队和各模型遥测另列，不能把容量默认变更解释成公开接口重命名。

#### Scenario: NPU 转写请求返回结果
- **WHEN** 客户端按既有 v3 表单提交音频
- **THEN** 返回保留既有字段及类型，VAD 起止边界和分段排序不因后端改变而变化，`gpu_time_ms` 仍采用迁移前的主链路计时口径

#### Scenario: 未请求增强能力
- **WHEN** 请求未显式启用 `showSpk`、`showEmotion`、`showRoleIdentify`
- **THEN** 这三个开关保持默认 false，系统不因迁移而自动加载或调用未请求的增强能力

#### Scenario: 模型加载或推理失败
- **WHEN** NPU 权重加载、设备可用性、语种或推理执行发生失败
- **THEN** 系统按既有错误分层返回，不新增对外错误码，不把推理错误伪装为无人声 `4008`

#### Scenario: 多实例保持公开响应
- **WHEN** 服务按continuous/process N=6启动并由不同owner承载普通或增强请求
- **THEN** 每个响应仍保留现有顶层和segment_words/role/emotion等字段及类型，A开关默认false、未请求能力不被调用；不要求客户端提交owner、测试SHA或额外路由参数

### Requirement: 批执行保持独立段的业务归属
系统 MUST 保持一VAD段对应独立逻辑任务及至多一个有效响应分段；物理批不得拼音频、混语种或改变时间轴，mask/padding须验证有效样本和数值。正式continuous中同一request_id的所有段 MUST 留在其动态租用owner至请求安全收尾，段可独立完成/补位。取消/超时不等于设备完成，忙owner、KV及PCM不能提前或重复出租；显式legacy仍保持原独立段归属。

#### Scenario: 多个独立段组成一批
- **WHEN** 调度器将多个独立段放入同一执行批次
- **THEN** 每个结果可追溯到原段、原请求和语种参数，输出排序仍与原 VAD 顺序一致，无效段仍按既有规则过滤

#### Scenario: 批次执行期间请求取消
- **WHEN** 批内某请求取消但设备工作仍未完成
- **THEN** 系统保持在途资源所有权直至收尾完成，其他段的输出不被错误归属，实例和名额不提前或重复释放

#### Scenario: 同名上传结果隔离
- **WHEN** 多个同名或同内容文件通过独立request_id并发转写，并在某请求结束后复用其owner
- **THEN** 各响应保持自己的原段序号、边界和文本，取消及复用不读取另一请求KV或清理另一请求PCM，不以文件名或SHA决定固定owner

## ADDED Requirements

### Requirement: 正式持续转写返回段警告状态
正式continuous响应 MUST 添加processing_status（complete或completed_with_warnings）及segment_warnings数组；每条警告含原segment_index、bg、ed、reason、action、recovered，不重排原警告索引或暴露内部原始生成轨迹。复读、长度耗尽、无效前缀经保守恢复保留可信文本，无法恢复明确警告并继续其它段；MUST NOT 将警告当完整无异常成功。显式legacy响应字段保持原样。

#### Scenario: 所有段正常结束
- **WHEN** continuous请求所有有效段正常处理且没有内容生成警告
- **THEN** 返回processing_status=complete及空segment_warnings；既有语种、扩展字段和计时口径不变

#### Scenario: 部分异常段恢复或不可恢复
- **WHEN** 某段发生repetition_stop、length_limit或invalid_output_prefix而其它段正常
- **THEN** 可恢复部分沿原边界保留文本，不可恢复段即使不进入segments仍在segment_warnings记录原因和动作；整音频继续并返回completed_with_warnings

#### Scenario: 全部有效段均不可恢复
- **WHEN** 有效VAD段均发生内容生成异常并无法恢复可信文本
- **THEN** 返回空segments/text和完整原段警告及completed_with_warnings，不伪装4008无人声，也不计严格无警告成功

#### Scenario: 基础设施错误不降级成警告
- **WHEN** 发生Qwen不可用、设备OOM、worker死亡、取消、超时或未知执行异常
- **THEN** 沿既有4011/4012、4009等适用错误及取消契约传播，不转换成空文本成功或段内容警告，设备未终态前保持资源所有权

#### Scenario: 真实无有效人声
- **WHEN** VAD及既有过滤后没有有效段，或没有有效文本且没有内容生成警告
- **THEN** 继续返回既有4008规则，不因默认continuous变成无条件成功
