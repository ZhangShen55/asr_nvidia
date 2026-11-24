# Spec Delta

## MODIFIED Requirements

### Requirement: 正式持续转写返回段警告状态
正式continuous响应 MUST 添加processing_status（complete或completed_with_warnings）及segment_warnings数组；每条警告含原segment_index、bg、ed、reason、action、recovered，不重排原警告索引或暴露内部原始生成轨迹。已确认异常复读折叠成功时保留短句和一次循环，action=repetition_collapsed、recovered=true，仍返回completed_with_warnings；其它生成异常按既有恢复及不可恢复警告继续其它段，MUST NOT 将警告当完整无异常成功。显式legacy响应字段保持原样。

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

#### Scenario: 折叠短句仍显式警告
- **WHEN** 一个短VAD段的异常复读被折叠成一次有效内容，其余段正常
- **THEN** 短句进入segments并保留原bg/ed，其原segment_index警告标记repetition_collapsed和recovered=true，整音频为completed_with_warnings，内部原始轨迹不出现在响应中
