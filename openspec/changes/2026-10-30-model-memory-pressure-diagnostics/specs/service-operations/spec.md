# Spec Delta

## ADDED Requirements

### Requirement: 暴露资源诊断状态
`GET /get_status` MUST 在不改变既有字段语义的前提下提供可用于资源诊断的主进程、VAD worker、模型池和请求资源快照；字段缺失或采集失败 MUST 明确标记，不得伪造为零值。

#### Scenario: 查询资源状态
- **WHEN** 运维调用 `GET /get_status`
- **THEN** 响应可区分模型池实例、VAD worker、busy/waiting、主机内存/Swap 和 GPU 资源状态，并保留现有服务状态字段

#### Scenario: 资源指标不可用
- **WHEN** 当前环境无法读取某项主机或 GPU 指标
- **THEN** 该指标返回明确的不可用状态或原因，其他状态字段仍正常返回

### Requirement: 支持安全的测试收尾
服务运维工具 MUST 支持停止新请求、等待在途请求回收并优雅停止 SeaCraftASR；测试收尾 MUST 不自动停止或重启独立运行的 Qwen-ASR 服务。

#### Scenario: 收到测试终止信号
- **WHEN** 资源安全闸门触发或运维结束测试
- **THEN** 工具停止提交新请求，等待可回收任务并记录后端、VAD worker、缓存和端口最终状态

#### Scenario: Qwen 服务保持运行
- **WHEN** SeaCraftASR 测试收尾完成
- **THEN** Qwen-ASR 的 8806 端口和服务进程保持运行，诊断报告记录其健康检查结果
