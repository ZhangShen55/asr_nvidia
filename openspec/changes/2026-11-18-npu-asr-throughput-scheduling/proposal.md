# Proposal

## Why

当前单份 Qwen 逐段串行处理完整课堂音频，Qwen 排队占据端到端耗时的主要部分；增加信号量没有形成真实批执行，慢速增强分支还可能持有转写名额。需要在华为 910B 物理卡7上，以单 Qwen 批处理优先、阶段独立调度和及时诊断，缩短完整音频处理时间。

## What Changes

- 优先实现单 Qwen 的真实多段批推理：单音频在途段上限16、全局128，批大小、等待窗口和输入预算配置化；批内每段保留独立音频、语种、时间轴和结果归属。128是验证目标，不能以排队128、实际逐段运行冒充支持。
- Qwen、CAM++、emo 分别排队与释放阶段名额，最终按请求/段标识汇合；模型实例数、预取等待深度、实际计算批大小分别定义，有界缓存避免慢分支无限积压。
- 验证配置为 VAD 八 CPU worker、CAM++ 八份 NPU:7、emo 八份 NPU:7，Qwen 先单份 NPU:7。VAD 按完整音频 FIFO 分派，任一 worker 完成即供下一条使用，不等待下游。
- 保持总音频上限8、B类独立子上限默认4；另设B上限8的测试配置。`showRoleIdentify` 本轮不传或为false，保留已有接口能力。
- 本轮测试关闭三项服务段级能量过滤，仅保留最短段0.3秒；不足0.3秒在三个模型分发前丢弃，0.3至不足1秒执行所请求CAM++，emo直接记“平淡”。
- 建立同源A/B配对、固定八条完整音频的1/2/4/8并发测试；记录各阶段排队/执行、真实批大小、HBM、吞吐及可复算P50/P95。420秒触发诊断、停止追加负载并安全收尾，未完成请求也受监测。
- 单 Qwen 容量或吞吐出现问题时，基于诊断评估2/4/8副本与段级负载均衡；不预先假定多副本一定更快，不静默升级依赖或切换生产引擎。

## Capabilities

### New Capabilities

无；复用已有推理、调度与运维能力边界。

### Modified Capabilities

- `local-qwen-asr-inference`：真实批推理、单音频/全局逻辑名额、批预算、取消所有权和单实例优先的容量验证。
- `local-model-runtime`：独立模型队列、可转移的待执行任务、预取窗口与实际批大小的独立语义，以及阶段完成后的及时归还。
- `local-model-inference-backends`：CAM++从0.3秒起实际推理及参与聚类，emo沿用1～10秒实际推理边界，保持FP32数值与业务映射。
- `qwen-asr-v2-adapter`：现行v3的阶段解耦、短段增强规则、VAD及时接力、指定测试过滤配置和新的配对性能验证范围；不恢复v2接口。
- `service-operations`：八CPU VAD验证配置、1/2/4/8完整音频报告、420秒诊断闸门、细分遥测和测试资源关闭证据。

## Impact

- 后续实施涉及 `api/routes/asr_qwen_common.py`、`core/qwen_asr_queue.py`、`core/qwen_asr_client.py`、`core/qwen_asr_runtime.py`、`core/models.py`、`core/local_inference.py`、`core/vad_pool.py`、`core/request_telemetry.py`、配置加载及性能脚本/测试；按实际设计拆出批调度模块。
- 配置变更同步 `core/config.py`、`config.toml`、`docker/config.toml`、README和AGENT；变更需重启测试服务，实际镜像构建与上线仍后置。
- `/v3/seacraft_asr` 字段、类型、错误码、既有 `gpu_time_ms` 计时口径不变；新增诊断放内部遥测/兼容状态扩展，不把7分钟变成业务HTTP超时。
- 使用 xjtu910b7 的 conda asr，仅占用物理NPU:7；保留共享既有服务和所有权重、原始音频、历史报告/日志。当前已进入apply；候选配置独立验证，只有完成验收后才更新交付默认值。完成实施及验收后中文提交推送，再独立同步规格并归档。


## 实施范围补充（v5）

按组件证据增加单Qwen模型进程隔离候选（`execution_mode`及独立关闭grace）：仍为同HF引擎、同权重与精度、单HTTP服务、单Qwen副本，新增core模型worker及ownership/IPC/保护构建验证。另将Qwen逻辑credit前置裁剪，避免首文件垄断PCM预算而饿住其它VAD-ready文件；完整音频与最终配置验收要求不减。

实施排障追加：完整8B触发420秒闸门后，CAM-only owned模型进程的同精度原型128段中位10.089→5.185秒且输出精确一致，支持进入正式候选实现。补充CAM模式/关闭与资源契约、保护版及全矩阵复验任务；仍保持单HTTP、单Qwen优先与原FP32模型，不将原型作为完整验收或提前改变缺省。

用户补充（2026-11-19）：保留根配置，提供可选Qwen启动预加载及单实例16/128独立测试profile；在test/提供Postman式v3请求客户端及示例，分清完整音频请求数/逻辑段窗口/物理批。此补充不宣布128为最优拐点，不扩大未测物理批上限、不替代最终矩阵。
