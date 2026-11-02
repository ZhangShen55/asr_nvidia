# Proposal

## Why

当 CAM++、emotion2vec、VAD worker 与多路完整音频同时运行时，服务可能出现主机内存压力、Swap 换页、GPU 显存/算力争用和请求长时间卡顿。当前状态接口能够显示模型池实例和部分显存信息，但还不能把模型常驻内存、音频缓冲、段级在途数据、Swap 和请求耗时关联起来，因此需要先建立安全的诊断和压测边界，再决定内部调度与内存释放优化。

## What Changes

- 建立资源压力诊断能力，采集主进程、VAD worker、模型池、主机内存、Swap、GPU、文件缓存和请求耗时的统一时间序列。
- 增加受控的分阶段验证流程，跳过已确认正常的 VAD-only A 项，从模型加载、单路完整模型请求逐步升压到多路音频和段级并发。
- 为测试建立黄色告警、红色终止和优雅收尾规则，发现 Swap 增长、OOM、GPU 显存不足、后端失联或 worker 异常时停止继续施压。
- 使用 PSS、`VmSwap`、`vmstat si/so` 和 GPU 进程显存区分模型常驻内存、请求缓冲增长、释放不完整和同卡算力争用。
- 形成可复算的测试报告，给出安全音频并发、段级并发、模型实例规模和后续内部逻辑优化建议。
- 将缓存清理、进程回收、Qwen 健康检查和 SeaCraftASR 优雅停止纳入测试收尾；测试不得停止或重启 Qwen-ASR。
- 本 change 只建立观测、诊断、测试和报告契约，不自动调整生产并发，不在未完成归因前修改模型实例数量或 VAD 算法。

## Capabilities

### New Capabilities

- `resource-pressure-diagnostics`: 定义本地模型、VAD worker、音频缓冲、Swap、GPU 和请求级资源压力的采集、告警、受控测试及归因报告。

### Modified Capabilities

- `service-operations`: 增加资源状态观测、健康检查、测试期间的安全终止和结果留存要求。

## Impact

- 影响监控/测试工具、`GET /get_status` 的资源观测字段、服务运维文档和测试报告目录；需要读取 `/proc`、`vmstat`、`nvidia-smi`、进程树和临时缓存目录。
- 测试目标服务端口为 8085，Qwen-ASR 端口为 8806；只检查 Qwen 健康状态，不停止 Qwen 服务。
- 现有 `/v2/seacraft_asr` 响应结构和模型推理逻辑保持不变；任何内存释放、段级队列或准入控制优化应在后续独立 change 中依据本 change 的证据制定。
- A 项 VAD-only 基线不在本 change 的测试范围内；测试从 CAM++/emotion2vec 模型加载和完整链路基线开始。
