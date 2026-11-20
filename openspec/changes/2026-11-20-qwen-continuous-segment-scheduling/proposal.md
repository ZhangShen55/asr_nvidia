# Proposal

## Why

当前单 Qwen 的整批 generate() 返回形成完成屏障，无法实现用户要求的每音频 16 段独立完成、即时补位；异常复读又可能占住执行时间直到生成上限。真实八音频服务在 420 秒仍未完成，与独立 128 段约 4.60 秒存在尚未闭环的性能差距，需要正确性、调度和真实接口共同验收。

## What Changes

- 单 Qwen、原 HF 权重/BF16、Transformers + torch_npu 路线下，支持逐段提交、独立生成状态/KV、独立完成及持续接纳新段；底层可按解码步批量计算。
- 最多八完整音频，各自最多 16 个 Qwen 在途段，全局最多 128；公平调度、完成即补位、有界前端预取与内存反压，保持 VAD 边界和结果归属。
- 生成中逐段检测异常复读并终止异常段，保留生成后文本修复；正常重复、非 EOS 结束与失败不得被掩盖。
- 明确未接收容量拒绝的最多三次重试/30秒窗口与幂等归属，取消/超时不提前释放真实设备资源。
- 增加段级阶段、槽位与结束原因诊断，配对定位旧服务和独立推理差距。
- 必须通过真实服务 v3 的原八 WAV、八并发、16/128 测试，报告成功率/P50/P95/420秒完成数/HBM及具体文本差异，结束释放自有资源。
- 规划多 Agent 分工：执行后端、上游调度、复读与质量分别并行，主 Agent 集成；同卡 NPU 验证串行。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-qwen-asr-inference`: 单实例执行所有权细化为串行设备步、独立序列生命周期；持续补位、公平窗口、异常重复和准入重试契约。
- `resource-pressure-diagnostics`: 连续序列诊断、同源性能归因、真实八音频验收与可复算统计。

## Impact

预计涉及 core/qwen_asr_runtime.py、qwen_asr_process.py、qwen_asr_client.py、qwen_batch_scheduler.py、qwen_asr_queue.py、segment_pipeline.py、core/config.py、api/routes/asr_qwen_common.py、utils/repetition.py 及诊断、测试、配置说明。实现阶段可能新增独立序列调度/KV/检测模块；本提案不修改运行代码、配置或依赖，不启动 NPU，不需重启或重建镜像。

不纳入 OM、vLLM/Docker 替换、多 Qwen 副本、模型重新训练及 VAD/增强能力算法变更。保留 v3 字段、错误码、输入前端和原时间轴。420秒八音频全部完成是待实测目标，不能以推算代替验收。

旧 `2026-11-18-npu-asr-throughput-scheduling` 保持 open，历史失败和未验任务不改写；本 change 只负责新增连续生成工作。实施前须做任务/差量映射，旧静态整批假设不能覆盖本契约，归档前检查两 change 合并顺序及冲突，禁止把旧验收默认勾完。
