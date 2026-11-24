# Proposal

## Why

本change最初将continuous多实例从隔离实验落入正式代码。正式N6纯A6墙钟186.32秒，但增强驻留A6真实OOM；随后用户授权的N5补测未复现OOM，增强驻留A5墙钟206.12秒。两轮原始结果、失败、警告和质量限制均保留，不能把纯A6或N5成绩冒充N6共驻通过。

2026-11-24用户明确采纳最终Qwen4／CAM4／emo1，完成该配置B4及同源增强驻留A4后关闭剩余验收。本次修订将N4作为替代最终验收范围，不删除原N6失败、不回填历史通过、不修改旧change或主规格。

## What Changes

- 正式continuous/process支持配置N个Qwen owned进程，最终默认local_instance_count=4，合法整数1～8仍保留；local_device="npu:7"、segment_concurrency=S=16，单HTTP服务instance_count仍为1。
- **BREAKING**：continuous总音频容量由N派生，全局段窗口由N×S派生；旧audio_concurrency/request_concurrency显式冲突须报迁移错误，根/Docker不重复维护派生值。最终总4/B4/global64，满4个A/B后新A/B均4013优先；该配置不能单独触发仅B满4017，显式N6下4B+2A及4017实测历史保留。
- 请求动态租用一个健康Qwen owner，整个音频生命周期固定，逐段独立结束和补位、安全清理后归还供任意下一请求使用；禁止SHA白名单、文件名绑定、实验环境开关或固定档位路由。
- 预加载、只读状态、取消、故障隔离、段级恢复/警告、排队和正常关闭保持正式契约；帧/token/物理步预算为每owner，PCM为全服务共享，不按N复制。
- 保持B类有效VAD段裁剪后三路异步分发，Qwen credit先归还、PCM等三分支收尾。CAM++/emo为独立共享池，最终根/Docker选择CAM4/emo1，须在四Qwen共驻B4条件下真实标定与配置冻结；emo时长skip及声纹聚类规则不变。
- 保留已执行N6 A6/A4/B4及共驻失败、N5补测；剩余只验证最终N4/CAM4/emo1的B4及增强驻留后的同a1～a4纯A4。记录完整/删失分位数、墙钟、警告/字段覆盖、池等待和HBM。约200秒仍为纯A目标单列，B4不强加200秒；336秒、OOM/故障停止和正常关闭要求保留，不恢复8GiB提前停止线。
- 不混入ITN执行器、供给隔离、prefill/KV、阈值或引擎重构；旧change任务和历史成绩不改写。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-qwen-asr-inference`：正式多owner、请求租约、每实例窗口、预加载及生命周期，最终缺省N4。
- `qwen-asr-v2-adapter`：派生总4/global64、B4准入与三分支独立归还/汇合，保留通用4017及N6历史。
- `local-model-runtime`：主协调/owned Qwen池和独立CAM4/emo1增强池、健康容量。
- `service-operations`：最终默认/迁移、实例身份、加载与正常收尾，根/Docker配置一致。
- `offline-asr-service`：continuous默认下段警告与既有字段/计时契约。
- `resource-pressure-diagnostics`：N6/N5历史及最终N4/CAM4/emo1 B4、增强驻留A4的统计与回收证据。

## Impact

实施涉及core/config、qwen client/queue/continuous process、lifespan/status、API租约、增强配置及验证工具。最终loader、根/Docker模板和说明须一致为Qwen4／CAM4／emo1，并以真实最终配置验收后冻结；用户选择不等于已经通过。保持原HF/BF16、conda asr、NPU7、v3字段和错误码，不更新权重/依赖、不实施Docker构建上线。

三个旧change保持open：throughput44/60、continuous20/25、feasibility26/32。新策略替代旧throughput跨副本按段均衡与旧continuous单实例的正式continuous范围，未验收旧delta不整体提升为main。当前只修订本change规划，5.3/5.6仍未勾；最终验收后由root独立处理同步/归档，旧失败和未完成矩阵不被新范围代替。
