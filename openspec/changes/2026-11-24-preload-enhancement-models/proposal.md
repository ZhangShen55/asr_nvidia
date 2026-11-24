# Proposal

## Why

CAM++、emotion2vec目前首次被请求需要时才加载，设备数组非空也不会在启动时加载。将加载提前到HTTP就绪前，避免首个增强请求承担权重加载耗时。

## What Changes

- **BREAKING**：已开启且设备数组非空的增强模型池，启动时完成全部配置实例的加载尝试并常驻复用，不再等待首个增强请求。
- 沿用功能关闭、空数组禁用及部分/全部加载失败的现有降级行为；状态如实报告健康数与失败原因。
- 复用现有启动收尾、模型池、设备数组和状态字段；同步当前行为说明。

## Capabilities

### New Capabilities

无。

### Modified Capabilities

- `local-model-runtime`：增强池启动预加载、共享复用与状态语义。
- `service-operations`：就绪前完成增强加载尝试，预检与查询不触发加载。

## Impact

涉及main.py、core/models.py及相关生命周期测试和README/AGENT.md/部署说明。不新增开关、依赖或接口字段，不改变实例数、设备、推理/调度策略。仅预加载权重，不加入启动语音推理暖身、吞吐目标、完整音频压力矩阵或Docker构建上线任务。
