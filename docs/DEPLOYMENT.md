# 部署与运维说明

当前服务为华为 Ascend 910B 本地 Qwen3-ASR 链路，部署入口统一维护于 [docker/README.md](../docker/README.md)，宿主运行见 [README.md](../README.md)。

仅物理 npu:7；容器映射后使用逻辑 npu:0。先 source CANN，显式使用经核验的 asr 解释器。模型权重不可修改，既有服务不可停止，测试实例必须关闭并记录端口释放。
实际镜像构建与上线待服务稳定后执行。现有基础镜像只是候选，不能冒充已完成运行验收。

原 NVIDIA 操作说明保留为 [迁移前快照](npu-910b-migration-20261117/legacy-deployment-before-migration.md)，只用于追溯，禁止照旧执行。
