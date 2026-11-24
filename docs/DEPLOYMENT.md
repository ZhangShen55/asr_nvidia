# 部署与运维说明

当前服务为华为 Ascend 910B 本地 Qwen3-ASR 链路，部署入口统一维护于 [docker/README.md](../docker/README.md)，宿主运行见 [README.md](../README.md)。

仅物理 npu:7；容器映射后使用逻辑 npu:0。先 source CANN，显式使用经核验的 asr 解释器。模型权重不可修改，既有服务不可停止，测试实例必须关闭并记录端口释放。
实际镜像构建与上线待服务稳定后执行。现有基础镜像只是候选，不能冒充已完成运行验收。

原 NVIDIA 操作说明保留为 [迁移前快照](npu-910b-migration-20261117/legacy-deployment-before-migration.md)，只用于追溯，禁止照旧执行。

## 正式continuous多owner迁移

loader及根/Docker模板默认N=4、S=16、process、continuous、preload=true；HTTP仍1。总音频N/全局段N×S由loader派生，删除旧audio_concurrency/request_concurrency显式赋值；相等值提示迁移，冲突及非法类型启动前报错。显式legacy保留原独立容量配置。physical步缺省S、prefill缺省min(16,S)，模板不固定；每owner frame/token=14400/10000，PCM全服务256段/256MiB。每owner线程owner_threads=1；父进程OMP/MKL=4。所有更改需重启生效。

宿主只用物理npu:7，单卡容器使用逻辑npu:0；不能直接复制宿主编号。VAD八CPU保持不变；最终选择Qwen4/CAM++4/emo1，CAM采用process，两个增强数组独立，N变化不复制增强池。总音频4、全局段64由loader派生。本轮宿主B4与增强驻留A4已完成4/4验收。B配置上限4，有效值不超过健康owner数。Qwen预加载失败须阻止ready并回滚本次资源；Qwen懒加载可显式preload=false。已开启且设备数组非空的CAM++、emotion2vec在HTTP就绪前完成全部配置实例加载尝试并常驻；关闭功能或空数组跳过，加载失败按增强池既有规则降级。此步骤不执行语音forward暖身，状态查询不得触发模型加载。

先完成正式源码及保护回归，再由root独占NPU7执行本轮B4与增强驻留A4验证。当前配置落地不表示性能/质量或Docker上线已验收。实际正常close/PID出生身份/端口释放证据必须保留，强制清理不计正常关闭通过。

2026-11-24历史N6实际验收（原冻结配置）：纯A6为6/6、P50/P95=173.24/185.42秒、墙钟186.32秒；B4双增强为4/4、289.81/301.37秒、墙钟302.60秒。增强驻留后的A6因真实HBM OOM中止，未通过长期共驻验收，CAM/emo各1仍是候选，未继续2/4。全部owned正常exit0并释放端口/PCM，共享服务保留。详见[正式报告](qwen-multi-instance-20261124/REPORT.md)。上述结果不对应当前Qwen4/CAM++4/emo1模板；最终宿主结果见下文。Docker CAM模式已同步process，仍保留其原有VAD过滤等差异，未构建或验收，不应直接视为宿主成绩复现。

最终宿主Qwen4/CAM4/emo1组合已验收：B4墙钟200.10秒、增强驻留A4 183.47秒、最低free采样15406MB；正常exit0、端口/owned/PCM回收、共享服务保留。当前change 29/29，原N6失败保留；详见[最终报告](qwen-multi-instance-20261124/FINAL4-REPORT.md)。Docker仅同步配置/逻辑设备映射，尚未构建/运行验收或上线；质量警告及缺人工GT不因功能验收而消失。
