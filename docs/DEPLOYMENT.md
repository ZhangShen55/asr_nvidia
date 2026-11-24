# 部署与运维说明

当前服务为华为 Ascend 910B 本地 Qwen3-ASR 链路，部署入口统一维护于 [docker/README.md](../docker/README.md)，宿主运行见 [README.md](../README.md)。

仅物理 npu:7；容器映射后使用逻辑 npu:0。先 source CANN，显式使用经核验的 asr 解释器。模型权重不可修改，既有服务不可停止，测试实例必须关闭并记录端口释放。
实际镜像构建与上线待服务稳定后执行。现有基础镜像只是候选，不能冒充已完成运行验收。

原 NVIDIA 操作说明保留为 [迁移前快照](npu-910b-migration-20261117/legacy-deployment-before-migration.md)，只用于追溯，禁止照旧执行。

## 正式continuous多owner迁移

loader及根/Docker模板默认N=4、S=16、process、continuous、preload=true；HTTP仍1。总音频N/全局段N×S由loader派生，删除旧audio_concurrency/request_concurrency显式赋值；相等值提示迁移，冲突及非法类型启动前报错。显式legacy保留原独立容量配置。physical步缺省S、prefill缺省min(16,S)，模板不固定；每owner frame/token=14400/10000，PCM全服务256段/256MiB。每owner线程owner_threads=1；父进程OMP/MKL=4。所有更改需重启生效。

宿主只用物理npu:7，单卡容器使用逻辑npu:0；不能直接复制宿主编号。VAD八CPU保持不变；最终选择Qwen4/CAM++4/emo1，CAM采用process，两个增强数组独立，N变化不复制增强池。总音频4、全局段64由loader派生。本轮宿主B4与增强驻留A4已完成4/4验收。B配置上限4，有效值不超过健康owner数。Qwen预加载失败须阻止ready并回滚本次资源；Qwen懒加载可显式preload=false。设备数组非空的CAM++、emotion2vec在HTTP就绪前完成全部配置实例加载尝试并常驻；空数组跳过，加载失败按增强池既有规则降级。此步骤不执行语音forward暖身，状态查询不得触发模型加载。

先完成正式源码及保护回归，再由root独占NPU7执行本轮B4与增强驻留A4验证。当前配置落地不表示性能/质量或Docker上线已验收。实际正常close/PID出生身份/端口释放证据必须保留，强制清理不计正常关闭通过。

2026-11-24历史N6实际验收（原冻结配置）：纯A6为6/6、P50/P95=173.24/185.42秒、墙钟186.32秒；B4双增强为4/4、289.81/301.37秒、墙钟302.60秒。增强驻留后的A6因真实HBM OOM中止，未通过长期共驻验收，CAM/emo各1仍是候选，未继续2/4。全部owned正常exit0并释放端口/PCM，共享服务保留。详见[正式报告](qwen-multi-instance-20261124/REPORT.md)。上述结果不对应当前Qwen4/CAM++4/emo1模板；最终宿主结果见下文。Docker CAM模式已同步process，仍保留其原有VAD过滤等差异，未构建或验收，不应直接视为宿主成绩复现。

最终宿主Qwen4/CAM4/emo1组合已验收：B4墙钟200.10秒、增强驻留A4 183.47秒、最低free采样15406MB；正常exit0、端口/owned/PCM回收、共享服务保留。当前change 29/29，原N6失败保留；详见[最终报告](qwen-multi-instance-20261124/FINAL4-REPORT.md)。Docker仅同步配置/逻辑设备映射，尚未构建/运行验收或上线；质量警告及缺人工GT不因功能验收而消失。


### 增强能力设备数组统一配置（2026-11-24）

CAM++、emotion2vec启用状态现在只来自各自设备数组：非空数组开启，每元素一实例；`[]`关闭。数组缺省为空。`showSpk`/`showRoleIdentify`选择CAM++执行，`showEmotion`选择emo执行，均不能开启空数组能力。模型在HTTP就绪前按非空数组预加载，请求复用健康实例；A请求不执行增强，已加载权重继续驻留。预加载不执行语音forward暖身。

旧 `[features].open_v2_role`、`open_v2_emotion` 已删除，残留无论true/false都会在任何worker/模型创建前报迁移错误。迁移办法：原true删除旧键并保留设备数组；原false删除旧键，并把对应的 `speech_campplus_devices` 或 `emotion2vec_devices` 改为 `[]`。删除空 `[features]` 表。不允许只删原false键却保留非空数组，以免开启原已关闭的模型。

单项关闭请求仍在音频准入前返回4015/4016，双请求且两项均关闭返回4018，优先于容量错误，拒绝计数不变。非空数组全部加载失败是模型故障，不是配置关闭，不返回4018。数组改变需重启源码服务；保护构建和镜像在后续构建纳入。本次不构建上线；三个镜像内置小模型的必需资产预检保持，空数组不表示可以删除权重。

正式continuous根/Docker模板省略两项legacy组批参数和两个固定增强batch配置；legacy显式参数支持、实际batch1及非1校验保持。两个 `prefetch_queue_depth=0`、全部 `online_repetition_*` / `repetition_*`、`retry_count`、队列及VAD参数保持；预取0是在上游反压等待，不是无限预取。早先章节中涉及旧功能开关的已日期标记说明仅代表当时版本，以本节为当前规则。
