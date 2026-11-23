# 实施检查点

2026-11-24：用户已授权apply。基线`fd217f8e3f2251a8347657d743132e00b5eeeab9`，正式工作区干净，上游一致。仅NPU7；共享PID1601397保留。独立记录目录`/tmp/qwen-multi-apply-20261124-sduu4fzl`；文件唯一作者见该目录setup.json。旧三个change任务/结果保持原样。配置、runtime、增强可并行，root负责API/控制器/全部NPU；各agent当前只编辑，不运行测试。CPU测试必须完整r4+candidate+support guard及独立cache/TMPDIR；不与NPU压力重叠。

配置、owner池与B分支实施及目标回归完成15/29。源码CPU首轮`cpu-target-v1`为103 passed/40 subtests、10新增owner异步测试因环境无pytest-asyncio未执行；保留失败证据后改标准asyncio.run，`cpu-owner-v2`12 passed。两轮guard设备/权重与共享cache拦截均0、source drift空、owned归零。完整source/实际保护回归和正式NPU矩阵尚未执行，不将组件/CPU通过当吞吐成绩。

完整源码`cpu-full-v3`1289 passed/379 subtests、2保护专属skip、1既定硬件probe排除；实际保护`cpu-protected-v1`364 passed/244 subtests，40份SO真实导入、源码fallback为0。两轮均return0、guard真实设备/权重执行0、共享cache拦截0、源无漂移、owned归零；保留安全只读npu-smi拒绝记录及ephemeral CPU loopback记录，不伪称拦截记录全空。保护stage`/tmp/qwen-multi-protected-20261124-ah3u4jrq`重编11改动模块、29未变已验证二进制复用。源码回归初轮和次轮失败/夹具修复均保留，最终full-v3已全通过。

正式目录日志轮转会清理历史，故测试OwnedService新增可选cwd，正式controller以绝对main.py入口+同一正式CONFIG_PATH在独立service-cwd运行；模型/预算/根配置未变。源与保护的日志隔离/控制器增量回归正在进行；通过并确认全部CPU结束才启动真实NPU矩阵。

正式v2运行中：六owner实际forward及npu7/BF16/每worker Torch与interop均1已核验；启动与暖身分列。正式A6已6/6完整返回，P50=173.241736s、P95=185.420846s、墙钟186.323683s，目标200s满足；13段警告/11不可恢复，人工质量未认证。当前继续同源A4/B4和共驻矩阵，最终正常回收尚未完成，不提前算整change完成。原v1因共享服务容器netns端口未观测在OwnedService前停止，未启动服务/模型；现已按fd inode筛选/proc/PID/net完成只读核验8007，源/保护各40case增量通过。

2026-11-24最终运行结束：正式A6 6/6，173.24/185.42/186.32秒；A4 4/4墙钟200.93秒；B4 4/4，289.81/301.37/302.60秒，CAM/emo真实覆盖2775、2226+549skip。真实准入探针通过。增强驻留A6在179.95秒因真实HBM OOM停止：3完成/1失败/2取消，整批P50/P95未知，dev7 owner5申请22MiB失败Err207001，最低剩余HBM采样132MB。按照test-plan停止后续压力，不继续CAM2/4。5.3/5.6未通过，保持open不归档；详见docs/qwen-multi-instance-20261124/REPORT.md。

全部owned正常exit0、TERM至父退出43.14秒、端口空闲≥5秒、PCM空，root再次PID/birth核验无残留/控制器退出、共享1601397/8007保留。背景HBM +2MB回到近背景，原始unknown不改写。源码完整1289/379、保护364/244以及日志隔离两边84/netns两边40通过，所有CPU/NPU任务已结束。配置仅补注释，根语义与实际快照一致；Docker未构建/验收。具体文本差异HTML/JSON已交付并独立核对实际响应例子。当前进行严格校验、提交推送及干净工作区检查，6.2完成后27/29。

交付审查完成27/29：change strict1/1、main strict9/9、临时合并预览strict9/9通过，逐Requirement覆盖审计无旧main条目丢失；无main/旧change/依赖/权重改动。任务6.2随本次中文Conventional Commit提交并推送，HEAD/上游/工作区最终状态记录于运行目录delivery.json及本机恢复索引；5.3/5.6保留未勾。
