# 实施检查点（2026-11-23）

当前12/18，暂停后续压力，等待用户对整批停止边界的选择。正式业务源码/默认未改，旧continuous20/25、throughput44/60均未改。权威规划起点c7547b7，实验业务基线a1de07f。

独立副本/tmp/qwen200-20261123-j9r2czoi/{b0,m2,s1}及assets保留；所有已启动服务正常exit0、无forced、222个登记出生身份退出、端口每轮空闲5秒、私有PCM空，现无本轮活跃NPU。共享PID1601397保留，driver已恢复测试前约15.5GiB占用。exec sessions均结束。

CPU最终common15/M2 31/S1 33/controller19通过；完整r4+candidate+support证据及源码SHA见docs/qwen-200s-feasibility-20261123。B0/M2/S1各8次真实v3短clip准入通过，实际worker完成分别8、4+4、8；24个同clip响应剔除计时后一致，仅准入不算性能/质量验收。

原八完整音频真实HTTP：B0在91.954秒driver free5714MiB触发8192MiB红线，0/8返回；M2在66.582秒free7135MiB触发同红线，0/8返回。均非完整处理时长，P50/P95及整轮精确提速比null。B0 worker扣暖身完成793段，M2内部485+401段，仅中止进度。

协议偏差：root误按单轮理解整批停止，在B0红线正常回收后继续启动M2。该偏差已向用户承认、记录m2-controller-audit.json protocol_admission_compliant=false/整体passed=false，M2仅观察，不支持原矩阵积极结论。随后所有后续压力暂停，异步问题待选择：结束本轮交付证据，或明确允许S1独立新批次且保留8GiB红线。未收到选择前不得启动S1完整轮或扩展/复验。

S1完整八文件、M4/M8/组合、三轮200秒复验未测。没有完整正式响应作全文对比，质量未知。report/decision-log/cleanup-verification及原始logs/qwen-200s-feasibility-20261123保留。新change继续open，不把资源停止泛化为硬件绝不可能200秒。

若用户明确允许继续：先修订并记录批次边界、再核对资源恢复和S1源码冻结，命令参照assets/controller-readme.md；s1副本QWEN200_MODEL_THREADS=4、QWEN200_CUT_THREADS=4、QWEN200_CUT_CAPACITY=128，八源manifest和八暖身manifest仍在assets。原HF@npu7、16/128、prefill16、3ms、生成256/复读参数不变。禁止CPU测试与NPU轮并行。不要启用--enable-eta-stop：既有真实回放不能证明ETA可信，只报告ETA，hard336+resource8GiB有效。
