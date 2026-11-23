# 最新追加检查点（2026-11-23）

用户已明确授权六/八模型实例独立验证，本次任务6.1～6.5完成；累计17/23，旧矩阵/最终结项不自动完成，仍open。六/八容量均真实加载及每worker v3短clip forward通过；六模型原八完整WAV八HTTP/16/128在49.362秒free7738MiB触8GiB红线，0/8完整返回，完整分位数/速度比未知。依分支闸门不进入八模型完整压力，S1不恢复。

全部八次已启动服务normalexit0、无forced、665出生身份退出、18083空闲5秒、PCM空，共享1601397保留，卡7约24%恢复。业务/默认/权重/依赖无改动，实验MN补丁与42+28目标CPU证据及真实日志均保留，source/assets无漂移。详见docs/qwen-200s-feasibility-20261123/MN-REPORT.md；无活跃CPU/NPU。旧M2准入不合规记录仍保留；200秒尚无可行性证明。

以下为追加前及过程记录，新检查点优先。

# 实施检查点（2026-11-23）

当前12/18，暂停后续压力，等待用户对整批停止边界的选择。正式业务源码/默认未改，旧continuous20/25、throughput44/60均未改。权威规划起点c7547b7，实验业务基线a1de07f。

独立副本/tmp/qwen200-20261123-j9r2czoi/{b0,m2,s1}及assets保留；所有已启动服务正常exit0、无forced、222个登记出生身份退出、端口每轮空闲5秒、私有PCM空，现无本轮活跃NPU。共享PID1601397保留，driver已恢复测试前约15.5GiB占用。exec sessions均结束。

CPU最终common15/M2 31/S1 33/controller19通过；完整r4+candidate+support证据及源码SHA见docs/qwen-200s-feasibility-20261123。B0/M2/S1各8次真实v3短clip准入通过，实际worker完成分别8、4+4、8；24个同clip响应剔除计时后一致，仅准入不算性能/质量验收。

原八完整音频真实HTTP：B0在91.954秒driver free5714MiB触发8192MiB红线，0/8返回；M2在66.582秒free7135MiB触发同红线，0/8返回。均非完整处理时长，P50/P95及整轮精确提速比null。B0 worker扣暖身完成793段，M2内部485+401段，仅中止进度。

协议偏差：root误按单轮理解整批停止，在B0红线正常回收后继续启动M2。该偏差已向用户承认、记录m2-controller-audit.json protocol_admission_compliant=false/整体passed=false，M2仅观察，不支持原矩阵积极结论。随后所有后续压力暂停，异步问题待选择：结束本轮交付证据，或明确允许S1独立新批次且保留8GiB红线。未收到选择前不得启动S1完整轮或扩展/复验。

S1完整八文件、M4/M8/组合、三轮200秒复验未测。没有完整正式响应作全文对比，质量未知。report/decision-log/cleanup-verification及原始logs/qwen-200s-feasibility-20261123保留。新change继续open，不把资源停止泛化为硬件绝不可能200秒。

若用户明确允许继续：先修订并记录批次边界、再核对资源恢复和S1源码冻结，命令参照assets/controller-readme.md；s1副本QWEN200_MODEL_THREADS=4、QWEN200_CUT_THREADS=4、QWEN200_CUT_CAPACITY=128，八源manifest和八暖身manifest仍在assets。原HF@npu7、16/128、prefill16、3ms、生成256/复读参数不变。禁止CPU测试与NPU轮并行。不要启用--enable-eta-stop：既有真实回放不能证明ETA可信，只报告ETA，hard336+resource8GiB有效。

## 用户追加：6／8实例独立验证（2026-11-23）

用户在HBM拆分讨论后明确要求测试6～8个模型实例，本次按6份与8份两个档位执行。这是新的独立验证批次授权，替代只限旧B0/M2/S1矩阵的暂停边界，不追认旧M2准入偏差，不恢复S1完整测试。仅临时副本接线，不修改正式业务/默认/权重/依赖。

顺序先M6容量准入，再M8容量准入：逐个真实加载、八次带原源路由文件名的真实v3短clip请求，核对全部实例真实forward。M6仅加载及暖身就触红线或无法正常收尾，则停止M8，记录未测；M6通过且资源恢复后才独立起M8。两档容量通过后，只有额外推理余量允许时，才考虑原八完整WAV真实v3的M6，再依据结果判断M8；任一完整轮资源红线或收益不足停止该压力分支，不继续更高档完整压力。短clip结论只描述加载/基础推理容量，不能证明16段/实例稳态或200秒性能。

总音频仍8、每音频16、全局128。M6按source index%6固定分配，owner音频数[2,2,1,1,1,1]、窗口[32,32,16,16,16,16]；M8各一条/16段。token80000、frame115200按音频份额分担，不能均分6或复制N倍。每模型CPU计算线程最少1，本轮M6/M8总线程6/8，明确不同于旧总4，不声称是严格单因素吞吐比较；父OMP/MKL4及原八VAD不变。单owner frame14400小于最长16×901，最长输入prefill可能只有15，记录此预算影响。

保持剩余HBM8GiB红线、200秒目标和336秒硬停止。加载期间HTTP未ready时每2秒通过只读npu-smi精确卡7采样；未知fail-closed。控制器私有TMPDIR内stopfile在红线/取消/超时触发，client在每worker加载前及安全返回后检查，不启动后续worker，当前加载仍等待安全终态。启动红线可能导致预期exit3，单列为已回收的启动中止，不能伪装正常推理exit0。startup观察期限600秒只覆盖顺序加载，不放宽正式处理时限。各批正常回收、资源恢复后才能下一批，保留共享服务。

MN副本/tmp/qwen200-20261123-j9r2czoi/mn；MN路由CPU42pass、startup controller28pass，均完整r4/candidate/support guard且设备/权重/cache0，CPU已全部停止。下一步6容量再8容量，旧S1仍暂停。

M6/M8容量均通过，六worker完成2/2/1/1/1/1、八worker各1个真实v3段，全部normalexit0、源码冻结及端口回收通过。管理连接曾短暂延迟，回联核验没有模型加载失败。现按追加范围测试M6原八完整WAV，M8完整压力仅在M6正常安全/目标有望时准入；触发红线不升级。
