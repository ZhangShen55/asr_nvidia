# 最新追加：六／八实例验证已完成（2026-11-23）

六/八实例均真实加载并覆盖每worker v3短clip推理，短暖身末card free分别22028/13019MiB。六实例原八完整音频、八HTTP、16/128在49.362秒因free7738MiB触发8GiB红线，0/8完整返回；八实例完整压力按闸门未进入。全部服务正常exit0、无forced，正式源码/默认未改，200秒可行性仍未证明。当前change17/23，保留open；详情见[MN-REPORT.md](MN-REPORT.md)。本次为用户追加授权的独立批次，不追认旧M2协议偏差。

以下保留追加前的检查点及解释，旧“等待选择”与“八实例未测”仅描述当时状态。

# 200秒真实场景验证检查点（2026-11-23）

当前结论：**证据不足，尚未证明200秒可行；暂停后续压力，等待用户选择。** 三组真实v3短片段准入通过，完整八文件基线因显存闸门提前终止。M2完整轮也中止，且存在矩阵准入协议偏差，不能用它支持遵守原计划的积极结论。正式代码/默认未改，旧change任务未改。

|真实服务轮次|实际处理墙钟|完整音频返回|结果|
|---|---:|---:|---|
|B0单模型原八完整音频|91.954秒|0/8|driver可用HBM5714MiB，低于8192MiB红线，停止|
|M2双模型原八完整音频|66.582秒|0/8|driver可用HBM7135MiB，低于相同红线，停止；准入协议偏差，单列观察|

这两轮均未跑到200/336/420秒，不能称为实测全轮耗时，完整P50/P95及精确提速比均为null。阶段完成段数和累计执行包络也不能当作完整音频完成。资源闸门是本验证预设的安全条件，非OOM，不能据此断言硬件绝对无法200秒。

## 已验证与真实性

冻结业务a1de07f，权威分支c7547b7；独立副本/tmp/qwen200-20261123-j9r2czoi/{b0,m2,s1}，原八WAV/SHA重新验证，真实POST /v3/seacraft_asr、A纯转写、八并发、每音频16/global128。完整轮执行真实音频加载/VAD/0.3秒过滤/裁剪/IPC/Qwen/ITN链路，未用选样128段或缓存VAD替代。

CPU目标回归：公共数值/RPC15pass、M2最终31pass、S1最终33pass、controller19pass；均有独立r4/candidate/support设备权重与cache隔离，正式NPU期间CPU压力停止。纯host计时helper的开销测量另列，只能描述helper开销，不是NPU性能证据。

B0/M2/S1各一次真实v3短准入、八次同clip请求。B0单worker实际完成8段，M2两个worker各4段，S1单worker8段；24响应剔除计时后payload一致，全部正常关闭。短clip只证明加载、路由、基础文本及收尾，不证明八完整音频速度或准确率。

正式实验未改生成参数/权重/精度/过滤，max_new_tokens256、BF16、复读/恢复策略一致。模型CPU重算线程总数固定4（B0/S1每模型4，M2每模型2），父OMP/MKL4、八CPU VAD；每worker预算分担见manifest。两正式轮source及assets均无漂移。

## 基线归因（扣除暖身）

停止前worker完成793段（EOS792、复读停止1），已接收865段，prefill193次、平均4.482段；decode212步。正式prefill包络36.844秒、decode26.276秒。

host细分：prefill模型forward及既有回读29.648秒、processor6.272秒；decode模型forward及回读14.438秒、KV布局4.702秒、逐序列bookkeeping6.774秒。均是已有安全边界的host包络，不是纯NPU kernel耗时；并行段累计值不加到墙钟。发送锁等待累计1903.189秒、实际Pipe写包络累计67.383秒，存在重叠，说明提交阻塞需要关注，但不能据此计算可节省1903秒。

requestTelemetry的completed=8及阶段分位数在停止时均属于暖身，不能作为正式八文件耗时。worker复读警告1与summary警告0口径不同：后者只统计完整返回，而本轮没有完整返回。

## 显存证据与边界

B0停止时worker allocator当前allocated5942.6MiB、峰allocated14624.4MiB、reserved42558MiB；KV逻辑约919MiB、存储约2053MiB。reserved与当前allocated差额很大，但不能直接等同立即可复用连续空间，也不能据此证明泄漏或具体碎片原因。driver同轮used59822MiB/free5714MiB还包含既有服务与设备运行时。

M2两个worker的最近缓存allocated4803.7/5599.4MiB，reserved12058/30422MiB；各自峰allocated8464.7/11086.1MiB。两worker缓存与driver不是严格同一时间的采样，禁止相加当作同时峰值。各worker正式完成485/401段，仅为中止前内部进度，不外推200秒完工能力。

## 协议偏差及未测范围

原design规定资源红线后整批停止。root曾按单轮解释，在B0正常回收、显存恢复后又启动M2；这是计划执行偏差，已向用户说明并停止全部后续压力。M2保留为观察记录，不隐藏，不作为符合原矩阵的积极证据。m2-controller-audit.json明确protocol_admission_compliant=false、整体passed=false；controller一致性通过不等于实验准入合规。

S1仅短准入，完整八文件未测；M4/M8、组合、三轮200秒复验未进入。当前没有完整八文件输出可作全文差异，质量为未知；preflight文本一致不能代替质量验收。用户已被询问结束本轮或明确允许S1新独立批次（保留8GiB红线），未收到选择前不启动新压力。

## 收尾及交付

五次已启动服务均normalexit0、无forced、所有登记PID出生身份退出、私有PCM空、端口每轮至少5秒空闲；既有PID1601397保留。B0/M2正式轮TERM到父进程退出分别56.206/56.863秒，单列，不计入正式处理墙钟。见cleanup-verification.json与checkpoint-npu-smi.txt。

原始日志保留logs/qwen-200s-feasibility-20261123/；冻结实验补丁在experimental-patches/，仅验证资产，不是正式业务改造。新change仍open，旧change及其420秒成绩不改写。

## 单实例HBM补充核对

刚加载后的tensor allocated3889.25MiB（3.80GiB）、reserved4296MiB（4.20GiB）；相同短clip暖身后，driver单Qwen进程4651MB（约4.54GiB），双实例各4651MB。此前41.56GiB是完整动态负载下allocator reserved，包含当前allocated，不能与权重/KV再相加。停止采样点allocated5.80GiB几乎可由约3.80GiB加载张量及约2.01GiB KV存储解释，live峰14.28GiB说明还有推理临时张量/工作区；reserved与current差额约35.76GiB，但是否为可复用缓存、碎片/等待事件等需更多allocator统计，不能认定泄漏。代码动态KV布局的index_select/pad/cat与逐token扩展会产生不同形状临时张量，是待证的保留空间增长机制。8实例仅短暖身基础线性估计约36.3GiB，未实际验证8份，更不代表能承载8份完整负载。原始数值与口径见hbm-breakdown.json。
