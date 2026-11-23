# B0真实轮只读分析

该轮在**91.954秒**因NPU7 free HBM=5714MiB低于8192MiB红线终止，HTTP完整请求0/8。不是336秒预测停止，不能外推完整处理时长或判断硬件绝对达不到200秒。源码冻结无漂移。

## 扣除八次暖身后的设备owner进度

- worker累计801减暖身8，正式完成**793段**：EOS=792、复读停止=1（不可恢复warning1）。
- 正式接收865段，prefill 193次，平均**4.482成员/次**；decode 212步。
- 正式prefill 36.844秒、decode 26.276秒，合计63.120秒；已减去暖身2.623/17.850秒及8/312步。
- 正式KV布局整理203次、直接沿用布局9次（合计212步）。这是布局统计，不代表只有少量token使用KV。
- worker采样活跃72、全生命周期活跃峰93。HTTP活动进度Qwen完成和为795，比worker多2，因HTTP/worker快照时间不同；不能误判串段或计数错误。

## 新增host包络计时归因

|阶段（正式增量）|秒|含义|
|---|---:|---|
|prefill forward与readback|29.648|含设备等待，不是纯kernel|
|processor输入处理|6.272|CPU前端及拼接包络|
|decode forward与readback|14.438|含设备等待|
|decode KV布局|4.702|缓存分组/选取/补齐/拼接包络|
|decode每成员bookkeeping|6.774|KV视图、成员状态/token等更新|
|prefill transfer/position/readback|0.350|已有边界包络|
|decode输入组装|0.317|输入/位置组装|
|prefill/decode显式sync|0.117|只有显式同步API这部分；不包含隐式readback等待|

prefill/decode的host包络中，forward/readback合计44.086秒；布局整理4.702秒、bookkeeping6.774秒也已可观察。阶段为子区间，不能将这些数重复加到prefill/decode总计上，更不能称全部为NPU纯算子耗时。

父端同时有**32个pending send**，最老2.699秒；正式累计submit锁等待1903.189秒、write包络67.383秒、write次数875。锁等待是多线程重叠累计，write又与模型计算重叠，不能与91.954秒相加；这些是提交线程被占用的证据，尚不是线程池隔离真实收益的证明。

活动请求没有完成，requestTelemetry的completed=8和phases分位数仍全部属于暖身。不得将其中VAD约72ms/ITN约22ms等当作八完整音频正式阶段耗时。

## 显存口径

|口径|MiB|
|---|---:|
|当前allocator allocated|5942.605|
|生命周期peak allocated|14624.352|
|allocator reserved|42558.000|
|reserved减当前allocated|36615.395|
|当前逻辑KV|919.078|
|当前KV底层storage|2053.188|
|driver归属到本worker进程|43971|
|整卡driver HBM used/free|59822/5714|

暖身reserved4390MiB到42558MiB，allocated3905.572到5942.605MiB。reserved显著高于当前活跃分配和逻辑KV，说明仅靠KV字节不能解释整卡资源压力；但未做allocator分配轨迹/碎片/稳态回收实验，**不能据此断言泄漏、某个碎片机制或硬件绝对上限**。进程driver占用还包含allocator外资源；整卡占用还包含既有服务和驱动口径，采样时刻也不同，不能精确归一或将峰值相加。

## 输出及关闭

三个preflight各8个真实HTTP同clip，共24个响应剔除gpu_time_ms/load_audio_time_ms后全payload一致：**True**。这是短clip正确性证据，不是完整八音频质量/性能证据。

正常exit=0、forced=False，TERM至parent退出56.206秒；剩余owned PID/PCM空、端口连续5秒释放，shutdown passed=True。原始路径及四份证据SHA见b0-analysis.json。
