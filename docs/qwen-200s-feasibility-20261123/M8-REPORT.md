# 八实例八完整音频真实接口验证（2026-11-23）

结论：本次已真实启动八实例服务、提交原八完整WAV八并发请求。33.289秒时NPU7剩余HBM7334MiB触发8192MiB红线，0/8完整返回，按约定停止并正常回收。停止由资源条件触发，没有跑到慢请求截止，没有发生OOM；完整效率与200秒目标仍未知。不能把资源中止时间称为完整处理耗时，也不能仅凭这轮声称Qwen纯计算慢或硬件到顶。

## 实测指标

|指标|结果|
|---|---|
|真实入口|POST /v3/seacraft_asr；A模式，八完整WAV八HTTP共同起跑|
|模型/设备|八Qwen HF/BF16/torch_npu进程，同一物理NPU7；每owner固定一音频|
|段窗口|每音频16/global128；逐段独立完成、补位|
|正式墙钟|33.288573秒；资源触发33.168503秒|
|完整音频返回|0/8，全部取消删失；并非八条业务错误响应|
|正式已完成段|安全快照增量188；累计196中含8段暖身|
|全八P50/P95、精确整轮提速比|未知/null|
|停止时card HBM used/free|58202/7334MiB；低于8192MiB余量线|
|服务关闭|normalexit0、无forced，TERM至父退出58.728624秒|

顺序加载观测约205.2秒、八次真实短clip暖身34.679秒、正式处理33.289秒、关闭58.729秒分别计时，加载/暖身不混入API处理目标。启动600秒只是加载期限，正式目标200、硬截止336、无进展监控及HBM8GiB未放宽；未启用缺乏可信回放证据的ETA早停。

用户在上一批M6资源停止、M8完整未进入后，明确追加八实例八完整音频测试。本轮作为独立授权，覆盖旧条件分支的不进入决定，保留当时结果和旧M2准入偏差。仅复用已冻结且通过42项路由及28项控制器目标CPU的MN副本；本轮没有新增业务代码或CPU压力测试。控制器重验原八SHA，真实加载/VAD/0.3秒过滤/裁剪/IPC/Qwen/ITN，没有复用缓存VAD或预制段。

## 时序决定了本轮无法给稳态吞吐

稀疏状态在20.963秒仍无正式活跃段，28.895秒只观察到3条活动音频有VAD段数，32.854秒才观察到8条。完整保留的prefill事件与计数匹配：各owner首次prefill输入host起点由安全事件减已有包络回推为29.870/29.922/29.860/30.057/26.925/28.308/22.905/20.976秒；快照在32～33秒。因此各owner真实生成观测仅约2～12秒，正式前部包含整音频准备/VAD供给过程。

这些host起点不是精确VAD结束或纯NPU forward时刻，稀疏状态也不是等间距全量trace。188/33不能当稳态吞吐、不能外推八文件总耗时；各音频启动/内容不同，也不能从完成段数差异认定路由不公平。requestTelemetry.completed=8和已完成阶段分位数全属暖身，正式裁剪dispatch等待分布仍未知。

## 已证实的资源限制

八worker最近allocator缓存：暖身allocated31248.4MiB/reserved35120MiB；停止时allocated33012.3MiB（32.24GiB）/reserved43048MiB（42.04GiB）。动态reserved增长7928MiB，而current allocated增长1763.9MiB；停止时reserved-current差额约9.80GiB。reserved包含allocated，不能相加；缓存采样并非严格同时，也不能将各worker个人峰值相加叫同时峰值。

整卡driver暖身used52518MiB、停止used58202MiB，包含既有服务及设备开销。多副本已消耗较多启动余量，动态推理分配/allocator保留继续增长是本轮直接资源问题。KV逻辑1445.4MiB/storage1843.8MiB只涵盖部分活跃KV，解释不了全部临时/工作区/保留空间很正常。高reserved不等于泄漏、不能证明全差额可立即释放复用，达到driver free红线亦不等于OOM。

Driver进程表仅记录7/8 owned：暖身小计32557、停止小计36395MiB；缺worker7 PID3256974。完整八份driver合计未知，不以七份小计或八份allocator替补。parser没有主动8行限制，缺PID已在parsed原始compute_apps层，非身份过滤；没有raw stdout，CLI/driver显示限制或格式原因未定。整卡HBM和八worker allocator仍分别可靠到其采样口径。

## 排队与执行开销的判断

停止前各worker活跃13～16段，合计122；父pipeline活动PCM128、预算waiting0。sender pending总1、最老0.087秒，正式锁等待累计22.204秒/write14.979秒相互重叠。供给路径仍用默认线程池及阻塞Pipe，但本轮末端没有B0的32 pending现象，不能把‘严重发送积压’认定为本轮主因；也不能以末端瞬时低积压证明全程没有裁剪排队。

正式prefill129次/310成员、平均2.403段；decode273步。八owner host prefill/decode累计18.959/21.733秒、KV布局2.251秒和bookkeeping1.939秒彼此重叠，不是单卡墙钟或纯kernel。prefill上限16是上限，滚动完成/补位时小批本身不等于调度错误，但大量小批仍会放大输入准备、调用和KV布局频率。

代码审查确认的待证路径如下，不能从源码存在直接认定耗时占比或所有reserved增长来源：

1. Transformers DynamicLayer.update每层每token用torch.cat扩展K/V，即使布局直接沿用也会分配/复制。
2. 成员变化时先index_select完整时域再裁剪，随后可能pad/cat/permutation；新旧KV/storage view和中间buffer短时并存。
3. 每次prefill/decode后的_refresh_memory逐活跃成员/层扫描storage，未纳入当前phase包络，也没有按1秒推送频率限频。
4. 每个owner先prefill再decode，CPU token回读与安全同步限制流水；单卡八进程并不提供八张卡算力。

具体行号、库源码SHA及可证伪点见m8-static-cause-review.md。显式sync包络不能覆盖forward中的回读等待；瞬时AICore23%也不能作为整轮算力利用率或容量拐点结论。

## 后续定位及边界

若继续评估200秒，应先在真实独立轮补allocator active/inactive_split/alloc_retries等细项和分配来源，区分可回收缓存、非可释放碎片与live张量；同时给_refresh_memory独立计时、记录KV中间复制字节及原八VAD/裁剪/首次prefill阶段。然后用同源真实接口比较有明确依据的缓存/布局实验。不能为获得完整成绩静默放宽8GiB红线，不能未经验证直接把empty_cache/StaticCache当作提速方案。

本轮没有完整输出可做全文质量对比，准确率与具体全文差异未知。源/资产冻结、控制器/准入/收尾审计通过只说明验证有效，不代表性能质量通过。正式业务、配置、依赖、权重均未改；全部九次服务843个出生身份消失，18083空闲至少5秒、私有PCM空，既有1601397保留，卡7恢复约24%占用。原fusion删除stash保留，旧任务20/25、44/60不变。本change20/26保持open，不自动恢复S1/其它矩阵或归档。

原始证据：logs/qwen-200s-feasibility-20261123/m8-formal。摘要与独立核验：m8-run-summary.json、m8-analysis.json/.md、m8-admission-audit.json、m8-controller-audit.json及cleanup-verification.json。
