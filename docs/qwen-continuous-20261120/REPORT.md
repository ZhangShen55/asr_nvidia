## 2026-11-21 最终独立八音频结论

**未通过420秒8/8门槛，候选不推广、change保持open（20/25）。** 用户确认的段级恢复/不可恢复警告已实现，设备故障/取消/超时仍失败。独立HTTP测试与CPU压力隔离，业务源码hash冻结通过；正式缺省仍legacy48/384。

|口径|最终clean3ms真实v3结果|
|---|---|
|处理完成/严格无警告成功|2/8，均25%|
|完成请求|a5 389.080083秒；a8 411.050558秒|
|未完成|6条在420秒闸门删失，不能记为终态业务失败或质量成功|
|P50/P95|400.065321/409.952034秒，仅完成子集n=2|
|整轮墙钟|420.959650秒，未达标|
|警告|两条完整返回0；worker闸门快照4：3丢弃不可恢复、1恢复前缀|
|关闭|normalexit0，无强制回收，54.699801秒；全部登记身份退出，端口空闲5秒，既有服务保留|

闸门安全步快照（含warmup可能，不能等同正式所有段最终计数）：已完成3961（EOS3957/复读4）、活跃91、实际活跃峰98、逻辑窗口峰128。prefill 876次/217.312秒，decode951次/150.194秒，4052个已接纳成员、平均prefill4.626段。接纳3ms实际poll累计1.143秒，说明不能把所有耗时简单归于等待或“128并发过高”；频繁prefill/KV重排与供给等仍需同源分解验证。独立128段3ms 7.99秒 vs0ms7.84秒没有提速，不能以CPU模拟运输改善推出NPU吞吐提升。

相对文本：旧16/128与新轮仅a8双方完整，旧353/新354段；56处同边界文本变化、0处纯字段变化、1个新独有段。存在书名等明显措辞变化，不能从完成率推准确率。7条缺完整配对（其中新6条未完成），diff标null而非0。完整可搜索原文见full-v3-differences-drain3-clean.html，具体示例见同名-summary.md；人工有效对话召回≥98%仍未知。

有效源码隔离回归1135pass/380subtests；保护40.so151pass/17subtests。先前plainCPU用例有真实预加载/共享stats隔离泄漏，已修三fixture；与其并行的观察3ms轮2/8不可作独立性能证据。本轮未并行该类任务。后续独立fatal-release修复只影响异常清理，另列增量验证，不重写此轮源码身份。

证据：http-a8-drain3-clean-summary.json、http-a8-drain3-clean-shutdown.json、cleanup-verification.json、final-npu-smi.txt。原始服务日志/请求/响应/完整快照保留在logs/qwen-continuous-20261121-a8-drain3-clean。旧失败、强制关闭及不隔离观察数据均保留，不改写。可选专用sender尚未采用。

# 单Qwen连续生成阶段报告（初版，尚未验收完成）

## 当前结论

真实NPU已经证明“短段结束后，新段实际执行早于旧长段结束”，所需连续接纳行为成立；但初版实现的128段耗时显著落后静态HF，尚不能认为达到吞吐优化目标。正在进行KV去clone等性能优化，以下数字对应优化前版本，后续复验不得覆盖原始证据。

以下固定段章节记录初版时间点；当时真实8并发尚未执行。文末已补首轮真实HTTP结果：0/8正常完成，未通过。不能把本报告视为change完成。

## 条件与证据

物理NPU:7，原HF Qwen3-ASR-1.7B权重、BF16、Transformers+torch_npu、greedy与max_new_tokens256。在线复读保护关闭，以避免提前停止影响正确性/速度配对。使用replay128.json固定真实VAD PCM；各档每种模式各一轮暖身、一轮稳态，模型加载不计入稳态；不是三轮稳态中位数，不代表统计稳定的P50/P95。共享服务保留，设备竞争条件须连同原始记录审阅。

原始证据：capacity-paired.json、npu-smoke.json、npu-smoke-summary.json；固定输入及校验见replay128.json、baseline-and-scope.json。

## 初版固定段吞吐

| 本轮处理段数 | 静态稳态秒 | 连续稳态秒 | 连续耗时增加 | 静态/连续 段每秒 | 原始文本差异段数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.641165 | 0.957876 | 49.40% | 24.95 / 16.70 | 0 |
| 128 | 4.553295 | 15.263578 | 235.22% | 28.11 / 8.39 | 4 |

这里16档只处理固定前16段，128档处理完整128段，不能将16档0.641秒与先前“128段分8批16”的17秒曲线直接比较。连续128本轮约为静态的3.35倍耗时；该数字尚未定位为纯KV拷贝成本，待去clone复验和分步计时。持续补位的正确行为本身不保证吞吐更高。

## 真实持续补位

三段真实VAD测试中，短段先EOS，新段随后完成prefill，旧长段最终EOS。事件单调时钟顺序为：

- 短段EOS：7254138.60848468
- 新段prefill：7254138.79703807
- 旧长段EOS：7254141.36392988
- EOS至新prefill间隔：0.188553秒


三段文本均与静态输出相同。这是实际模型执行事件，不仅是future入队；仍有约189毫秒补位间隔，未承诺物理零间隙。此小试不替代128容量或八完整音频测试。

## 具体文本差异

初版16段文本16/16相同；128段124/128相同、4段不同，来源边界与全文见DIFFERENCES-initial.md，可搜索并逐字高亮见differences-initial.html，机器记录见differences-initial.json。

差异包括中医词“数时/宿食”、法语末尾“hein./mais”、逗号变化，以及index123较明显的措辞变化（“商业运作”与“少油”等）。不能只按标点差异处理，需主Agent听音复核显著变化；没有人工真值时不能判哪一种准确率更高。静态容量记录只有文本，无静态生成token，不能声称这轮token一致。两种路径均未通过降低生成上限或新增过滤规则获得速度。

## CPU与资源回收

主Agent完整CPU回归：1060 passed、3 skipped、287 subtests，约322秒；原日志/tmp/qc-regression/run.log。复读/差异/重放组件此前独立30项通过，详见repetition-validation.md。单元测试不能替代真实端到端验收。

capacity-paired.json的四个owned worker（1657228、1874297、2086220、2353144）均记录normal_exit_ok=true、forced_cleanup=false、worker_alive=false；smoke同样正常关闭，最终active_sequences=0、kv_bytes=0。未启动HTTP监听，所以此阶段没有“完整服务端口释放”的验收结论。主Agent后续测试仍须登记新PID出生身份并finally回收。

## 历史比较边界与剩余工作

旧真实HTTP16/128曾在420秒内2/8完成；成功子集P50约350.75秒、P95约365.27秒，并不是全八请求的分位数。新固定128段的结果不能与完整HTTP直接计算端到端提速比，也不能宣称已解释旧服务满128批约10秒与独立约4.60秒的差距。

待完成：去clone性能复验、动态KV/持续生成质量复核、在线复读真实关开配对、服务集成/故障和关闭验证，以及原八完整WAV真实v3八并发16/128。最终须报告420秒内完成数、成功率、明确样本数的P50/P95、总墙钟、HBM和具体文本差异，未过门槛则保持change open。


## 首轮真实v3八音频16/128（未通过）

已按原八完整WAV、A纯转写、并发8启动真实服务测试；服务完成暖身后发起正式请求。原始目录`logs/qwen-continuous-20261120-a8/`，精简证据`http-a8-initial-summary.json`；未复制巨量response正文，原日志保留。

本轮 **0/8正常完成，成功率0%**。a2约183.695626秒、a4约199.955456秒返回HTTP200但业务4012，属于提前终态失败；其它六条在420秒仍未完成，随后因诊断闸门被客户端取消。停止与取证后记录的全批墙钟为421.329564秒，不能称八音频完成耗时。正常成功样本n=0，**P50/P95不可定义**；两个提前失败不能填作超过420秒的删失。

状态采样173条：全局逻辑在途峰值121，worker真实活跃序列峰值109，采样看到的decode步最大成员数109，逻辑KV峰值1249411072字节；这些分别是逻辑窗口、实际活跃状态和物理步观测，不能混称128真实并行。

最后worker采样累计prefill 124.741180秒/554步，decode 260.798329秒/592步；结束计数{"eos": 1786, "cancelled": 13}。计数从worker启动累计，可能包含暖身，且最后采样在排空之前，不能当最终已完成段总数或把共享步按每段重复求和。

目前捕获的结束原因只有eos/cancelled，不能确证两个4012的首因，更不能归因于复读保护。首因、取消传播及是否存在结束原因遥测缺口仍需定位。动态packing与小prefill碎片的成本需要后续bundle优化配对，不以猜测代替结论。

已补`shutdown.json`证据副本`http-a8-initial-cleanup.json`：forced=true、exit_code=-9、passed=false，TERM到父进程退出约210.252秒；最终remaining_pids=[]且端口连续5秒无监听，owned资源已退出，但正常关闭验收失败，不能将“资源回收”冒充优雅关闭通过。 后续bundle补丁会另轮复验，本轮失败记录不得覆盖；尚不满足420秒八音频性能门槛，不推广、不归档。

## 后续优化与真实异常重放（2026-11-21）

- 只读KV视图+快照节流：128段稳态连续11.6637秒，同期静态4.446秒；仍未胜出。
- 进一步复用整批DynamicCache，成员变化时按组重排且裁剪物理宽度，CPU覆盖30代插入取消无无界宽度增长；真实128段稳态连续7.623秒，同期静态4.431秒。两轮均全部EOS、正常关闭，仍有4条文本差异，具体值以capacity-paired-optimized.json/capacity-paired-bundle.json为准。
- 当前已修复HTTP事件循环同步Pipe写入、取消ACK覆盖request_timeout首因；尚未以新版本重做完整八音频，不能把首轮0/8归为已修复。
- 两个首轮失败段已按原VAD重新裁剪（a2零基8：64.08–72.09秒；a4零基29：148.49–156.50秒）。单独NPU重放：静态HF约16.08秒返回循环文本；连续保护关约16.40秒、开约15.69秒，均两段达到256token、length_limit。新保护在这两真实循环中未提前触发，不能宣称已解决复读时延。原始结果见failed-segments-replay.json，纯CPU准备来源见/tmp/qc-failed-segments/manifest.json。
- 因当前阈值要求正文超出时长×40或120字符的异常长度，256token上限可能先到；真实循环检测标定仍未通过。
- 小规模真实HTTP（最新bundle+非阻塞IO版）单短WAV成功，端到端约4.595秒；启动就绪约36.91秒。正常关闭exit0、未强制，TERM到退出约26.86秒、端口观察5秒通过；这不替代八长音频中止的正常关闭验收。

## 设计决策待明确

当前change明文规定repetition_stop/非EOS长度耗尽按4012使完整音频失败。已实测两个输入段会达到此条件，因此不改变异常段结果策略就无法达到该语料8/8成功。已请求用户明确：维持严格整音频失败，还是允许段级复读修复/不可信段显式异常并继续其余段（需修订成功口径与规格，不静默吞段）。现有证据不支持宣称质量验收或420秒性能通过；候选不推广，change保持open。

## 最终回归与资源检查

1080 passed, 3 skipped, 2 warnings, 356 subtests passed in 326.34s (0:05:26)。运行使用独立TMPDIR、CANN环境、TORCH_DEVICE_BACKEND_AUTOLOAD=0，仅CPU回归。

最终核验128个owned PID出生身份无存活，18083连续6次为空闲；共享uvicorn1601397保持，见cleanup-verification.json。初轮完整八音频的强制关闭失败仍保留，不能由后续小规模exit0覆盖。当前任务16/25，尚未质量/性能完成，不归档、不推广根配置。

## 用户确认后的段级恢复（2026-11-21）

已按用户选择更改为三类段生成异常保守修复/无法恢复警告并继续其它段；设备故障、超时/取消仍失败。continuous API新增processing_status和segment_warnings，全部无法恢复也不能伪装4008或无警告成功。完成率、严格无警告成功率和可用文本请求数分列。旧0/8不改写。

两段真实8.01秒PCM再次NPU验证：静态16.1786秒，continuous保护关闭16.6117秒（两段length_limit告警），保护开启13.0934秒（两段repetition_stop告警），约节省21.18%墙钟；未恢复可信文本，均明确discard_unrecoverable。只代表两段，不能外推整堂课；详细token、原文与资源见anomaly-warning-npu-replay.json/summary。

强循环检测的896条既有真实token回放为0触发，但无人工参考，不能称召回≥98%或零误杀。质量门槛限制见recovery-quality-evidence.md；候选不推广。


## 段级warning后的真实8并发（3ms运输补位改动之前）

原始目录`logs/qwen-continuous-20261121-a8-warning/`，精简结果`http-a8-warning-before-transport-summary.json`。同八完整源、A模式、已暖身、单Qwen16/128，测试源码身份检查完整且运行中无变更。

本轮仍为 **0/8处理完成、0/8严格正常成功、0条已返回带警告完整响应**；八条均在420秒闸门时未完成，随后客户端取消。闸门前未再观察到上一轮的两个fatal业务4012；worker累计有3段repetition_stop、action均discard_unrecoverable，说明异常已按段警告继续，但不能把三个丢弃说成经人工确认无有效语音。HTTP summary的warning_segments=0只统计已返回响应，不能据此否认worker三段警告。全批取证停止墙钟420.675553秒，P50/P95因完成n=0不可定义，不能将取消时刻当完成时长。

闸门前最后有效worker采样completed=2970（EOS=2967、warning=3）、active=45；累计prefill 220.949768秒/1110步，decode 153.822638秒/1165步，repacks=1118、cache_reuses=47。逻辑窗口峰值128、worker真实活跃序列峰值76，观测decode最大成员75。这些是worker自启动累计/采样状态，可能含暖身，不是420秒后排空终值。

此版本没有transport.prefill_members直接计数。仅按(completed+active)/prefill_steps=(2970+45)/1110估算平均prefill约2.716成员，依赖无未计取消/失败prefill等假设，不能当精确测量或反推每批分布。

HBM须分口径：驱动采样全卡7峰值41240MB（含既有服务），owned Qwen worker驱动观测峰值25389MB；worker缓存allocator max_allocated=11082.99072265625MB、max_reserved=23694.0MB，后者peak_scope含加载/未重置且不一定实时，不能直接当驱动进程HBM。

本轮正常关闭已通过：exit0、forced=false、remaining_pids=[]、端口连续5秒无监听；TERM至父进程退出206.872060秒。虽接近关闭210秒宽限，不能与旧轮强制退出混淆，也不能把关闭耗时计作请求完成时间。

后续运输3ms候选尚未完成正式八服务验收。`transport-npu-paired.json`的burst128一次稳态：0ms为7.840453秒、3ms为7.989917秒，3ms本轮没有提速；两档均128正常EOS，但有11条模型输出差异，不能说质量等价。首轮暖态10.101/10.333秒单列，不冒称三轮中位。CPU运输测试prefill调用42→9等现象只能说明CPU调度行为，不能作为NPU吞吐提升证据。420秒内八音频完成目标仍未通过。

## 隔离审计更正与最终有效CPU证据

此前未使用完整guard的source全量回归不能标为纯CPU：StatusContract及其它生命周期测试在根preload=true配置下遗漏预加载mock，可能真实加载Qwen权重；状态测试也曾读取共享asr_stats。并行3ms整音频轮次（logs/qwen-continuous-20261121-a8-drain3）因此仅保留观察值，不能当干净单Qwen性能对照。已修三个测试fixture，原断言不放宽。之前1080等未guard回归仅是历史功能结果，设备隔离资格被本审计更正。

真正guarded source全量：1135 passed、2保护专属skip、1既定硬件probe排除、380subtests，242.22秒；targeted88通过。188条有效组合guard启动SHA一致，8次npu-smi在启动前拦截，真实weights/CANN/device执行及共享cache写删0，source_changes和owned残留为空。guard调试首轮9项失败是夹具/runner隔离问题；其重复误启动导致原日志不完整，已如实标注并保留观察摘要，不能冒充完整原始日志。最终r2独立目录报告/原始日志完整。

保护版本最终40.so，151测试+17subtests通过；最终.so真实NPU三段文本与静态一致、旧长段未结束前新段进入，正常关闭。source真实NPU非法语种、运行中取消及注入owned worker故障验证通过；注入故障的强制回收/TBE及resource_tracker告警单列，不当正常关闭成功。

当前正在独立运行clean3ms八音频，不与任何本轮CPU回归/其它NPU测试并行：logs/qwen-continuous-20261121-a8-drain3-clean。最终结论以后续该轮及必要优化后的有效结果为准。

最终清理增量：fatal-release修复避免worker死亡/已关闭Pipe清理盖住原4011/4012，仅所有Future已终态才清去重。源码完整r4隔离目标13pass；单模块重编后实际保护.so增量31pass（含RPC/shutdown/timeout），设备/权重/cache写删0、owned构建/CPU进程归零。业务源码hash无漂移；文档及profile注释变化、TOML值相等已审计。该微小清理修复后没有重新跑NPU或八音频性能，前述HTTP数据对应不可变run源码身份。证据fatal-release-source-cpu.json/fatal-release-protected-audit.json/fatal-release-protected-cpu.json。严格change校验1/1、独立临时合并预览主规格9/9；根主规格未同步、未归档。
