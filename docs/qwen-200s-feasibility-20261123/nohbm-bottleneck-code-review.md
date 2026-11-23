# 6N6-noHBM阻塞点代码审阅（只读）

## 范围与结论边界

本次仅阅读`/tmp/qwen200-20261123-j9r2czoi/m6n6-nohbm`源码和已保存的`nohbm-analysis.json/md`、`nohbm-bottleneck-worker.json/md`。没有修改源码、执行CPU测试、NPU、HTTP或新重放。以下“优化”均为后续可验证方向，不是已实施或已测得的提速。

本轮是六模型、六完整音频、各16/global96，一张NPU7；194.809秒6/6处理完成，严格无warning3/6，16个warning中13段无法恢复。它不是原八音频200秒验收。HBM停止线关闭不是HBM观测或缓存统计被关闭，更不是生成算法变化；最低已观测free6323MiB及共享占用限制了向八模型外推。

已知慢请求是a3（194.531秒）和a4（189.750秒）。只优化a6末safe到HTTP的10.359秒包络不会直接缩短当前整轮；只单独大幅加速a3，a4也会成为新的末请求，其他不变时整轮最多先缩约5秒。需要共同降低关键owner的长链路，而非把六份累计秒数相加。

## 已有证据支持的成本排序

worker统计已扣除各自一次暖身。a3已计时prefill/decode73.139/72.258秒，其中forward+readback合121.953秒，约83.7%；processor5.920秒、KVlayout11.322秒、bookkeeping4.581秒。a4对应forward+readback128.118秒，占约84.7%，KVlayout10.057秒。不同worker同时执行，六份包络累计不是194.809秒的互斥分解；forward+readback含提交、设备排队、kernel及同步读回，不等于纯模型kernel。

3752个正式段共2138次prefill（平均1.755成员）、4297次decode，3151次repack/1146次直接布局沿用。不同段没有内容关联，不该共享内容KV；布局沿用次数少不等于同段没有使用KV。

### P1：整音频后处理直接占用唯一async事件循环——路径已确认，跨请求损失待量化

位置：`api/routes/asr_qwen_common.py:883-907`在async `run_qwen_asr`中直接调用`normalize_segments`、`sanitize_segments`、逐段语速、`build_speed_info`、语言比例及字段清理，中间没有await。单uvicorn worker下，这一段运行期间事件循环不能调度其它请求的裁剪提交、future回传、段补位及HTTP状态处理。root已从本轮日志核对a3 ITN包络4.642秒、a4约3.227秒。

可验证方向：把整个纯文本后处理单元放入独立有界执行器，初版仅一个worker，所有ITN normalize调用经过同一owner；当前八/六音频准入为队列容量自然上界。不要简单换成原`run_in_thread`，因为那仍共享已可能被Pipe发送占用的默认pool。输入使用请求独占段字典/配置快照，工作返回后由主loop发布结果及阶段统计；取消仍等待真实线程结束，生命周期统一关闭。

线程安全边界：`utils/itn.py:87,131-141`的`_CACHE_LOCK`只保护normalizer构造/缓存出版；`normalize_segments:239`调用共享实例`.normalize()`时没有调用锁。这不证明WeText/Pynini对象可多线程重入。因此优先独立单worker保持调用串行；并行多线程或每线程normalizer是另一方案，需要单独验证实例状态、构造成本和内存。不能因“cache有锁”直接开多worker。单线程转移也不能预先保证消除GIL影响；FST库是否释放GIL需要实际loop-lag证据。若GIL仍卡住loop，独立进程可研究，但复制FST/IPC成本更大，不能在本次只读分析里许诺收益。

验证指标：ITN期间event-loop lag/P95/最大值、其它音频completion→下一段prefill间隔、GET状态延迟、后处理queue-wait/compute、各请求末模型safe→HTTP间隔和整轮墙钟。a3本身4.642秒是现有后处理成本，不是可保证删掉的4.642秒；移到线程的第一目标是让其它音频继续补位，影响可能大于或小于单条尾部，不能把六条ITN累计全部算成可节省墙钟。

质量要求：同一输入文本/语言/ITN开关/数字和语气词策略、标点卫生、语速/语言比例、字段顺序及时间边界逐项一致；passthrough/异常行为不变；没有normalizer实例交叉污染，输出未完成前不提前返回或丢弃。

### P1：prefill/decode交替及大量小调用——已量到频次，策略收益未验证

位置：`core/qwen_continuous_process.py:63-177`每轮先收命令、至多一个prefill chunk，再推进一次decode；新到段只在安全边界接纳。上限prefill16，全轮实际平均1.755说明固定输入处理、forward启动/读回和context切换发生得频繁。每个model只拥有同一完整音频的16槽位，短段不断结束后仍会有成员变动；这不是旧的整段batch返回屏障。

可验证方向：先使已获准槽位的ready供给连续、合并已经ready的成员，研究按可控耗时预算交替prefill/decode。不得用延迟很久凑满16恢复整批屏障，也不能让连续新prefill饿死旧decode；改prefill chunk或短接纳窗必须单因素。当前3ms只是收到submit后的固定总交接窗，并非每段都等100ms；不能把所有时间归给这3ms。

验证指标：prefill批直方图、每段ready→prefill、连续decode间隔、每步成员/token数、forward+readback包络与真实HTTP末请求；同时保持每音频16/global96及生成/过滤参数不变。主成本是forward+readback，所以减少碎小调用有潜力，但不意味着批越大越好，更不能从代码存在该策略推出必然提速。

### P1/P2：默认executor与每worker Pipe锁——明确共享路径，当前贡献尚未分离

位置：`core/async_utils.py:30-32`的`run_in_thread`使用`asyncio.to_thread`；API `_cut:632,639`在同一默认executor裁剪/编码；`core/qwen_asr_client.py:425-427`同样用默认executor运行`runtime.submit`；`core/qwen_continuous_process.py:311-334`的submit持有该worker `_send_lock`完成可能阻塞的Pipe写。多模型有六把锁，并不是六个互不干扰的父进程默认pool。cancel/release也经过相应发送锁。

本轮sender_pending采样峰31说明存在未发送完的提交，但不提供等待分布、executor满载证据或因果结论；它不是31个正在NPU生成的段。NoHBM候选没有包含S1专用sender/裁剪pool，不能把194.81秒解释为S1已生效。

可验证方向：独立有界裁剪pool与每owner一个专用有界FIFO发送者，避免默认pool中大量线程等待同一Pipe锁；保留数据/控制预留、同key顺序、稳定去重、capacity_rejected未接收语义。release每完整音频仅一次，可保留发送完成ticket；关闭/abort必须join非daemon sender。该方向已有其它隔离候选的CPU契约资产，但不是本6N6实例配置的NPU性能证明。

验证指标：default executor实际max_workers/排队深度、cut queue/compute、submit lock/wire各自累计与分布、pending_send年龄、活动逻辑槽→真实prefill/decode成员、完成到补位空窗；再看a3/a4及全轮。若队列等待实际很小，该优化收益也可能小。

### P2：KV dense布局整理——已有每owner约5～11秒host成本，需控制改造风险

位置：`core/qwen_continuous_runtime.py:154-210`只有整个活跃key tuple相同时才能直接沿用bundle；成员改变时按group执行多层`index_select`、`pad`、`cat`及重排，构造dense DynamicCache。逐段生命周期本来就会改变成员；需要减少的是布局变化成本，不是跨VAD内容缓存。

具体候选从小到大：在group处理前裁掉无效历史padding再select（需验证NPU处理非连续view成本）；缓存row/permutation等小索引并避免`flattened.index`的二次查找；分组稳定cache/固定槽位或增量更新，减少全层全成员重打包。最后一项接近缓存布局重做，风险和投入大，应先证明本轮模型/共享forward主成本中有多少是拷贝带来的，而非先做完整PagedAttention替换。

验证指标：per-owner layout host包络、设备copy事件、allocated/reserved峰、缓存物理宽度、重排次数及对关键请求forward+readback的影响。当前host layout计时不包括全部异步copy完成成本，后续forward读回可能承接这些成本；也不能只因layout包络不大就认定设备复制免费。反过来，不能把整个forward包络算为KV复制。

另核对锁定环境的transformers/cache_utils.py:149-167，DynamicLayer.update本身每token对keys/values执行torch.cat。这发生在forward内部，即使外部bundle直接沿用也不是零复制；现有forward+readback包络包含这种缓存增长成本。固定槽位/预分配如果研究，需同时考虑这一层，但不能据cat存在直接断言其占了多少kernel时间。

质量要求：CPU/logits/greedy与真实混长NPU位置、mask、左padding、cache_position和有效特征一致；EOS/取消后slot不能先复用，旧KV不串新段；30代以上轮换缓存宽度仍有界。本轮已出现具体措辞变化及13无法恢复段，后端布局改变必须逐段对照和复核，不能以少输出获得速度。

### P2：每步_refresh_memory是诊断盲区，先测而非认定主因

位置：runtime `prefill:137`、`decode:243`在phase_wall_s累计完成后调用`_refresh_memory:354-366`。每次扫描活跃序列×层×K/V的numel/element_size/untyped_storage/data_ptr并建字典；本轮仅prefill+decode即至少6435次调用，且这部分不在已计时prefill/decode及experimental子阶段中。它的确是未覆盖host路径，但尚无独立耗时。

可验证方向：给它补计时；若显著，再把物理storage去重降频到安全快照采样或按group增量记账，逻辑KV字节/活跃峰按成员事件更新。不能把真实资源预算/停止条件也停掉，也不能用较稀疏采样伪造更低峰值。缓存指标须标采样年龄和峰值统计范围。

验证指标：refresh单次/P95/累计host秒、step间未归属间隙、采样开销和driver/allocator峰的交叉误差。即使有余量，也不能直接将a3模型运行以外的剩余时间全部归于它；还包含receive/poll、finish、回调及调度。

### P2/P3：必要读回与显式同步应区分，不能无条件删同步

位置：runtime prefill `lengths.sum().tolist():100`、`sampled.tolist():109`以及decode `sampled.tolist():221`与显式`synchronize:129-132,235-238`。同步token读回用于CPU EOS/guard和安全成员变动，属于当前设计的必要环节。

可验证方向：prefill有效长度可尝试在CPU processor结果尚未move之前计算，避免仅为长度产生一次NPU→CPU读取；进一步检查读回之后是否还有可证明冗余的设备级同步，或能否用执行流/事件保持所有权安全。不要直接删掉所有synchronize，也不要断言每进程同步必然阻塞其他所有进程；跨进程设备调度影响需要实际证据。

指标：各类readback/sync包络及设备事件、跨进程重叠、首token/持续decode间隔；取消/报RESULT前设备真实完成和精度契约保持。

### P3：重复WAV解码可消除一次，但现有证据不支持它是最大项

位置：worker admission `process.py:104`调用完整`decode_wav_bytes`只为长度/预算，adapter prefill `runtime.py:92`再次执行相同PCM16→float32解码。该函数在`core/qwen_asr_runtime.py:21-33`会readframes、astype和归一化。scheduler的`audio_frames`只读WAV头，不是第三次完整PCM解码。

可验证方向：在worker接纳处完成一次同样的格式/完整性验证，保存已解码数组及长度给prefill，计入worker有界缓冲生命周期；或者仅在接纳阶段读取可信头部，完整校验移入prefill，但这种方法改变拒绝时机，风险较大。不可把上游原float波形直接代替PCM16往返后的样本，否则改变模型输入。

指标：admission decode新分项与现有prefill_audio_decode、CPU分配/RSS和整轮效果；验证PCM字节/数值、帧/token预算、坏WAV错误、取消回收与输入前端完全一致。processor每owner约4～6秒已计时，重复解码未独立计时，因此先列低成本小优化，不承诺大幅提速。

### P3：上游有界任务数量与后处理汇合

`segment_pipeline.run_bounded_segments`确实限制了协程总数，但API把worker_count设为pending_segment_limit而非16；每个worker随后等待每音频Qwen credit。于是可能有大量等待窗口的元协程。pending budget FIFO同时限制段数和PCM字节，旧状态观测waiting峰0，当前没有证据它在阻塞主路径，不应简单调高PCM预算。

后续可量任务数/loop开销，再研究窗口感知的有限dispatcher。直接把元worker数降成16不一定等价：Qwen credit在文本后处理/增强汇合前释放，若所有元worker仍在这些步骤，下一段可能反而补不进来。需要区分Qwen完成和整段汇合，保留原PCM/异常/取消所有权。

## 建议后续验证次序（本次不执行）

1. 先补关键路径缺项与loop-lag证据；优先确认整音频后处理同步占loop对其他音频的影响，以及_refresh_memory未计时部分。
2. 小范围独立比较“ITN/文本后处理独立单worker”与“供给隔离”，保持六音频、六model、每音频16、global96及相同生成/复读策略，不同时改动后端布局。
3. 若仍由forward+readback主导，再验证现成ready合批/prefill-decode调度；KV整理有足够证据后才投入布局改造。重复解码作为独立低风险分支。
4. 始终使用相同完整六音频真实v3、质量对照和资源闭环；八音频结论需要另外真实验收，不能用6/6×比例外推。16warning、13无法恢复段和人工真值缺失必须与速度一起报告。

只读结果不能证明某一优化能从194.81降到某个具体数字；也不能将关闭HBM停止线解释为已经消除了显存瓶颈或证明八音频容量安全。
