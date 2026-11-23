# M8静态成因审查（只读，结合M6已有结果）

## 范围与证据等级

仅阅读正在运行的MN副本及本机已安装Transformers源码、既有M6/B0摘要；没有修改代码、运行测试/NPU/HTTP，也未读取大status.jsonl。此审查不包含M8正式轮结果，不给预计完整时长或速度比。

- **C：代码事实**——可由下列精确行号及末尾SHA复核，表示该路径确实存在。
- **O：已有观测**——来自M6/B0已完成的资源中止轮；host包络与allocator缓存都有采样/重叠边界。
- **H：待证归因**——路径可能造成成本，但本审查不能量化其占比，需可证伪的独立测量。

M6正式轮49.362秒因free7738MiB低于8192MiB停止，0/8完整请求；扣暖身后831段完成。六owner host prefill累计63.626秒、decode66.801秒是重叠区间，不能加成单卡墙钟。下述观测均受该边界限制。

## 1. “直接复用布局”路径仍会逐token扩展Dense KV

**C：**`core/qwen_continuous_runtime.py:154–161`在成员/顺序完全相同时沿用DynamicCache对象；但安装库`transformers/cache_utils.py:149–168`的`DynamicLayer.update`对keys和values分别执行`torch.cat(..., dim=-2)`。因此“直接沿用布局”仍然有逐层、逐token的新缓存分配/复制，并非固定槽位原地写入。

**H：**不同序列长度及成员数使KV分配尺寸持续变化，旧buffer与新buffer短时并存，可能推高allocator保留及复制带宽成本。它是候选原因，不能从高reserved直接认定泄漏。

**可证伪点：**后续独立profile若显示缓存cat/分配事件占用极小、reserved增长主要出自audio encoder/workspace，则应降低此路径优先级。若cat事件与序列长度/步数强相关，再比较兼容的预分配布局；不能在当前M8轮热改StaticCache或假定其支持独立位置。

## 2. 成员变化时，同一轮可能多次复制KV；先选行再裁时域会放大临时量

**C：**`runtime.py:184–209`每个group先创建NPU row索引，再对每层K/V执行`index_select`，之后才`[:, :, -group_width:]`裁剪。若组底层tensor比选中成员的有效时域更宽，当前顺序会先复制完整历史宽度，再保留尾部view。随后可能执行`F.pad`、跨group `torch.cat`以及permutation `index_select`，最后还进入模型的DynamicCache cat。

即便选择的是某prefill组的全部行，该代码也不检查identity选择，仍走index_select复制；`flattened.index(key)`是Python O(B²)重排索引计算。B≤16时O(B²)本身预计比大tensor操作更小，不能笼统视为大瓶颈。

**O：**M6正式各owner KV布局host包络相加7.925秒；B0单owner正式布局4.702秒。两者不同执行长度、CPU预算和重叠关系，不能直接比较为速度提升/下降。

**可证伪点：**先统计group原宽/裁后宽、选择行是否identity、pad/cat/permutation字节和次数；再在独立副本做“先裁时域后选行/identity选行省略”的逐token等价验证。若实际group宽相同、很少pad或permutation、布局包络低，则不会有预期收益。

## 3. 每段KV是完整批storage的view；新旧缓存短时并存会被逻辑KV字节掩盖

**C：**`runtime.py:112–120`和`224–233`把每成员KV切成view，同时保留`seq.group`完整DynamicCache。Decode重排期间，`self._bundle`及旧sequence引用旧group，`chunks/packed`持有新中间缓存；到forward返回/成员更新后旧引用才逐步释放。`_finish_impl:305`删除active成员，不会立即收缩其它view所引用的整批storage。

这不是“不同VAD共享上下文”；每段逻辑状态仍独立。这里关注的是view持有的底层storage及临时生命周期。

**C：**`_refresh_memory:354–366`只扫描active序列当前KV view，并按storage指针去重；它不覆盖prefill输入、音频encoder/attention工作区、重排瞬间所有中间tensor或allocator闲置块。它也不能完整代表某一瞬间所有group/_bundle临时引用。

**O：**M6暖身后cached allocated总计23418.2MiB，正式cached reserved总计38920MiB，动态reserved比暖身增加12580MiB。逻辑KV和storage只是部分live数据，无法解释全部保留量很正常；仍需分配事件证据才能区分工作区、可回收缓存和碎片。

**可证伪点：**在独立profile里核对活跃tensor的storage生命周期及峰值分配来源；若主要峰来自encoder临时buffer，而KV group仅少量storage，则应停止把显存增长归于KV视图。

## 4. 有一段未被现有prefill/decode计时包络覆盖的逐步诊断扫描

**C：**`runtime.py:137/243`每次prefill/decode后都调用`_refresh_memory`；该函数对每个活跃成员、每层K/V两次遍历，并执行`untyped_storage/data_ptr/nbytes/str(device)`。而phase计时在`135/241`已提交，host subphase也不包住该扫描；快照虽然约1秒推送一次，这个扫描并没有按1秒限频。

**H：**这是每步 O(active×层数) 的CPU诊断成本，可能拖慢补位/发起下一步，却不会出现在现有prefill/decode总包络内。源码证明存在，不证明它占主导。

**可证伪点：**后续仅给这个函数加独立轻量包络/调用计数，或在独立对照中限频而保留安全计数；若总计不足实际墙钟的很小比例，则不应重构它。当前轮不能热改关闭诊断后与旧轮混为单因素对照。

## 5. 一个新小批prefill会先于旧序列下一步decode；多模型不自动形成大prefill

**C：**`process.py:69–118`接收submit后最多3ms固定交接窗口；`141–162`有waiting就进行一次最多16成员prefill，之后`163–177`才decode一次。不存在完整generate整批返回屏障，但CPU前端+prefill+readback会暂停这个owner的旧序列继续解码。

**C：**`qwen_asr_runtime.py:117–121`每个成员分别做FP32前端，再collate；`52–106`多成员输入按最长text/features补齐，随后`runtime.py:99–109`搬到NPU并执行prefill。不能为效率合并PCM或改变single尾部特征契约。

**O：**M6正式prefill477次/948成员，平均1.987段；host processor包络跨owner累计9.457秒、prefill forward/readback累计52.706秒。M8各音频独立owner也可能更碎，但本审查不推测它的实际结果。

**可证伪点：**从M8已经记录的ready接收burst、实际prefill成员数和decode间隔检查是否仍大量1/2成员；若平均成员明显增加、decode间隔并未受prefill影响，则该解释对M8不成立。只延长3ms无法由这些数据证明有效。

## 6. MN仍使用共享默认线程池和每worker阻塞Pipe锁，并未包含S1供给隔离

**C：**`api/routes/asr_qwen_common.py:688–703`先占每音频/global credit，再提交裁剪；`632/639`裁剪/编码通过默认`run_in_thread`；`core/async_utils.py:30`底层为asyncio.to_thread。`qwen_asr_client.py:415–417`提交同样to_thread，`process.py:313–334`持每worker send_lock执行可能阻塞的Pipe写。N模型有N个锁，仍共享父进程默认执行器。

**O：**B0正式快照有32 pending send，累计锁等待1903秒（多线程重叠）和write67秒。M6存在更多Pipe并不证明等待已消失，需要看其各sender counters。逻辑16在途也包含等待裁剪/发送，不能都叫正在NPU计算。

**可证伪点：**核对M8父sender pending/oldest、lock/write分量和裁剪dispatch等待。若这些都低、模型active持续接近16，则供给不是当前瓶颈；若仍32线程长期pending，S1独立隔离才是相应假设对照，而不能说增加进程已解决供给问题。

## 7. 现有读回与同步边界约束了CPU/设备流水，但显式sync低并不代表无等待

**C：**`runtime.py:100`读取长度到CPU，`109/221`每个模型步都把sampled token读回CPU以判EOS/复读；`131/237`随后仍显式device synchronize。`_finish`又在CPU执行token检查，异常保护每4 token批解码。当前安全补位在这些边界之后，不能直接删除同步来提前复用KV。

**O：**M6显式sync包络相加仅0.255秒，forward/readback相加103.459秒；后者含设备执行、排队与读回等待，不是纯kernel，也不等于全部可消除等待。host区间重叠不能证明多个进程的kernel实际同时执行。

**可证伪点：**未来需要NPU事件/trace区分kernel、队列等待和readback，且先证明异步生命周期正确；若硬件本身持续忙且等待来源为同卡争用，多进程并不会线性提高总能力。

## 8. M8的帧预算会让某些16段prefill拆成15+1，属于预先声明的实验约束

**C：**MN每单音频owner帧预算14400；`process.py:104`按ceil(samples/160)+100估算，8.01秒段为901帧。`147`用最大帧×成员数限制prefill，所以16×901=14416超过14400，不能一次放入16个最长成员。这不改变音频边界/丢段，但会增加一次prefill机会和形状变化。

**可证伪点：**检查M8实际prefill是否出现15+1及estimated frame分布；若就绪成员始终少于15，预算不是小批的直接原因。不能仅因配置prefill16便声称实际16成员运行，也不能在看到结果后静默增加预算。

## 因果结论及后续证据优先级

目前最硬的事实是：多模型重复常驻成本已存在，持续真实音频又产生动态保留增长；KV dense增长/重排、不同形状前端/模型工作区、共享供给等待都有明确代码路径。M6的0完整响应和资源中止只说明当前资源余量协议未通过，不证明某一条路径是全部原因，更不能说“模型越多越快”或“硬件绝不可能200秒”。

若M8正式轮仍慢/资源中止，建议先从现有记录给出各owner实际完成数、ready/active/批大小、host包络和allocated/reserved；下一次经用户决定的定位工作优先补_refresh_memory包络、KV中间复制字节/分配源、sender与裁剪队列相关性。不要直接展开PagedAttention、静态KV或其它正式重构。重复WAV decode在接收和prefill各一次（`process.py:104`与`runtime.py:92`）也是代码事实，但B0已测audio decode仅0.229秒，当前优先级低于上述路径。

## 本次阅读的源码身份

- `/tmp/qwen200-20261123-j9r2czoi/mn/core/qwen_continuous_runtime.py`: `732ad8f0ed7f7f06250c0b294d33486a2184cda14b3f062823e43a990998dba7`
- `/tmp/qwen200-20261123-j9r2czoi/mn/core/qwen_continuous_process.py`: `23999110ce85be9d90d4afcf421dec3e3d869cc3b1eea0ca56d5fdf2ca5b3a9c`
- `/tmp/qwen200-20261123-j9r2czoi/mn/core/qwen_asr_runtime.py`: `120924885e4ed8cb11947edf77911a57aa8ed46775e54d9883759f9ad3797abe`
- `/tmp/qwen200-20261123-j9r2czoi/mn/core/qwen_asr_client.py`: `1deeb4ed4c6892d852222b934ead8cd1316744de24e127f5979a43624fbffc1e`
- `/tmp/qwen200-20261123-j9r2czoi/mn/api/routes/asr_qwen_common.py`: `4be388d03cf4c546f0b3119caa9451e95c91676110199e98f1becfd1e366db3e`
- `/tmp/qwen200-20261123-j9r2czoi/mn/core/async_utils.py`: `bbf61919dc7a5b986450c9b00ffed7df0980c99e94c5f16c4f2dca1f35abe5c6`
- `/home/zhangs/conda-envs/asr/lib/python3.10/site-packages/transformers/cache_utils.py`: `16bbbba2394c0151af53009800e5a4052479417a8c7e5edfbd5ef128a46fcbc1`
