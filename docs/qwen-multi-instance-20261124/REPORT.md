# 正式Qwen多实例实施与真实验收（2026-11-24）

正式代码已支持动态整音频owner租约和逐段连续生成。纯A6在186.32秒内6/6完成，达到约200秒目标；B4双增强4/4完成但需302.60秒。**增强保持驻留后的A6发生真实HBM OOM，最终共驻验收未通过**，change保持开放27/29，不归档、不将当前候选当作长期稳定部署配置。

## 实现与运行配置

根`config.toml`：continuous/process，N=6、npu:7、S=16、owner_threads=1、preload=true，HTTP仍一个进程。audio=N=6、global=N×S=96派生；每owner physical默认S、prefill默认min(16,S)，14400帧/10000估算token预算不随N复制。PCM全服务256段/256MiB；B上限4。原HF权重/BF16/torch_npu、生成上限256和复读阈值保持，在线复读保护true、IPC交接3ms沿用已测参考。

每个HTTP请求以随机request_id租用一份健康owner；每owner段独立EOS、清理和补位，到整音频增强/临时资源安全收尾才归还owner，不按文件名/SHA绑定，不等16段一起返回。顺序预加载有部分失败回滚、健康容量降级、故障隔离和取消安全收尾。状态区分configured/healthy/leased/available和每owner身份。

B类在有效VAD段裁剪后创建Qwen/CAM/emo三个独立任务。Qwen先归还自己的段credit；PCM等三路终态后释放。CAM/emo池各自配置，不复制六份；0.3秒以下丢弃，CAM覆盖有效段，emo只在实际1～10秒闭区间推理，其它有效段按规则“平淡”skip。聚类/ITN和原响应字段保留，新增有界计数/等待/冷加载遥测；同卡任务并发不保证kernel同时执行。

当前CAM process1/emo thread1是实测候选；池数最终标定和共驻容量尚未通过。宿主只用NPU7，既有PID1601397约12417MiB占用保留。Docker逻辑npu:0，N/S/派生预算一致，仍保留原有VAD过滤和CAM thread等环境差异；本轮未构建/部署Docker，不能引用宿主成绩作为其验收。

## 真实HTTP结果

绝对正式`main.py`入口、正式根配置、v3接口，完整原始a1～a6 SHA清单；计时前六owner均真实forward，启动/暖身另列。同一服务顺序A6→A4→B冷加载暖身→B4→真实准入探针→增强驻留A6；不是截取短音频或离线段跑分。下列P50/P95是该小样本完整请求分位数，不能当生产SLA。

| 阶段 | 业务完成 | P50秒 | P95秒 | 墙钟/停止观察秒 | 警告 / 不可恢复丢弃段 |
|---|---:|---:|---:|---:|---:|
| 纯A6，6完整音频 | 6/6 | 173.241736 | 185.420846 | 186.323683 | 13 / 11 |
| 同源A4对照 | 4/4 | 179.744930 | 199.810247 | 200.926860 | 16 / 13 |
| B4，showSpk/showEmotion=true | 4/4 | 289.808711 | 301.368365 | 302.597690 | 11 / 8 |
| 增强驻留后A6 | 3完成、1失败、2取消 | 整批未知 | 整批未知 | 179.952671时停止 | 仅完整子集可观察 |

A6严格无警告4/6，B4为1/4；“完成”包含显式段警告及保守丢弃，不能写成全部内容无损或准确率通过。B冷加载暖身4/4、墙钟42.808571秒，不计入暖态B4。A4没有约200秒目标承诺，不能挑它的快/慢请求和A6合并分位数。

实际准入探针通过：4B后第五B返回4017；再加入2A达到总6；进一步A/B均返回4013。探针使用真实在途请求，结束归零，不用fake hold。A6每owner峰16、全局峰96，段/PCM末值0。

resident失败a4在178.839555秒HTTP200但业务code4012，不能算成功；a2/a3由控制器取消，不能说它们均超过336秒。整批未完成，不用3个完成子集计算“全批P50/P95”。触发是request_failed，**未触发336秒墙钟线，也没有8GiB或其它free-HBM替代停止线**。

## 性能卡点及OOM证据

纯A6准入等待22～24ms、VAD派发等待低于54ms，VAD包络22.44～25.40秒，分段处理85.36～155.73秒。各owner prefill/decode包络占对应segment约91～95%，主要时间在worker前端/生成区间；processor、forward、传输/同步和KV整理尚未独立计时，不能认定其中任一项为唯一瓶颈。ITN尾部最高6.884秒，裁剪派发累计可达15.416秒，值得后续计时，但不能直接当可消除墙钟。见[逐owner阶段报告](a6-phase-analysis.md)。

B4 CAM真实成功2775次，emo成功2226次+549次时长skip=2775。CAM调用包络累计263.255秒、平均94.867ms；跨段累计等待61031.822秒、平均21.993秒。CAM队列P50/P95/max均255；emo队列0/2/23。345次GET中242次PCM活动段达到256，造成供给反压；PCM字节未达到256MiB。证据支持CAM是主要排队点，尚不能把263秒全归为NN kernel时间；CPU fbank、IPC、传输、NN、同步需要后续拆分。并发累计等待不能相加解释302.60秒墙钟。见[B4与内存报告](B4-MEMORY-REPORT.md)。

按独立资源时间戳统计，NPU7最低free原始MB值：A6=6473，A4=2183，B4=406，resident=132。资源采样中位约5秒，存在最长11.8秒缺口，不能把132MB当失败瞬间可分配连续块。

底层[owner5 CANN日志](evidence/owner5-cann-oom.log)明确：PID1027285、dev7，申请23068672 bytes（22MiB）失败，ErrCode207001，driver error:out of memory。共驻失败是已确认OOM，而非仅从4012推断。观察到owner0 reserved从5704→8640→8758MiB增长，idle allocated约3906MiB；各owner跨请求保留allocator容量是明确线索。`reserved-allocated`不能全部视为立即可释放或证明泄漏，现有证据不能认定缓存是唯一根因。

因此按已批准test-plan在真实OOM后停止整批及后续压力，未继续CAM2/4，未擅自清缓存、减少N/S或换推理引擎重跑。后续应先独立验证安全终态的owner缓存生命周期/回收代价，再拆分CAM前端/IPC/NN/同步；这些优化未在本change临时实施。

## 具体输出差异

[旧noHBM同六源→正式A6完整差异HTML](differences/baseline_vs_formal_A6.html) / [JSON](differences/baseline_vs_formal_A6.json)：6/6同SHA完整配对，unknown0，匹配边界文本变化257处，左独有4/右独有8边界，总差异边界269。旧参考是隔离实验，不能当本轮正式性能验收。

[A4→B4完整差异HTML](differences/A4_vs_B4.html) / [JSON](differences/A4_vs_B4.json)：4/4同SHA完整配对，匹配边界文本变化179处；role/emotion各2567处变化符合开关差异，左独有2/右独有8边界。字段变化与转写变化分别保留，可搜索或筛选仅文本变化。resident不完整，不伪造完整配对。

root已对照原始响应核验以下实际内容，记录在[独立核验](evidence/root-cleanup-and-text-audit.json)：

- a1 228.34～229.03秒，旧参考`pests.` → 正式A6 `pesticides.`。
- a1 355.34～361.55秒，旧参考`…join this group?` → 正式A6 `…join this cooking?`。
- a1 414.47～422.48秒，A4 `EAP. So I'm…` → B4 `EAP. So, I'm…`，role null→spk_0，emotion null→平淡。

没有人工听音真值，不能由这些变化证明质量提高或计算WER/CER；有效对话召回≥98%的上线门槛尚未认证。

## 回归、配置一致性与安全退出

完整guard源码`cpu-full-v3`：1289 passed、379 subtests、2保护专属skip、1既定硬件probe排除，250.82秒。保护`cpu-protected-v1`：364 passed、244 subtests，55.59秒；40实际SO导入，11改动模块重编/29未变已验证SO复用，保护模块无源码fallback。日志隔离增量源/保护各84通过，netns增量两边各40通过。无依赖安装、真实设备/权重执行、共享cache修改或源漂移；保留安全只读npu-smi拒绝和CPU loopback记录，不称guard日志全空。初轮失败和修复证据保留，不用最终通过抹去历史。

控制器因共享服务位于容器netns，改为只读/proc/PID/net并按其fd socket inode过滤，确认8007；不改共享网络。正式相对日志路径在独立service-cwd执行，入口和CONFIG_PATH仍为正式绝对路径，避免日志轮转影响项目历史。运行期间源码/配置/控制器完整性全部通过；测试后仅增加配置注释，[TOML解析审计](evidence/final-config-semantic-audit.json)证明根配置语义与实测快照相等。

最终`shutdown.json`：exit0，forced=false，orphan_cleanup=false，owned身份全部退出，TERM到父退出43.137364秒，18083连续≥5秒无监听，私有PCM缓存为空。root后续再次独立核验全部PID/birth和控制器已退出、共享PID1601397出生时间不变/8007保留。原测试缓存目录保留，未删除历史日志/报告。卡7背景used15908→15910MB（+2MB），共享进程占用12417MB不变；可描述回到近背景，不能说精确同值或证明每个分配均无泄漏，控制器原始near_background=unknown保持原样。

host内存充裕，背景后available约1.916TB、swap_used0；阶段主存/swap采样见[evidence/host-swap-phase-summary.json](evidence/host-swap-phase-summary.json)，OOM前后driver/allocator口径见B4-MEMORY-REPORT.json，不能把HBM OOM误判为host swap耗尽。

## 状态与证据索引

当前27/29：代码、CPU/实际保护、A6/A4/B4、真实准入、具体差异和正常收尾完成。5.3最终增强池规模标定、5.6共驻A6验收保留未勾；B4当前1/1配置是冻结测试快照，不等于最终共驻池选型。所有mandatory验收完成后才同步主规格并归档；三个旧change和失败矩阵原样保留。

原始完整目录：`/home/zhangs/asr_nvidia/logs/qwen-multi-instance-20261124/formal-m6-s16-v2`；完整逐请求响应/状态时间序列保留远端，不覆盖。仓库[evidence](evidence/)包含关键原始summary、run/完整性、输入SHA、六owner真实forward、配置快照、关闭与背景证据；分析JSON保留输入绝对路径/SHA。CPU证据原件`/tmp/qwen-multi-apply-20261124-sduu4fzl`、保护stage`/tmp/qwen-multi-protected-20261124-ah3u4jrq`，所有owned任务已结束。Git交付身份见后续交付元数据及提交记录。

严格OpenSpec校验：本change 1/1通过，当前main 9/9通过；只在临时副本合并预览main+本change，9/9通过。覆盖审计逐Requirement确认未修改main条目保留、新delta精确生效；未写回main或旧change。见[evidence/spec-review-ledger.json](evidence/spec-review-ledger.json)。
