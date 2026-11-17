# 华为910B本地ASR迁移验证报告（2026-11-17～18）

v3接口保持原契约，生产路径改为本地Qwen与原生torch_npu增强模型。两档八条完整课堂WAV均8/8成功并正常关闭；完整MP3、短样本、错误映射、安全取消与模型数值参考已有实机证据。依据完整课堂结果，正式默认采用单Qwen副本、segment/request双门限1/1。最终1/1源版本449项CPU回归和编译版NPU稳态均通过，测试自有服务全部正常关闭。提交与独立规格归档记录随后留档；实际Docker构建与上线仍按授权范围后置。

## 环境、执行设备与接口

| 项目 | 本次条件 |
| --- | --- |
| 远程项目 | `xjtu910b7:/home/zhangs/asr_nvidia`；分支`refactor/adapt-qwen-asr-2huawei910b`；迁移前HEAD `79c7c1bbc94ec1824281c8e06c26e4f2a3ae5ff9` |
| 硬件 | Ascend910B3、65536MB，仅物理npu:7。既有uvicorn PID1601397保留，测试采样11875MB；所有结果属于共享卡测量 |
| 正式环境 | `/home/zhangs/conda-envs/asr/bin/python`；Python3.10.22/aarch64；CANN8.1.RC1；torch/torch-npu/torchaudio2.5.1、transformers5.13.0、numpy1.26.4、funasr1.2.6、av14.2.0、torchvision0.20.1、attrs26.1.0 |
| Qwen | `/home/model/Qwen3-ASR-1.7B-hf`，单副本、BF16、默认SDPA、Transformers eager+torch_npu、max_new_tokens256；707个浮点参数实际BF16 |
| CAM++/emo | 原checkpoint，3CAM++/1emotion2vec，参数/浮点缓冲FP32；项目实例适配pooling和精确GELU(none)，进程一次关闭conv/matmul HF32；不改site-packages或转换权重 |
| 容量 | segment_concurrency=1、request_concurrency=1、audio_concurrency=8、B上限4；模型实例容量与逻辑闸门分开计数 |
| VAD | 固定spawn/FIFO，2CPU+2npu:7 worker；不声称该配比最优。容器物理7映射为逻辑0，以实际可见性预检为准 |

**Qwen、CAM++和emotion2vec实际推理均在npu:7。** 音频解码、fbank/部分前端、后处理和两个VAD worker使用CPU；CPU逻辑/故障注入测试与NPU硬件验收分别记录。g2实时`npu-smi info -t proc-mem -i 7 -c 0`曾显示主PID455613占5746MB、两个NPU VAD worker245/241MB。这些是当时设备进程快照，非逐模型独占内存。

硬件命令先`set +u; . /usr/local/Ascend/ascend-toolkit/set_env.sh`，保留CANN的PYTHONPATH。包metadata、实际导入、pip依赖、模型前向和HTTP验收分别留证。主asr不原地升降级；候选依赖使用独立前缀。

v3请求字段/默认值、八种增强组合、响应类型、两位小数bg/ed、字符串load_audio_time_ms/gpu_time_ms、speed_info列表及错误payload保持。gpu_time_ms仍为原处理墙钟范围，不改为kernel时间。空/省略/auto语言自动识别；4009非法语言，4011加载不可用，4012执行/超时，4008无人声；原v2/旧版路由保持404。客户端类名、transcribe_bytes、异常及raw前缀解析保留，内部不再调用独立Qwen HTTP服务。

## 样本与冻结参考

音频清单登记54文件（52WAV、2MP3）、5,144,064,354bytes；WAV头合计157175.093375秒（43.659748小时），2组重复SHA涉及4文件不计独立来源。清单保存GBK原始path_hex，上传使用ASCII sample_id，不重命名/覆盖原样本。文件头检查、片段/完整解码、实际转写与人工标注分开，未转写全部43.66小时语料。

迁移前有效CPU单测为247通过/19失败/1跳过/118子断言；原HEAD独立v3合同重放5通过/11子断言。最终逻辑回归与硬件结果见最终门槛。旧343、397、446项以及旧29模块编译版属于历史阶段，不能替代最终源版本。

## 八路完整课堂WAV与默认选择

同一八条完整SHA去重WAV，4B+4A、总峰8/B峰4；总22253.408音频秒，g1双门限1/1、g2为2/2，均单Qwen、3CAM++/1emo、相同VAD/业务阈值。先逐文件核SHA/size/mtime再完整上传，不裁切、不用重复片段充数。

| 档位 | 成功 | P50秒 | P95秒 | 最大秒 |
| --- | ---: | ---: | ---: | ---: |
| g1 | 8/8 | 4104.338199446 | 4781.662453105 | 4795.935395310 |
| g2 | 8/8 | 4304.155876740 | 5002.106922458 | 5018.234334021 |

| 样本ID | 类型 | 音频秒 | g1请求秒 | g2请求秒 | g1/g2 RTF | 完整SHA256 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| audio-bbdf7c1a76f88262e22a | B | 2615.808 | 3628.947267 | 3795.947365 | 1.387314 / 1.451157 | `25782e0ad459d6be270188951da7343a6951a78126bd549e9ee5c69a476a2f0e` |
| audio-ff1d4cae9950cf37211f | B | 2702.016 | 3944.004417 | 4130.308729 | 1.459653 / 1.528603 | `99195ce63638c7a03c37b319054a655112325fe34c5b615e4e3151a88a6df8be` |
| audio-6fbfabcc5926e1583dab | B | 2820.736 | 4264.671982 | 4478.003025 | 1.511900 / 1.587530 | `2e42db093a4149752ddeb1dd5d0c05efe7bec18a5f3147fe4e7cd947e3c8b6a1` |
| audio-61d3d13113dbcf87833e | B | 2821.056 | 2108.681031 | 2230.408657 | 0.747479 / 0.790629 | `5ae4039dea035b36b64c4dd4a0f1f774190a2080ec95d67aca91dceccf9db82a` |
| audio-28a0c66565d3814b32af | A | 2821.120 | 3873.073087 | 4085.840344 | 1.372885 / 1.448304 | `5c663ae9217e2ec58d86459f115e11136d015671928b5cfceffe226d96b29a2e` |
| audio-b0b2abaab027fbd04b81 | A | 2822.560 | 4755.155560 | 4972.156015 | 1.684696 / 1.761577 | `0e9f4df99a73f4e5ea9b7415a9cac4db3a710dabe0beda5b722e7236ad36985e` |
| audio-36029767f173c95244c6 | A | 2823.040 | 4795.935395 | 5018.234334 | 1.698855 / 1.777599 | `f466ac0e160fb591b982e72a9f3e0faa99657abf5a807d4af2fcac7aa6526ffe` |
| audio-5d5328ae33020b071a83 | A | 2827.072 | 4685.418327 | 4895.311655 | 1.657340 / 1.731584 | `0d23b47226d120d621ac5c0a66ae71a03fe365964873ac304bc84ccb72b67212` |

P50/P95使用numpy.percentile默认linear，n=8；RTF=逐请求elapsed/完整音频时长。elapsed_s严格测量客户端开始上传到完整响应，包含首轮Qwen加载，VAD启动预热在请求前。旧工装未保存每请求独立起止时间或精确批次monotonic跨度，不能补造。用总音频秒/最大请求elapsed仅可给近似批吞吐代理，非精确实测批吞吐；两档该代理分别为4.640056/4.434510音频秒/秒；这是明确标注的近似值。原始逐请求数据可复算。

g2相对g1的P50/P95分别约慢4.87%/4.61%，host RSS采样峰也更高，因此采用保守的1/1课堂默认；显式2/2仍可配置。共享卡、先后批次和n=8不支持严格单因素因果或全局最优结论。两档计算模块完全相同，g2仅main增加关闭失败非零退出修复，差异记录于full-source-provenance.json。

先前12秒同源暖标定每档两轮4B+4A（16次）为：g1/i1 P50/P95=20.02011025/23.56287287秒，g2/i1=14.14107574/23.72603279，g2/i2=14.45330876/22.89813554。短测曾支持2/2，但不代表完整课堂；双副本无足够收益，不采用。历史数字保留，最终选择依据完整课堂。

| 资源口径 | g1 | g2 |
| --- | ---: | ---: |
| 主进程RSS峰MiB | 5407.438 | 7914.422 |
| 主进程allocated采样峰MiB | 4710.774 | 4671.431 |
| 主进程reserved采样峰MiB | 5446.000 | 5342.000 |
| owned进程树npu-smi采样峰MB | 6298.000 | 6232.000 |
| 整卡HBM采样峰MB | 21507.000 | 21441.000 |
| 主进程CPU平均%，单核100 | 137.437 | 137.120 |

5秒采样可能漏瞬时峰。RSS仅主进程，CPU不含worker及他人进程；allocator仅服务进程配置设备，reserved包含allocated，不能相加。owned树HBM与整卡HBM分开，整卡包含既有11875MB；不由窗口差值声称逐模型精确占用。两批排空active/waiting/busy/lease/cache均回零，主进程与4worker退出0，无强制回收、孤儿或监听，关闭后观察5秒。

原始目录均在`/tmp/asr-apply-20261117-a99trpfv/e2e-full-g{1,2}-final/`；仓库full-pair-final-evidence.json包含数字摘要及每份原始文件SHA，full-pair-capacity-decision.json记录最终选择，不将课堂全文放入摘要。

## 质量、阈值与完整长音频

CAM CPU FP32参考与NPU原生数值、0.3阈值聚类、完整课堂聚类分布及复读修复观测见classroom-threshold-review.md。g1八文件输出段数311/351/400/166/341/516/535/489；四B聚类2/1/7/9，无人工说话人标签，不据簇数判准。真实复读样例506→6字（2秒），原约253字/秒，超过max(120,40×秒)门槛；响应无残留超门槛段或非法时间轴。本轮保留speaker_similarity_threshold0.3及复读40，不宣称人工准确率或召回达标。

B3首个输出在1207.77秒，独立CPU同权重全长参考原196→过滤189段，首段同为1207.77秒，NPU最终166边界均为其子集。前20分钟一分钟RMS约−94.53～−93.95dBFS、峰不超过−61.37dBFS，所有10ms帧RMS低于−60/−40，支持输入近量化噪声解释；无人工声音标注，不称自动证明语音召回。CPU参考路径/SHA与选项来源保存，不将头20分钟空输出误判为迁移新增丢失。

FunASR nested vad_kwargs不等于覆盖构造。原权重实际model_conf为max_single8000ms、end800ms、noise0.6、global decibel/snr均−100；声明12000/800/0.8/−40/18保留既有传递。服务段级SNR18、独立segment_decibel−40/RMS−40与前后3分钟窗口实际生效，模型全局值与服务后过滤不能混称；segment_decibel缺省不回退global decibel。迁移不顺手改变参数绑定、阈值或8秒实际段上限。

完整MP3 `audio-37a1c322625a308c612b`：源SHA `706d93e891b352d330b55cc7e55bf9f12c2673d59d5e26d73ff1d1316d17a894`，2422.476音频秒，请求725.2029859600589秒，RTF0.29936436355202645。处理360有效段/失败0，最终349段、12070字符，时间轴0.38～2419秒，无越界/倒序/重叠。其主进程/4worker正常退出、端口释放。104快照RSS峰3310.555MiB、allocated/reserved峰4648.907/5296MiB、owned树5927MB、整卡21137MB；该历史批未采CPUtimes，CPU使用率未知。

短HTTP e2e-short-5共32request记录，八增强组合、语言、静音/空/损坏和排空通过；主进程/4worker正常退出。真实4011/4012/4009请求符合映射，4012先等实际设备工作结束再收尾。真实取消actual_started=actual_finished=4、单实例peak_actual_active=1，最终计数0且close成功。冷A/B双副本边界测试27.563/34.476秒、两请求成功，default dtype恢复FP32，无CAM/emo实例失败，全部正常关闭；双副本只作竞态边界，不是正式部署。

## 正确性修复与候选取舍

### 首次加载、CPU前端与取消

Transformers5.13在BF16 from_pretrained构造期临时改变**进程全局default dtype**。真实CPU复现曾导致native参数构造成BF16，或两个HF实例交错后全局默认残留BF16；仅在load_state_dict后float不能追回已舍入精度。项目增加共享RLock：先取得锁再读旧dtype，finally恢复；Qwen懒导入、processor加载及完整HF权重加载均在guard内，所有浮点权重参数按请求dtype核验，保留库有意的FP32 buffers。native构造、`.float()`及statecopy同FP32 guard，实际参数/浮点buffer核验后才报告FP32。

热实例CPU factory同样受保护：Qwen真实Hann window、CAM Kaldi fbank、CPU增强AutoModel的内部fbank、emo实例ALiBi factory/cache均避开另一个HF BF16窗口。65600个PCM采样点输入（4.1秒）对应的真实Qwen CPU特征与FP32参考逐元素完全相同；浮点特征再明确转BF16，原int32 mask不变。**NPU generate/forward不持有全局guard**，CPU回归以另线程可取得锁证明这一边界。

队列/client采用不会在Python3.10完成边界吞取消的等待方式；取消/超时/重复取消仍等待真实owned线程、设备同步及释放完成。close失败保留实例所有权和失败信息，继续关闭其他实例并可重试；VAD晚到IPC结果在正常stop中排空，强制回收即使最终清干净也视为失败。status只在线程中运行重诊断采样，业务计数/模型池状态仍在主事件循环组装，查询不触发模型加载。

### Qwen候选

详细逐样本SHA、mask、原顺序、raw/parsed与逐次耗时见`docs/npu-910b-migration-20261117/qwen-backend-selection.md`。下表速度均是相应短样本工具范围，含预处理/传输/generate/读取/同步；P50/P95按一次调用或一次批次，不能跨batch大小解释为等价单段延迟。

| 候选 | 成功/速度 | 文本/兼容性 | 本轮采用情况 |
| --- | --- | --- | --- |
| 默认SDPA+单段eager BF16 | 30次稳态成功；P50/P95 736.635/1021.953ms；RTF0.156736、吞吐6.38014音频秒/执行秒 | 6片来自3来源，部分片重叠，2片无人声；无人工参考 | 正式后端 |
| 原生列表batch2、长度分桶 | 15次稳态成功；批P50/P95 958.527/1201.109ms；RTF0.100202、吞吐9.97984，约+56.4% | mask/有效410或510帧/原顺序正确；5/6 parsed等价，1/6七轮稳定实质文本差异，缺人工参考不能判质量通过或失败 | 不进入生产；保留离线工装和后续设计门槛 |
| eager-attention+eager执行 | 30次稳态成功；吞吐6.46508（约+1.33%）；P50/P95 669.214/1075.484ms | 真实root/audio/text config已传播eager；1/6文本变化，P95约恶化5.24%，收益不足稳健 | 不替换默认attention |
| TorchAir+默认SDPA | 42次预热/稳态全失败；首个冷尝试36669.060ms | `Unsupported torch op aten._safe_softmax.default by ge`；无成功转写/稳态速度 | 不采用，不声称永久不支持 |
| TorchAir+eager-attention | 42次全失败；首个冷尝试31653.850ms | 动态音频路径graph break；最终linear输入NoneType，缺完整错误栈不能虚构精确linear根因 | 不采用 |
| vLLM-Ascend历史候选 | 未安装/运行 | 锁定v0.23.0rc1候选需CANN9.0.1、torch2.10等及原版thinker checkpoint，与当前CANN8.1/torch2.5.1/扁平HF资产不匹配；另行隔离可评估 | 当前栈不适用，不能宣称永久不可用 |

TorchAir确实内置于torch_npu；主asr的setuptools84缺pkg_resources导致初期导入失败。独立`/tmp/asr-rebuild-20261117-torchair`仅改setuptools80.9.0后解决该首阻点，再得到上述真实编译失败；不能把“import通过”当“图前向通过”，也未以suppress_errors静默回退伪装图优化。

### CAM++、emo与VAD数值参考

CPU参考使用同原始checkpoint FP32，预登记FP32门槛为CAM cosine≥0.9999且maxabs≤0.001；emo七类概率maxabs≤0.001、七标签顺序及Top1一致。两个不同来源分别取互不重叠1/3/5秒，共6片，重复和回访不计更多独立来源；容差未因失败而放宽。

| eager候选 | 数值gate通过片数 | 最坏maxabs | 暖延迟中位数（18次）/选择 |
| --- | ---: | ---: | --- |
| CAM FP32 | 6/6 | 5.36442e-06 | 79.772ms；采用 |
| CAM BF16 / FP16 | 0/6；5/6 | 0.0815699；0.0103995 | 80.028/98.047ms；不采用 |
| emo FP32 | 6/6 | 4.90993e-06 | 18.465ms；采用 |
| emo BF16 / FP16 | 5/6；6/6 | 0.0280496；0.00148398 | 19.942/20.399ms；BF16有Top1变化，FP16通过其自身0.01门槛但比FP32慢约10.5%，不采用 |

CAM三个dtype在六片上阈值0.3聚类partition碰巧一致，不能抵消失败数值gate或作为人工说话人准确率。emo七标签保留，不能把下游“平淡”等输出当人工标签。TorchAir六个模型×dtype候选首次前向均失败：CAM为BNInfer动态scale通道与192不符，emo为OmegaConf DictConfig tracing拒绝；没有compiled暖态/动态长度成功结果，不采用未验证的ONNX→ATC→OM路径。

VAD AutoModel CPU/native CPU/native NPU在六个短片的边界完全一致，实际误差0（CPU容差0，NPU预登记20ms）。三路跨片单次耗时中位数596.608/609.767/45.075ms，包含不同长度与首推，不能当稳定加速比，也不能据短片优势自动改变2+2配比。完整录音VAD-only、多轮暖态及与Qwen同卡争用需另测。

模型比较原始文件：`model-validation-final-eager.jsonl` SHA `9b181283b6f7a9a1e54466b97cd30b03e1667b66e8067941e2c2ffb4572e1784`；`model-validation-torchair.jsonl` SHA `19288b46f0cb73efb1f674175fe591cd972b7b8e7a28279678dabb642dd0d618`。完整数值、标签、时间与复算程序见`docs/npu-910b-migration-20261117/local-model-comparison.md`。部分源SHA为manifest declared而非本次重hash，原件有明确标记，不混称verified。


## 环境重建、编译与生命周期

空前缀conda explicit+pip lock重建runtime-v2，113包与基线一致，170conda/106pip artifact与来源hash保存；实际导入、依赖和单模型/HTTP证据分层关联。旧clone构建前缀虽metadata NumPy1.26.4却残留2.2.6二进制，触发SciPy类型分裂；保留失败前缀/日志，不将其当正式环境问题或编译代码缺陷。

新干净build前缀`/tmp/asr-rebuild-clean-build-20261117-qr9yhpvu`按锁重建115包=基线113+Cython3.0.12/conda-pack0.8.1，实际NumPy RECORD/无游离.so、SciPy数值、pipcheck、CANN导入和source/compiled各10项CPU parity通过。重建脚本加入payload检查，不依赖clone或metadata掩盖残留。主asr未变更。

最终原30模块stage`/tmp/asr-cython-frontguard-20261117-nhxp0l35`的aarch64/CPython3.10实际扩展导入、HF tiny及实际CPU前端、native guard均通过。shared初始化锁覆盖HF临时全局dtype和懒导入/CPU factory；不在锁内运行NPU forward。后续1/1默认只重编core.config，其他29扩展和main逐字节保留，独立新stage及最终NPU结果见最终门槛。

真实CPU Uvicorn复现了shutdown失败却exit0；main改用同一app对象并将关闭失败映射exit1，正常TERM仍exit0，源版/编译版随机端口均核对释放。Docker supervisor不吞子进程退出码，TERM正常0、INT130、子进程非零/未知127/超时强回收均失败；完整Linux wrapper与竞态测试通过，具体结果见main-shutdown-exit-validation.json和最终回归记录。

所有验收服务按PID+创建时间/自有进程树登记，finally停止接收、等在途工作、关闭worker，核对退出码/孤儿/5秒无监听；超时即失败，即使强清后已空也不改判通过。既有PID1601397不接管、不重启。模型、音频、历史报告、日志、镜像和缓存保留；kernel_meta运行诊断仅gitignore，不删除。

## 最终门槛与证据

| 最终门槛 | 实际结果与证据 |
| --- | --- |
| 最终1/1源版本CPU全套 | **449 passed、146 subtests、2 warnings，85.18秒**；runner exit0、自有子进程全部退出、无强清。`cpu-regression-final-g1-default.json`；原log SHA `093f20f359c0c9ff11d878d40f83be638e75b0e3c0dfb8fa6920c4ab0ae2c3b0` |
| 最终编译产物 | `/tmp/asr-cython-g1-default-20261118-nkmpk_2a`，30扩展实际导入，aarch64/CPython3.10；仅core.config重编，其余29扩展与main字节不变；`protected-build-final-g1-config.json` SHA `30cff6534757e52eefefcc6a0c363e95be8d277672e6162812bb921cd5847d56` |
| 源版本关联 | snapshot SHA `4050173f89303bc5477078a23dbe8ac3d4be84f95f95dba6eb0ccb75a427a992`；canonical fingerprint `8ee9574b395dbfafd0a62954d4f5bf304c86d6044717bbf90e9a1a7a13f19534`；source_changed=[]；7文件默认变更见`g1-default-config-application.json` |
| 最终编译版NPU稳态 | **3次全B预热+16/16交替A/全B请求通过**，12秒同源固定shape，10秒冷却；P50/P95 **2.966116319876164/3.499544832855463秒**；`compiled-final-g1-steady-evidence.json` |
| 稳态资源 | allocated/reserved增长0，RSS增长5701632bytes，threads/FD增长0；预登记容差allocated16MiB、reserved/RSS256MiB、threads8、FD4；cache/lease空、4worker健康，8项check全true；只支持此短稳态窗口，不声称长期无泄漏 |
| 最终编译版收尾 | 主服务PID3681279与4worker均exit0，harness3672743退出；无强清/孤儿，端口18083释放并观察5秒；原有1601397保留 |
| 分层硬件验收 | 短HTTP/错误/取消、完整2422.476秒MP3、相同8完整WAV的g1/g2各8/8、最终compiled稳态均有原件；质量参考限制保留 |

八WAV两档合计3109个最终分段，parsed文本、bg/ed、segment_text、role/emo、language/ratio全部逐值相同；比较摘要不含课堂全文。g2跨午夜轮转文件一并统计，同样有1次506→6复读修复、0超长保留/错误签名；不能只读取当天application.log的0告警。详见full-pair-output-comparison.json及课堂阈值报告。

CPU源码回归的实际命令为source CANN后`TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/zhangs/conda-envs/asr/bin/python -m pytest -q`，退出0；未初始化NPU。最终硬件的实际命令为：

```bash
cd /home/zhangs/asr_nvidia
set +u
. /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/zhangs/conda-envs/asr/bin/python -u scripts/validate_npu_service.py \
  --mode steady --concurrency 1 --instances 1 \
  --app-root /tmp/asr-cython-g1-default-20261118-nkmpk_2a \
  --models-root /home/zhangs/asr_nvidia/model \
  --manifest /home/zhangs/asr_nvidia/docs/npu-910b-migration-20261117/audio-manifest.json \
  --output /tmp/asr-apply-20261117-a99trpfv/e2e-compiled-final-g1-steady --port 18083
```

上述输出目录为已执行原件，复跑须改为新目录；harness的finally已执行正常关闭。提交推送、主规格同步与独立归档由随后中文Git记录及交付记录留档，不把“生成报告”等同于已推送。


配套文件均位于`docs/npu-910b-migration-20261117/`：audio-manifest/model-assets登记资产；test-coverage映射用例与任务；qwen-backend-selection/local-model-comparison记录候选；classroom-threshold-review记录质量参考；full-pair-final-evidence及capacity-decision给出逐请求和原始SHA；环境/编译/关闭CPU证明单独命名。最终数字须与各自JSON、原日志和source snapshot关联，不把旧版本证据提升为新版本。

复跑八完整WAV使用`validate_npu_service.py --mode full --concurrency 1或2 --instances 1 --output <新目录> --port 18083`；最终编译稳态使用同一工装`--mode steady --app-root <最终stage> --models-root /home/zhangs/asr_nvidia/model --concurrency 1 --instances 1`。每个输出目录必须新建、NPU批次串行。原始响应/全文/embedding留远端原件，不提交权重、音频或编译二进制。

## 限制与失败留档

没有可追溯CUDA golden；CPU FP32参考不是NVIDIA与910B性能比较。没有人工转写/角色/情绪标注，因此不宣称WER/CER、说话人/情绪准确率或有效对话召回≥98%；上线质量门槛仍需带标注验收。未在共享卡制造OOM、ACL损坏或断卡，精确锁/拷贝/释放故障主要CPU注入，不冒充硬件故障已验证。共享卡有限样本原生方案属于当前栈可验证选择，不能称全局性能最佳。

未source CANN的导入失败、PYTHONPATH覆盖导致4011、工装字段误断、旧INT关闭、CAM HF32/emo GELU精度失败、TorchAir失败、HF懒导入dtype竞态、NumPy clone污染等失败原件均保留；修复后新证据单独关联，未放宽原数值门槛。**实际Docker镜像构建、conda-pack发布、容器NPU承载和上线后置，未执行。**
