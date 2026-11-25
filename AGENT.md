# AGENT.md

> 本文件面向在本仓库工作的 AI Agent，以及第一次接手这个项目的人。  
> 前半部分说明「这是什么系统、代码在哪」；后半部分说明「哪些事不能做、做完怎么算完成」。
>
> **冲突时的优先级**：实测数据 > `openspec/specs/` > `README.md` / `docs/` > 本文件。  
> 本文件若与代码现状不一致，以代码为准，并请顺手修正本文件。

---

## 0. 速查

| 项目      | 值                                                                     |
| ------- | --------------------------------------------------------------------- |
| 代码位置    | `xjtu910b7:/home/zhangs/asr_nvidia`   |
| 服务端口    | `8083`（容器内 `9000`，compose 映射 `${ASR_PORT:-9000}`）                     |
| 进程名     | 主服务 `asr-offline-model`，VAD 独立 `asr-offline-vad`                             |
| 启动      | 先 source CANN set_env.sh，再用 `/home/zhangs/conda-envs/asr/bin/python` |
| 单测      | `pytest tests/`                                                       |
| 规格目录    | `openspec/specs/`（当前基线）、`openspec/changes/archive/`（演进历史）             |
| 镜像构建    | `docker/Dockerfile`源码镜像、`docker/docker-compose.yml`；本版独立构建/B4验收 |
| Qwen 依赖 | 本地原始权重，`[qwen_asr].local_model_dir` + `local_device`                      |

> 注意 `test/`（单数）是**实验与压测脚本**，`tests/`（复数）才是 pytest 单元测试。两者不要混。

---

## 1. 项目是什么

**SeaCraftASR**：面向课堂录音的离线语音转写服务（FastAPI）。

- 当前**只有 v3 一条链路**：`POST /v3/seacraft_asr`。v1.1.8（SeACo-Paraformer + PUNC + Whisper）和 v2 路由已下线，代码与模型均已清理，不要"顺手恢复"。
- FSMN-VAD是固定spawn进程池；CAM++启动预加载池可选thread/process（loader兼容缺省thread，当前模板process4实例），emotion2vec保持主进程runner。
- **Qwen3-ASR使用本地原始权重**，BF16 / npu:7；当前continuous/process、顺序预加载4份owner，每请求独占一份、每份窗口16、全局64，HTTP仍1；CAM++4个process实例、emo1。用户已采用该最终配置，宿主B4与增强驻留A4已完成4/4验收（200.10/183.47秒）。历史N6/N5结果及失败保留，详见文末检查点。
- 对外只有三个接口：`POST /v3/seacraft_asr`、`GET /get_status`、`DELETE /clear_tasks_list`。

### 业务目标（容易理解错，先看这段）

要的不是"把更多音频塞给 Qwen"，而是**让 Qwen 只处理真正该转写的内容**：

- **不该送进 Qwen 的**：课堂闲谈、咳嗽、翻页、敲击、椅子摩擦、教室环境底噪等非讲授语音
- **必须保住不丢的**：教师讲授、学生发言，**包括带方言口音的发言**（不因口音、低音量、语速慢而被当成噪声剔掉）
- 理想结果是两者同时成立：噪声几乎不进 Qwen，学生方言一句不丢

阈值取舍的原则由此确定：**降噪的收益是省一次推理，误删的代价是一段内容永久缺失**。
两者不对等，所以宁可放过少量噪声，也不能误杀有效发言。"放过噪声"是容错边界，**不是目标**。

> **当前能力边界（别误以为已经做到）**：现有三重过滤（低能量帧比例 / SNR / RMS dBFS）全部基于
> **能量与信噪比**，能拦的是咳嗽、翻页、敲击、椅子摩擦这类弱能量段；**课堂闲谈是正常人声，能量不低，
> 现有实现拦不住，仍然会送进 Qwen**。要真正区分"讲授 vs 闲谈"，属于
> `docs/课堂录音VAD细小声音过滤与说话人统计优化方案.md` 里的后续阶段（环境音分类模型 / 专用分类器），
> 不是调阈值能解决的。讨论这块需求时先讲清这条边界。

---

## 2. 一次请求怎么跑

```
上传音频
  ↓ 整段解码并使用服务专属缓存供VAD进程读取
VAD 固定进程池（vad_device 数组，每元素一个进程）
  ↓ FSMN-VAD 切分 → min_segment_duration_s 短段丢弃
  ↓ 过滤窗口判定：只有落在开头 after_start / 结尾 before_end 分钟内的段才进三重过滤，
    窗口外的中间段无条件保留（宁可放过噪声，也不误删讲授主体）
  ↓ 段级三重过滤（命中即丢，不发 Qwen）
      ① vad_decibel_low_frame_ratio_thres  低能量帧比例
      ② vad_snr_thres                      段级 SNR
      ③ vad_low_volume_rms_dbfs_thres      段级 RMS dBFS
  ↓
每段1个Qwen逻辑任务（内存WAV，单音频/全局窗口独立计数，多个独立段可组成物理批）
  ├─ 独立队列：CAM++声纹（从0.3秒实际推理）→ 全音频聚类；仅showRoleIdentify需要时作教师/学生判定
  └─ 独立队列：emotion2vec情绪（1～10s闭区间外记"平淡"）
Qwen完成即还自身名额，增强队列/汇合受段数和PCM字节预算反压；按原段汇合后继续后处理
  ↓
复读修复 → ITN → 标点卫生（纯标点段删除、去段首标点）→ 语速 / 语种占比
```

**核心不变量**（改代码时最容易踩坏的地方）：

1. **一个有效VAD段对应一个独立Qwen逻辑任务及至多一个有效响应分段**。允许物理批执行，不合并相邻段、不拼PCM；`bg`/`ed`取原VAD边界，既有空文本/标点过滤仍可能删除响应段。
2. **分段音频在内存流转**，不新增段级临时 WAV；整段上传沿用服务专属缓存供VAD worker读取，请求finally清理。
3. **VAD 池是静态的**：启动时一次性预热，任务 FIFO 分派，运行期不扩缩容。数组长度就是并发上限。
4. **NPU 加载失败禁止静默走 CPU**；预检/加载错误必须按阶段暴露。

---

## 3. 代码地图

### 入口与装配

| 文件                                                 | 职责                                   | 改动前必读                    |
| -------------------------------------------------- | ------------------------------------ | ------------------------ |
| `main.py`                                          | FastAPI 装配、lifespan、进程命名、NPU 环境/资产预检 | `core/deployment.py`     |
| `config.toml`                                      | 本地运行配置（唯一配置源）                        | `core/config.py` 的字段定义   |
| `core/config.py`                                   | 配置加载、校验、派生属性                         | 570 行，改字段要同步 README 配置章节 |
| `model_status_runtime.py` / `telemetry_runtime.py` | 运行时轻量封装                              | —                        |

### HTTP 层

| 文件                              | 职责                                  |
| ------------------------------- | ----------------------------------- |
| `api/routes/asr_v3.py`          | `/v3/seacraft_asr` 路由与参数校验          |
| `api/routes/asr_qwen_common.py` | v3主链路：VAD → 有界独立分支 → 原段汇合及响应 |
| `api/routes/asr_common.py`      | 请求上下文构建、情绪标签映射、临时文件清理               |
| `api/routes/status.py`          | `/get_status` 路由                    |
| `entity/data.py`                | `AsrRequestParams` 请求数据类            |

### 核心能力

| 文件                             | 职责                                                             |
| ------------------------------ | -------------------------------------------------------------- |
| `core/vad.py`                  | VAD 切分配置组装                                                     |
| `core/vad_pool.py`             | 固定spawn VAD池，完整音频FIFO、任一完成即补位与排队/worker包络计时 |
| `core/vad_volume.py`           | 段级声音特征（RMS / SNR / dBFS）与低音量过滤，**运行时与标定脚本共用同一套口径**             |
| `core/qwen_asr_queue.py`       | 三层并发闸门：音频准入分类 + 段级/全局请求信号量                                     |
| `core/qwen_asr_client.py`      | 本地Qwen实例租约、批执行/重试与批遥测 |
| `core/qwen_asr_runtime.py`     | 列表音频、逐成员语言/mask/输出及实际processor shape |
| `core/qwen_asr_process.py` | 单Qwen spawn模型进程、READY/RPC归属、独立资源与显式关闭grace；CPU状态不触发加载 |
| `core/qwen_batch_scheduler.py` | 公平共享队列、长度分桶、有限等待及估算帧/token预算 |
| `core/segment_pipeline.py`    | 每音频有限元worker、独立全局段数/PCM预算、取消安全汇合及异常链脱离 |
| `core/models.py`               | CAM++/emo启动预加载、共享有界预取、FIFO反压、即时交接、HBM/忙闲状态 |
| `core/cam_process.py` | CAM原生FP32模型spawn worker、READY/RPC出生归属、Tensor[1,192]与显式关闭 |
| `core/local_inference.py`      | 设备 → 后端映射（`pytorch_cpu` / `torch_npu`） |
| `core/deployment.py`           | 昇腾 profile 启动前环境与资产校验                                 |
| `core/status.py`               | 状态接口响应组装                                                       |
| `core/logging.py`              | 日志按天 + 按大小双轮转                                                  |
| `core/request_telemetry.py`    | 有界请求级阶段遥测（只存统计，不存音频与正文）                                        |
| `core/progress.py`             | 可选的每音频处理进度显示                                                   |
| `core/async_utils.py`          | `await_completion`：取消时等待已启动工作收尾，避免提前归还资源                       |
| `core/resource_diagnostics.py` | 模型内存压力安全采样                                                     |

### 文本与工具

| 文件                          | 职责                                         |
| --------------------------- | ------------------------------------------ |
| `utils/itn.py`              | WeTextProcessing 逆文本规范化                    |
| `utils/repetition.py`       | 生成复读（幻觉循环）检测与折叠                            |
| `utils/text_postprocess.py` | 标点卫生、语气词处理等确定性规则                           |
| `utils/qwen_response.py`    | Qwen 返回解析：`language X<asr_text>` 前缀剥离、语种占比 |
| `utils/audio_utils.py`      | 音频解码、共用round端点裁剪/EOF clamp、内存WAV、服务缓存目录 |
| `utils/asr_stats.py`        | `asr_stats.json` 原子读写（带文件锁）                |
| `utils/feature_utils.py`    | 说话人角色判定（教师关键词等）                            |

### 周边目录

| 目录                                   | 内容                                                     | 是否入库                         |
| ------------------------------------ | ------------------------------------------------------ | ---------------------------- |
| `openspec/specs/`                    | 9 份能力契约（系统基线）                                          | 是                            |
| `openspec/changes/archive/`          | 19 个已归档变更，是**演进决策的唯一可信来源**                             | 是                            |
| `docs/`                              | 压测报告、方法说明、验证结论，命名带日期                                   | 是                            |
| `scripts/`                           | 压测 / 标定 / 导出 / 校验脚本（`bench_*`、`validate_*`、`export_*`） | 是                            |
| `tests/`                             | pytest 单元与集成测试                                         | 是                            |
| `test/`                              | 实验脚本与样本音频                                              | 脚本入库，`test/*/` 产物忽略          |
| `docker/`                            | Dockerfile、docker-compose.yml、config.toml、README.md四文件源码交付 | 是                            |
| `model/`                             | 三个模型权重                                                 | **否**（`.gitignore`，但会进构建上下文） |
| `.agents/skills/`                    | openspec 工作流 skill 定义                                  | 是                            |
| `asr_service.log` / `asr_stats.json` | 运行产物                                                   | **否**                        |

---

## 4. 三层并发与错误码

| 层       | 参数                                                | 语义                            | 当前值                 |
| ------- | ------------------------------------------------- | ----------------------------- | ------------------- |
| 整音频准入 | `audio_concurrency` / `audio_spk_emo_concurrency` | continuous总数由owner数N派生 / B子集上限 | 4 / 4 |
| Qwen 请求 | `segment_concurrency` / `request_concurrency` | 每音频段窗口S / 全局由N×S派生 | 16 / 64 |
| 本地增强池 | `speech_campplus_devices` / `emotion2vec_devices` | 数组长度=各自实例数，独立于Qwen N | 4×npu:7 process / 1×npu:7（宿主本轮已验收） |

音频准入、Qwen逻辑窗口、物理批大小与模型实例容量分别计数；增加等待任务不增加模型实际计算容量。表中当前值是用户最终选择Qwen4/CAM++4/emo1，宿主B4及增强驻留A4已完成验收；历史压测见`docs/cam*`、`docs/vad-pool*`。

**错误码契约不可随意改号**（客户端已按此判断）：完整定义分散在  
`core/qwen_asr_queue.py`、`core/qwen_asr_client.py`、`core/models.py`、`core/vad_pool.py` 的异常类 `code` 属性；  
对外说明在 `README.md` 错误码表。要点：

- `4008` 无人声 · `4011` Qwen 不可用 · `4012` Qwen 调用失败 · `4013` 总容量满 · `4017` 仅 B 类满 · `4010` 排队超时
- `4015` / `4016` / `4018` 能力未启用或无可用实例
- **设备数组派生的能力启用校验早于容量检查**；两项容量同时满时优先返回 `4013`

---

## 4.1 当前吞吐调度变更（2026-11-18）

`2026-11-18-npu-asr-throughput-scheduling`处于apply，尚未完成性能验收或归档；前次`2026-11-17-npu-910b-local-qwen-asr`已完成并归档，历史实测结果不因本轮候选改写。以当前change的tasks、design和test-plan跟踪必需验收。

- 当前安全缺省：单Qwen、batch1、每音频/全局窗口1/1；现有`config.toml`的混合四VAD及CAM三/emo一属于前次交付配置。本轮独立候选profile为`/tmp/asr-throughput-20261118-xvior99k/candidate128.toml`，定义VAD八CPU、CAM/emo各八NPU:7、Qwen每音频16/全局128、批上限128、预取5、总8/B4；纯B8另配profile。配置校验不等于真实容量/吞吐通过，最终值须按实测更新本文件与README。
- 新`[qwen_asr]`字段：`max_inference_batch_size`缺省1（1～384；384为独立候选）、`batch_wait_timeout_ms`缺省10（0～1000ms）、`max_batch_audio_frames`缺省102400、`max_batch_tokens`缺省78336、`pending_segment_limit`缺省256、`pending_audio_bytes_limit`缺省268435456（256MiB）。完整类型/范围见README和`core/config.py`，修改后重启。帧估计为最长100Hz帧数×B，token估计为`(ceil(max_frames/8)+max_new_tokens+256)×B`；processor实际shape另存，估计不保证显存容量。
- 新`[model_device_pool]`的`speech_campplus_prefetch_queue_depth`、`emotion2vec_prefetch_queue_depth`缺省0（0～64，0无预取），各`*_batch_size`仅支持1。数组长度=实例数；每健康实例等待深度=预取；batch=实际计算。八实例深度5为40等待+最多8运行，不是batch5。共享任务源不将未执行段锁死在忙副本，空闲即补位，反压等待另计。
- 每条完整音频最多占一个VAD worker。8条/4worker时任一FINISHED即分派下一条，不等最初四条或下游；VAD租约与整音频租约分开归还。Qwen完成即释放自己的段窗口，CAM/emo各自归还，最终仍按请求/段标识汇合；段数/PCM预算满时明确反压，不无限堆积future或PCM。
- 每音频元任务worker至多`min(段数,pending_segment_limit)`，缺省256；总八音频至多2048个元任务，不是全局future最多256。真正切片和分支开工另受全局256段/256MiB PCM预算限制，预算满则worker等待，必要分支全部收尾才归还；Qwen逻辑窗口独立释放。响应文本/embedding不计作在途PCM。
- 每次物理attempt独立记录输入成员、真实有效成员、状态及runtime包络；重试排除等待中已取消成员，批summary仅代表最后attempt，不能把上轮有效索引映射到新的幸存列表。传给结果future或汇合结果的异常脱离traceback/cause/context，保留类型/code/message，避免异常链滞留PCM与活动worker frame。
- 正式tensor/NumPy/毫秒裁剪共用起止端点分别round后的采样格并clamp至源范围，EOF不padding、不改原VAD bg/ed；实际裁剪不足0.3秒在三模型前丢弃，emo的1/10秒边界同样依据实际样本。旧FFmpeg秒级转码路径保持兼容，不为此逐段重读完整大音频。
- 本轮候选最短段0.3秒，三项服务能量过滤均false，`vad_decibel_thres=-100`、`after_start=-1`、`before_end=-1`；-1/-1仅不限位置，不能充当关闭开关。FSMN实际绑定/四层证据不变。不足0.3秒在三模型分发前移除；0.3至不足1秒CAM实际推理且embedding参与聚类，emo直接“平淡”；emo仅[1,10]秒推理，未请求能力仍null，roleIdentify本轮不传/false但保留原可选能力。
- 必测同组八条完整SHA去重音频的1/2/4/8并发和同源A/B，冷/暖态分列；默认B4与B8profile分列。单Qwen128必须区分纯Qwen与八CAM/八emo已加载且实际工作的容量，不能互相替代。单份有问题先诊断，再有条件试2/4/8副本与段级均衡。
- 完成P50/P95或任一在途完整请求墙钟超过420秒，下一监测周期（默认1秒）停止追加，捕获准入/VAD/三个模型/汇合/后处理及资源卡点，有界取证并关闭owned资源，记录diagnostic_aborted/未完成。不得原样无限重跑或只报成功快请求；不改变业务HTTP或单Qwen超时。阶段无进展提前告警，定位后先小规模复验。
- 每轮finally按PID和创建时间关闭本轮服务/worker，等待设备和读取完成，验证无孤儿和测试监听释放至少5秒；既有服务保留。内部切片/排队/批等待/CPU前端/传输/generate/同步分开，批包络只计一次，原`gpu_time_ms`不改义。真实Docker构建和上线仍后置。

### CAM-only模型进程候选（吞吐变更v9）

- `[model_device_pool].speech_campplus_execution_mode`支持thread/process，缺省和宿主/容器模板均保持thread；process仅支持合法NPU设备数组（含容器npu:0），数组长度仍是唯一实例数来源，空数组禁用。新增`speech_campplus_process_startup_timeout_s=60`、`speech_campplus_process_shutdown_grace_s=30`均要求真实有限数、0<值≤300，拒bool/字符串；就绪前配置校验不加载模型。
- 每CAM worker保持原torch_npu、FP32及CPU float32 Tensor[1,192]输出；emo留父进程、Qwen独立配置，不新开HTTP端口、不换格式/权重或升级依赖。尚未请求CAM和查询状态不spawn/RPC。普通取消等真实DONE，只有显式关闭启用grace并按PID/出生身份回收；强制、坏ACK或非0退出为失败，确认死亡前不还真实容量。
- 主循环读取ModelPool的`model_workers`缓存与每实例memory（含loading/retired），资源collector仅只读OS。父/Qwen/CAM/外部/未验证进程分开，CAM只按出生身份吻合且未退出归属；worker allocator/lastRPC峰值、OS RSS/PSS、IPC额外副本及设备HBM各自保留，不用父0代worker0、不加不同时刻峰值。保持状态singleflight/age语义。
- 原型组件结果不等于正式TorchTensor聚类、动态全语料或完整音频矩阵通过。每worker约16秒启动、八进程原型约272秒全程是冷启动/装载/执行/关闭成本，不能当暖态推理耗时。最终默认只有完整矩阵及源/保护版本验收后才推广，当前安全配置其他值不变。

### 单模型进程与多文件公平供给（吞吐变更v5）

- 新`execution_mode`支持thread/process，当前安全缺省thread；process支持经校验1～8配置实例（实际NPU合格数量另列），每子进程加载一份同HF Qwen，父进程不重复加载，不增加HTTP服务或更改引擎/精度。设备直接沿用宿主npu:7/容器逻辑npu:0。模式更改需重启，最终默认待整音频验收。
- 新`process_shutdown_grace_s`缺省30，有限正数≤300秒，仅显式关闭生效，不将单请求120秒timeout改为kill；超期按worker PID/创建时间有界终止/join且关闭失败。服务shutdown入口提前启动Qwen关闭看护，避免增强关闭先阻塞。状态查询不懒启动、不等待RPC；worker与父进程的allocator/峰值/RSS分开，IPC副本不伪称受主PCM256MiB覆盖。
- 已完成固定128段batch1/8/16/32/64/128每档三轮稳态及独立质量复算，文本差异不作准确率结论。单模型process在16/128段B配对比同进程更快，所有Qwen/CAM/emo对照及owned关闭通过。process预取0/1/5三档实测当前0中位最低；新候选`final-process-b4.toml`及`final-process-b8.toml`保留旧候选不覆盖，整音频矩阵仍待验收。证据见`docs/npu-throughput-20261118/qwen-process-paired-v5.json`、`process-prefetch-selection-v5.json`。
- 先获取每音频/全局Qwen credit再占PCM预算，保证八个VAD-ready文件可共同供给128。Qwen实际收尾即释放，剪切错误/EOF/取消幂等归还。等待PCM的预留credit不等于实际执行段，真实缓冲耗尽仍反压。

### Qwen独立前端与批预算（吞吐变更v2）

- 每成员在FP32 guard中独立调用processor，再把有效tokens左补齐、完整single features/mask额外右补齐；不得先把不同PCM补齐再STFT。保留single自带的无效尾部特征上下文，batch1不重排。未知processor键/shape/dtype明确失败，不改site-packages或权重。
- 当前锁定processor为`hop_length=160`、`min_length=8000`、`n_window=50`。调度估计`frames=floor(max(samples,8000)/160)`，完整特征补至`100*ceil(frames/100)`；帧预算必须计入这100帧补齐。token预算仍是8:1音频估计加生成上限和256提示预留，不当作真实token数/HBM保证。模型/processor版本改变先复验这些属性。
- 新`[qwen_asr].batch_max_duration_ratio`缺省4.0，接受1～64有限数，布尔/字符串/NaN/Inf就绪前拒绝。它控制有效STFT帧最长/最短比，含原0.5秒前端下限；32.0仅为候选。与实例数、单音频/全局窗口和批上限独立，修改后重启。
- 越过不兼容队首取尾成员需要防饥饿：每批公平锚点选全局最老等待队首，再在兼容集合中按请求轮转。不能仅凭成员轮转声称旧长段必有机会。
- CPU真实128段前端等价证据见`docs/npu-throughput-20261118/qwen-frontend-equivalence-v2.json`；原差异及HOL诊断见`qwen-quality-initial.json`。CPU等价及配置校验不表示NPU文本一致、完整并发矩阵或默认性能已经验收。

## 5. 配置体系

- 唯一源：`config.toml`（容器内通过 `CONFIG_PATH=/config.toml` 挂载覆盖，镜像内另有模板）
- 加载器：`core/config.py`，字段名与 TOML 键一一对应
- **配置错误必须在模型预热和对外就绪之前抛出**（`main.py` lifespan 第一段就是这个目的），不要改成懒校验
- 部分参数改动**需要重启生效**（设备数组、本地目录、`vad_speech_noise_thres` 等），改完必须在注释里标明

---

## 6. 构建与部署

当前源码容器入口为 `docker/Dockerfile` 与 `docker/docker-compose.yml`，命令见 `docker/README.md`；目录恰好四文件，另两份为config.toml和README.md。本版直接运行必要.py，不使用Nginx/旧start.sh/build.sh，不编译Cython；仓库独立保护工具及历史验证仍保留，保护模式的源码剥离只允许在有标记的staging中执行，禁止真实仓库剥离。

底座固定本机CANN8.2.RC1的实际ID，容器独立Python/torch2.5.1/torch-npu2.5.1.post1，精确post1门禁由ASR_CONTAINER_RUNTIME=cann82-source启用；宿主CANN8.1/asr环境和旧锁不升级。构建工具、环境归档、三个必要小模型分别准备，最小上下文不得含项目测试/文档/脚本/样本/OpenSpec或旧业务扩展；不将整个仓库或工具环境打进镜像。

Compose只读挂载宿主config到/config.toml，缺文件/目录/不可读明确失败，不含镜像配置fallback；修改后重建容器。三个小模型内置，Qwen原始HF独立只读挂载。单HTTP仅支持instance_count=1，模型数使用local_instance_count，默认Qwen4/S16、CAM4 process/emo1；增加Qwen不复制VAD/增强池。宿主只用物理7，单卡容器使用逻辑0，实际映射须验证。

镜像名asr_offline_910b:v3_YYYYMMDD，最终tag引用已通过真实完整B4和正常关闭的同一image ID，不在验收后重建。HTTP就绪、加载、真实forward、暖身覆盖及完整性能分级记录；健康启动窗口360秒、Compose停止240秒。B4使用canonical a1～a4双增强完整WAV，336秒诊断中止及真实故障规则、严格无警告与完成率分开，不能用HTTP200或短片代替。未取得证据不宣称可用/生产上线/人工准确率通过。

用户已明确授权本change成功验收后清理本轮中间镜像、独立builder缓存和辅助资源。此授权不包括其它镜像/共享服务、基础/最终镜像、最终必需共享层或日志报告；失败期间保留缓存。执行前列出精确ID和引用检查，禁止全局prune及按端口/名称批量终止。测试服务finally正常关闭，核对owned身份、端口连续空闲≥5秒、PCM为空及共享资源保持，强制/非零退出不计通过。

## 7. openspec 工作流

本项目的规格与变更记录是**一等公民**，不是可选文档。

```
openspec/specs/<capability>/spec.md      当前系统「必须做到什么」（基线）
openspec/changes/archive/<change>/       历史上的每次变更（proposal / design / tasks / spec delta）
```

已归档的 19 个 change 构成完整演进史，遇到"为什么这么设计"的问题，先查 archive，别猜。

**写作约定**（`openspec/config.yaml`）：

- 正文中文；`Purpose`、`Requirement`、`Scenario`、`ADDED Requirements` 等结构关键字保持英文
- change 目录名 `YYYY-MM-DD-kebab-case`，日期取创建当天

**工作流**（`.agents/skills/` 里有对应 skill）：

1. **propose** — 只产出规划产物（proposal / design / tasks / spec delta），**不写业务代码**
2. **apply** — 按 tasks 实现，勾选任务
3. **archive** — 实现完成 + 验证通过后归档，**同时把 delta 同步进 `openspec/specs/`**

---

## 8. 项目开发约束（人 + Agent 共同遵守）

以下约束来自这个项目已经形成的做法，不是凭空规定。

### 8.0 一次改动的标准流程

不要从"打开文件开始改"起步，按这个顺序走：

1. **先同步**。`git status` / `git pull --rebase` 把本地拉到最新，确认在 `refactor/adapt-qwen-asr-2huawei910b` 分支上、没有别人刚推的改动。**不基于过期代码开工。**
2. **再规划**。业务改动先开 openspec change（propose）；纯文档/typo 可跳过。
3. **改代码**，同步改配置与文档（见 8.2）。
4. **验证**：`pytest tests/` + 相关压测，结论写进报告。
5. **提交并推送**：中文 Conventional Commits，代码 + 报告 + openspec 产物一起进。
6. **变更完成就归档**：openspec change 的 tasks 全勾选、验证通过后，`archive` 并同步 delta 到 `openspec/specs/`，不要让它长期挂在 open 状态。
7. **回写文档**：接口、配置项、默认值有变化时，同步 `README.md` 和本文件。

一句话：**改前先同步，改后必推送留档，change 完成即归档。**

### 8.1 绝对红线

1. **不碰 `model/` 下的权重文件**。它们不进 Git，重新获取成本极高。需要换模型/换 engine 时只改路径与配置。
2. **不执行 `docker builder prune` / `docker system prune`**。构建缓存是这套交付流程能跑通的前提，清掉等于每次全量重建。
3. **不擅自重启或 kill 正在跑的服务**。当前有真实请求在跑，`kill` 会丢任务；需要重启时先确认空闲或按 `docs/DEPLOYMENT.md` 走。
4. **不删除任何镜像 / 日志 / 报告**，除非明确授权。删除前先列出清单并确认无容器引用。
5. **不改错误码数值**，不改接口字段名。这些是对外契约。
6. **不在 `main.py` 之前做"临时验证"式的代码改动**去绕过启动预检。预检失败就是失败。

### 8.2 改代码

1. **先有 openspec change，再动业务代码**。纯 typo、注释、文档修正可以例外。
2. **参数取值必须来自实测标定**，不接受"看起来合理"。已有先例：
   - `speaker_similarity_threshold = 0.3` ← 来自 `scripts/speaker_threshold_sweep.py` 的 sweep
   - 复读阈值 40 字/秒 ← 来自 68,018 段真实样本 p99.9 = 20.3、峰值 26.2
   - VAD 三重过滤阈值 ← `docs/vad-segment-filter-range-validation-20261105.md` + 上线后实测；其中 `vad_snr_thres = 18.0` 是**当前实际承担主要过滤**的一档

   改这些值，必须附上新的统计依据。
3. **改配置项就要同步三处**：`core/config.py` 字段、`config.toml` 默认值与注释、`README.md` 配置章节。少一处就算没改完。
4. **保持 1 段 = 1 请求 = 1 响应分段**，不要为了"效率"合并相邻段。
5. **新增可选依赖必须延迟导入**（`importlib` 或函数内 import）。CPU 环境、无 NPU 依赖的 CPU 测试环境都要能启动。
6. **注释写"为什么"，不写"做了什么"**。现有代码的注释密度偏高，是刻意的，保持一致。
7. **关键逻辑处必须有中文注释**：阈值来源、单位、边界条件、为什么这样取舍（尤其是踩过坑的地方）。新增过滤/判定/并发逻辑时，注释要能让下一个人不看 commit 就看懂意图。
8. **可解耦的配置值一律释放到 `config.toml`**，不要硬编码在代码里。判断标准：这个值调大会变、不同机器/场景可能不同、或者需要在不改代码的情况下试。释放时配套三件事：
   - `core/config.py` 里给类型化字段 + 默认值
   - `config.toml` 里给注释说明取值范围与影响
   - `tests/test_config_contract.py` 覆盖新字段，避免默认值被悄悄改掉
   - 反过来，**纯结构常量（错误码、协议字段名、模型输入 shape）不要配置化**，那属于契约
9. 遵循现有代码风格：`from __future__ import annotations`、dataclass 配置对象、异常类携带 `code` + `message`。

### 8.3 验证与留档

1. **跑 `pytest tests/`**，改动涉及 VAD / 模型池 / 并发时必须跑相关子集并说明结果。
2. **性能与阈值类改动必须出报告**，放 `docs/`，文件名带日期（参考现有 40+ 份报告的命名）。
3. **上线门槛**：有效对话召回 ≥ 98%。低于此值的方案不予采纳，写进报告结论。
4. **不做"只保留老师声音"这类规则**。会误删学生发言，与业务目标直接冲突。
5. 报告要写清：测试条件（NPU、共享占用、并发、样本量）、结论、以及**不适用的场景**。

### 8.4 提交与留档

- **改前先同步、改后必推送**。改完不推送等于没做——这个项目靠 Git 历史 + openspec archive 追溯决策，留在本地工作区的东西下一个人看不到。
- 中文 Conventional Commits，scope 用中文或模块名：`feat(VAD): 增加段级过滤配置与范围控制`、`fix(状态): 显示处理中音频文件名`、`docs(部署): ...`
- **一次提交一件事**，代码 + 对应报告 + openspec 变更一起提交
- 归档 openspec 变更单独一条 `chore(归档): ...` 或 `chore(openspec): ...`
- **openspec change 一旦 tasks 全部完成且验证通过，当次就归档**，不要攒着。长期挂在 open 状态的 change 会失真。

---

## 9. Agent 行为约束

除了上面 8.x 的通用约束，Agent 在本项目还须遵守：

1. **先读后写**。改动前至少读过：目标文件、它调用的 `core/config.py` 字段、对应 `openspec/specs/` 条目。不确定就去读 archive。
2. **改动前说清影响面**：会动哪些文件、影响哪条链路、需不需要重启、要不要重建镜像。用户确认后再动手。
3. **区分"只读探查"和"修改"**。用户说"理解项目"就是只读，不要顺手改格式、补注释、删无用代码。
4. **不许在没有实际执行的情况下声称完成**。测试没过就说没过；构建被超时打断，要区分"构建失败"和"客户端断连但远端仍在跑"。
5. **长任务的正确姿势**：
   - `docker build` 单份 8~25 分钟，SSH 命令超时会断开，但远端进程仍存活
   - 先 `nohup`/后台方式起，再用 `ps` / `docker images` 轮询，**不要因为 SSH 退出码 137/143 就判定构建失败**
   - 判断"是否真在推进"看 IO 采样（`/proc/<pid>/io`）或 `iostat`，不看 CPU
6. **删除类操作三件套**：列出清单 → 确认无引用（`docker ps -a --filter ancestor=`）→ 用户确认 → 执行。绝不 `rm -rf`。
7. **不擅自升级依赖、不擅自改 `requirements*.txt`**，也不跑 `pip install -U`。
8. **不碰 git 历史**：不 `reset --hard`、不 `push -f`、不 `amend` 已推送的提交。
9. **发现规格与实现不一致时，先报告，不擅自"修正"**。这类偏差往往是有意的上线调参，写清楚差异让用户判断。
10. **输出用中文**，结论先行，给出可核对的证据（文件路径、行号、命令输出），不要只说"已完成"。
11. **能并行就并行，但要知道什么不能并行**。任务可拆成互不依赖的子任务时，用多智能体并行推进，不要串成一条长链：
    - **适合并行**：多个 `openspec/specs/` 条目的交叉比对、多份 `docs/` 报告的检索与归纳、互不相干的独立文件编写、互不争抢同卡的独立评审
    - **不该并行**：同一文件的改动（写冲突）、三份镜像构建（共享 build cache 与磁盘 IO，实测并行反而更慢）、需要前一步结论才能决定的验证
    - **并行前先划分资源**：NPU 卡、端口、临时目录、输出文件都要明确归属，不能两个 agent 抢同一张卡或同一个 `asr_service.log`
    - **汇总由主 agent 负责**：子 agent 的结论必须自己核对过（读文件、看命令输出）才能写进最终答复，不做纯转述

---

## 10. 常见坑


| 坑 | 说明 |
|---|---|
| CANN 环境脚本 | `set_env.sh` 部分变量未定义；在 `set +u` 下 source，随后显式使用 asr 解释器。 |
| CANN 编译器依赖 attrs | 包名是 `attrs`，导入模块是 `attr`；预检缺失必须明确报错。 |
| 容器内设备编号从 0 重新映射 | `ASCEND_VISIBLE_DEVICES=7` 是宿主物理 7，单卡容器内配置逻辑 `npu:0`。 |
| FP32不等于默认内部精度 | 原生FP32进程一次禁用conv/matmul HF32；参数dtype为FP32本身不足以保证数值参考一致。 |
| NPU context 默认卡 | 工作线程先 set_device 目标卡，再构造 device context，避免初始化/恢复到卡 0。 |
| VAD声明不等于实际选项 | nested传入值未覆盖当前权重model_conf；FSMN实际8000/800ms、noise0.6、global decibel/snr均-100。服务段级SNR18/-40独立生效；见课堂阈值评审，不借迁移改参数绑定。 |
| `test/` vs `tests/` | 前者是实验脚本，后者才是 pytest。 |
| `config.toml` 与文档默认值不一致 | 上线后经过实测调参，历史文档里留下的默认值（如 `after_start/before_end` 写过 3/3、6/6、0/0）都是调参过程的快照。**以 `config.toml` 为准**，同时记得回写 README。 |
| `vad_snr_thres` 的定位 | 当前取 `18.0`，**它是段级过滤的主要承担者**（历史统计里 665 次丢弃中 614 次是 snr 命中）。不要因为早期报告说"SNR 不承担主要过滤"就把它调回去。 |
| `after_start` / `before_end` 是**过滤窗口**，不是"处理范围" | 默认 **3/3** 的含义是：只有开头 3 分钟、结尾 3 分钟内的段才做三重过滤，**中间的段一律不过滤、直接保留**（`core/vad.py::_in_filter_scope`）。极易被误读成"只处理前 3 分钟"。出现过 6/6、0/0 都是试参数的过程，不是新基线；改动需重启生效。 |
| Qwen 原始输出 | 保留 `language X<asr_text>` 给既有解析器；浮点输入转模型 dtype，整数索引保持原 dtype。 |
| dBFS ≠ FunASR 的 log 能量 | `vad_decibel_thres` 是 FunASR 帧级 log 能量，不是 dBFS；段级 RMS 才是 dBFS。单位混淆会导致阈值差 40 个数量级。 |
| HBM 与实例数相关 | `speech_campplus_devices` 数组每多一个元素就多一份权重。`instance_count` 同理。 |
| 共享卡测量 | 卡 7 有既有 uvicorn；分开记录本项目进程 HBM 与设备聚合占用，不把既有服务计为本项目。 |

---

> **本文件的维护**：新增能力、修改接口字段、调整默认配置时，同步更新本文件对应章节。
> 发现本文件与代码不符，先改本文件再改代码注释——不要让第三个人再踩一次。


### v6状态资源采样约定（2026-11-18）

- 新`[status].resource_sample_interval_s`缺省1.0秒，严格0.1～60有限数（拒bool/string/NaN/Inf），修改需重启。按需TTL singleflight，不维护无人查询的永久NPU轮询；每app/loop独立拥有者。
- GET实时计数不等待重资源collector；缓存首次unknown，旧值带`resourceDiagnostics.sampling`的age/in_progress/last_error和last attempt起止/elapsed。elapsed含线程派发，非核时；cache age与失败attempt耗时分别记录。
- shutdown入口先发布sampler closing，与模型收尾并行，finally等待实际线程结束；重复取消不丢在途任务。采样后台线程只读OS，模型/准入可变状态仍主事件循环获取。
- Qwen独立worker的HBM按PID及出生身份核对，backend/Qwen/external/unverified分列。父进程0不是worker0；资源快照age与worker allocator的最后RPC口径不能混为实时值。
- 实机已发现旧GET约2秒且诊断读取5秒超时：两轮准入未通过均保留，新增本change任务2.8；v5源码695/保护696及NPU模型验证原证据不改写，v6状态改动须重新CPU/保护/实机及完整矩阵验收。


### 吞吐变更v16的状态历史窗口（apply中）

`/get_status`的Qwen `batch_scheduler.recent_batches`改为最近8条完整记录；保留全部成员/attempt/shape/计时，`history`明确可用/返回ID范围、截断和缺口提示，实时计数不截断。直接客户端/调度器`snapshot()`默认仍最多64条。HTTP直接渲染隔离的JSON数据，避免完整历史重复递归编码阻塞推理调度；这是有界快照，不是无损日志，审计必须去重并检测采样遗漏。CPU候选开销降低不等于完整音频性能已验收，仍须实机420秒闸门和最终矩阵；模型精度、生成上限及v3业务响应未改。


### 吞吐变更v17：内存WAV补位（apply中）

段级编码增加经原SoundFile逐字节验证的PCM16格点快速路径：仅原生一维native float32、有限[-1,1]且值落在16bit格点、采样率合法时直接序列化同一PCM16 WAV；其他输入保留原SF分支，不改变CAM/emo原PCM。128真实段及全部65536格点须源/实际so等价。CPU真实裁剪ABBA支持100ms组批等待候选，50ms仍碎；假模型结果不代表NPU速度，正式默认仍待完整NPU矩阵。NumPy/bytes临时工作buffer与pending PCM预算分开统计。模型仍原HF/BF16与原生FP32 torch_npu，不改模型格式/依赖或生成上限。


### v18 Qwen多owner条件验证（当前apply）

v17单份8B仍420诊断3/8且30秒显式关闭宽限触发inner abort/main1，不能以资源已回收判正常。当前进入process2×batch64、16/128逻辑窗口候选，保持一HTTP和原八VAD/八CAM/八emo/精度/源资产。process1～8配置范围不代表4/8实测；首次全体发现、发现与drain同锁、缓存后健康降级、全部坏4011、原drain后并发CLOSE与每attempt实例身份均须source/实际so测试。关闭候选grace120独立于业务timeout120/测试420，旧失败保留。双份顺序两个真实clip合用420总暖身，必须各index真实forward+PID出生/RPC覆盖；READY不算完成。先小规模同64参考、2×64真实并行/数值/HBM/取消关闭，再完整8B和最终18；正式默认未推广。

### 2026-11-19必要代码阶段提交

用户要求保存并推送当前必要实现。清单仍41/57，未归档、不推广候选默认；当前源码隔离完整CPU968项/281子测试通过、2保护专属skip、1硬件探针排除，237.17秒；源码/profile稳定、owned归零、共享cache写删尝试0。双份NPU及最终1/2/4/8完整音频矩阵待验收，v17的420秒/3of8及关闭main1失败保留。当前权威检查点为docs/npu-throughput-20261118/v18-precommit-checkpoint.json，历史章节只代表各冻结版本。


### 单Qwen启动预加载及128段测试配置（2026-11-19）

`[qwen_asr].preload_on_startup`为严格bool，缺省及根/容器配置均为false以保留原首次使用加载行为。true在VAD启动后、HTTP服务就绪前加载全部配置Qwen权重与processor并常驻；显示model名称不替代加载，失败/取消同样回收。该开关不执行语音generate，不承诺所有shape已暖身；不是在线换权重或免重启更新配置。预加载主要移除首请求的模型加载耗时，不能据此宣称暖态吞吐更快。

用户选择保持根配置，单份128独立候选在`test/config.qwen128.toml`（总8/B4）与`test/config.qwen128-b8.toml`（仅B8准入实验）。二者启用预加载、process1/BF16@npu7、每音频16／全局128逻辑段、物理批最多128、100ms有限等待、115200帧/80000估计token、256段/256MiB预算、显式关闭120秒；VAD八CPU、CAM八NPU process/emo八NPU thread、预取0且实际batch1。最短0.3秒及三过滤false/-100/-1/-1按本轮约定。仍未完成完整性能验收，不是新生产默认。

```bash
cd /home/zhangs/asr_nvidia
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
CONFIG_PATH=./test/config.qwen128.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py --host 127.0.0.1 --port 8083 --workers 1
```

`local_instance_count=1`是一份权重；`segment_concurrency=16`是一条完整音频的Qwen段窗口；`request_concurrency=128`是全部音频共享的逻辑在途段窗口；`max_inference_batch_size=128`是一次物理批上限，仍受就绪段数/长度/帧/token预算限制。完整HTTP音频并发为1/2/4/8，总容量仍8、B4或显式B8，不能用128并发上传完整音频代替段批处理。请求示例与客户端见test/中的使用说明。关闭使用本次父进程TERM并等待子进程/端口释放，勿接管已有服务。

纯Qwen历史固定同128段曲线中，batch64/128处理总量的三轮稳态中位约5.232/3.871秒，吞吐仍增加35.2%；128是最高已测档，不是已证最优。当前物理批校验上限128，未执行>128实验；不能推断拐点已在128之外，也不能拿该组件成绩替代完整A8/B8。

- 2026-11-19本补充已验证：当前源码CPU1001/281子测试通过，预加载NPU7 load-only与exit0/port5关闭通过；清单43/59。test/请求客户端已提供，未执行完整A8请求，不将load-only当推理吞吐验收。证据见docs/npu-throughput-20261118/preload-request-validation-20261119.json。


### 单Qwen八音频48／384段候选（2026-11-19）

参数已在`config.toml`与`docker/config.toml`显式列出，配置重启后生效。依用户此前选择，根配置保留1／1、物理batch1、预加载false及原小模型池行为；384只在独立候选`test/config.qwen384.toml`和`test/config.qwen384-b8.toml`启用。128测试配置仍保留。

| [qwen_asr]参数 | 384候选值 | 含义 |
| --- | ---: | --- |
| local_instance_count | 1 | 常驻Qwen权重副本数 |
| preload_on_startup | true | 权重/processor就绪后才开放HTTP，不做生成暖身 |
| segment_concurrency | 48 | 每条完整音频在途Qwen段窗口 |
| request_concurrency | 384 | 所有音频共享逻辑段窗口 |
| max_inference_batch_size | 384 | 每次物理批最多384，校验范围1～384；不开放512 |
| batch_wait_timeout_ms | 100 | 等待有限，到期发已有段，不强凑384 |
| batch_max_duration_ratio | 32 | 按最长/最短有效特征帧限制同批跨度 |
| max_batch_audio_frames | 345600 | 按每成员900padding帧×384估计 |
| max_batch_tokens | 240000 | 按(ceil(900/8)+256+256)×384估计 |
| pending_segment_limit | 512 | 独立活动PCM段数预算，旧256会先反压384供给 |
| pending_audio_bytes_limit | 268435456 | 256MiB原PCM预算，仍可能先反压；不含模型/前端/IPC临时内存 |
| audio_concurrency | 8 | A+B完整音频准入总上限 |
| audio_spk_emo_concurrency | 4或独立8 | B4默认保留；纯B八路使用-b8候选 |

八条各48为逻辑上限384；两条最多96、四条最多192，不等待凑满。B4纯B只准四条、最多192逻辑段，第5B仍4017；总8满时原4013优先级不变。四A四B也可供384，真实批仍取决于各音频VAD就绪、切片/WAV供给、长短/帧/token/PCM预算与增强汇合，不保证总是满384。VAD八CPU、CAM八process/emo八thread继续各自独立排队、batch1/预取0，不因Qwen384增加NN实例或跳过分支。

在远程项目根目录加载CANN后启动测试候选：

```bash
CONFIG_PATH=./test/config.qwen384.toml OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/zhangs/conda-envs/asr/bin/python main.py --host 127.0.0.1 --port 8083 --workers 1
# 八条纯B另用CONFIG_PATH=./test/config.qwen384-b8.toml；不会自动扩大根配置B4。
```

请求客户端仍用`test/v3_request.py --audio-dir test --limit 8 --mode A --concurrency 8`，其并发8是完整HTTP音频，不是384个HTTP请求。客户端不拥有服务；测试服务启动者须正常TERM并确认worker/端口关闭，不杀既有服务。

已有纯Qwen组件384是固定原128段各3份：中位7.415秒、同期128×3为11.389秒，峰allocated21.69GiB，27/384原始输出SHA不同，未验准确率。512首次暖身在Conv2D分配额外11.72GiB时OOM；该结论限当时共享卡/缓存条件。384在八CAM/八emo共驻和八完整音频上的稳定性、OOM、输出质量及吞吐尚未验收，不能以纯组件成绩推广正式默认。基于100ms/900帧的预算是供给/拆批门槛，不能当作HBM容量保证。

- 2026-11-19配置48/384补充完成：根/容器显式释放预算但旧值保持；test/config.qwen384{,-b8}.toml为候选，当前源码CPU1023/281子测试通过，正式source client/process实际四轮1×384（八owner各48）与正常关闭/parent后全birth退出通过，中位8.521633秒。44/60，完整/B共驻/保护仍待测。音频已由用户迁移为test/test_audio，引用8源SHA一致，仅临时清单刷新路径，原件与失败保留。证据docs/npu-throughput-20261118/config384-validation-20261119.json。

### 910B 依赖清单整理（2026-11-19）

运行依赖统一在根目录 `requirements.txt`，原四份运行子清单已合并；版本和依赖集合不变。开发/构建清单保留，完整 Conda/pip 参考锁移至 `environment/910b/`，重建脚本同步引用。av/pynini 仍由 Conda 提供；历史报告与模型原始文件不改。本次仅整理交付文件，不修改业务行为、配置字段和现用环境，不代表重建环境、镜像或 NPU 验收。

### 根配置同步384（2026-11-19）

按用户明确要求，根 `config.toml` 已原样同步 `test/config.qwen384.toml`：单 Qwen process、预加载 true、每音频48/全局384、物理批384、八CPU VAD、八CAM process/八emo thread、关闭三项能量过滤，总8/B4。修改需重启生效；此次未重启服务。此前“根配置保留1/1、batch1、预加载false”的描述仅代表同步前状态；容器模板未改。此配置选择不表示完整八音频/增强共驻性能已验收，现有验收缺口仍保留。

### 无效VAD配置清理（2026-11-19）

已删除 `vad_max_single_segment_time_ms`、`vad_max_end_silence_time_ms`、`vad_speech_noise_thres`、`vad_decibel_thres` 及对应Settings正名/别名和默认nested转发；它们不会覆盖FSMN真实构造选项。模型原始配置与有效选项不变，不将删除操作当作阈值调整。`vad_snr_thres`仍用于服务段级过滤，独立 `vad_segment_decibel_thres`、RMS、开关及范围参数均有效并保留。复读参数移至实际读取的 `[qwen_asr]`，此前误放 `[vad]` 的值恰与有效缺省一致，修正不改变当前有效值。历史章节中的旧参数仅代表清理前版本。当前服务未重启。

### 2026-11-20 连续生成实施中

新change `2026-11-20-qwen-continuous-segment-scheduling`保持单Qwen HF/BF16@npu7。generation_mode=legacy缺省保留静态批；continuous候选仅process单实例，online_repetition_enabled缺省false；test/config.qwen-continuous128.toml为8/16/128验证入口，未验收不推广根配置。每段独立结束及KV安全清理后补位；真实短长交错与完整v3八并发420秒为必要验收，离线128段不替代。当前进度以该change tasks及docs/qwen-continuous-20261120为准。

### 2026-11-21 用户确认段级恢复

continuous模式对三类段生成异常做保守恢复/无法恢复警告并继续其它段，不再仅因该段整音频4012。v3新增processing_status和segment_warnings，保留所有原字段/类型；全部不可恢复也须显式警告，不能伪装4008。设备故障、超时及取消仍失败。完成率与严格无警告成功率/可用文本分列。原0/8与强制关闭证据保留，新验证以本change tasks为准，不提前推广或归档。

### 2026-11-21 连续调度最终独立验证检查点

段级恢复已按用户选择实现；最后独立真实A8/16/128仅2/8在420秒内完成（389.08/411.05秒），6条删失，n2子集P50/P95=400.07/409.95秒。全部owned正常退出、端口释放5秒、原PID1601397保留。源码1135项隔离回归及保护151项通过，独立fatal-release清理修复另做增量验证。文本仅一完整配对可比，人工质量真值缺失，不能推广。当前20/25保持open，不同步主规格、不归档；证据docs/qwen-continuous-20261120/REPORT.md。


### 2026-11-24正式多实例实施中

用户已apply `2026-11-23-qwen-multi-instance-serving`。正式默认正切换continuous/process、N6/S16、宿主npu7、预加载true；audio=N/global=N×S派生、B4，HTTP保持1。在线复读保护true/运输交接3ms沿用已测noHBM参考，阈值/生成上限不改。增强池独立各1起标定，尚未通过正式NPU验收。此前单实例/384/单步缺省描述仅属历史检查点。实施证据目录`/tmp/qwen-multi-apply-20261124-sduu4fzl`；配置/运行时/增强与控制器唯一作者见setup.json。完整guard CPU及保护回归后才root独占NPU7真实A6/A4/B4/共驻A6。336/真实故障停测；用户已取消8GiBfree提前停止，不恢复或换阈值。保留shared PID1601397。旧change/成绩不改写；本change未必需验收完成不归档。

### 2026-11-24正式多实例验收检查点

实现与测试交付27/29，change保持open：纯A6 6/6、P50/P95=173.24/185.42秒、墙钟186.32秒；A4 4/4、墙钟200.93秒；B4 4/4、P50/P95=289.81/301.37秒、墙钟302.60秒。严格无警告A6 4/6、B4 1/4，不能当准确率认证。CAM真实2775次、emo2226次加549时长skip；CAM持续排队/PCM反压。增强驻留后的A6实际3完成/1失败/2取消，在179.95秒停止，owner5申请22MiB失败、dev7错误207001，真实OOM，整批分位数未知。按test-plan停止后续压力，5.3最终增强池标定与5.6共驻验收未通过，不归档、不改主规格或旧change。

源码1289/379subtests、保护364/244subtests通过；日志隔离与netns增量两边各84及40通过。所有测试/模型owned身份正常exit0、TERM后43.14秒父进程退出、端口空闲≥5秒、PCM为空，共享PID1601397及8007保留。根配置测试后仅补注释，解析值与实测快照完全相同；Docker未构建/验收。报告docs/qwen-multi-instance-20261124/REPORT.md，原始formal-v2和所有失败证据保留。没有继续CAM2/4、修改allocator或新增HBM提前停止线。

### 2026-11-24用户授权N5补充验证

根配置副本仅N6→5，正式默认和业务代码保持不变。原完整a1～a5纯A5 5/5，P50/P95/墙钟173.70/183.18/183.83秒；A4 4/4墙钟195.44秒；双增强B4 4/4，284.91/296.24/296.96秒；增强驻留A5 5/5，173.32/203.23/206.12秒，严格200目标超约3.06%。本轮未复现OOM，最低HBM采样free5131原始MB（约5.01GiB），CAM排队/PCM反压及allocator缓存增长仍在。正常exit0、owned/端口/PCM回收、共享1601397/8007保留；N5不能替代原N6必需验收，change仍27/29/open。详见docs/qwen-multi-instance-20261124/N5-REPORT.md。

### 2026-11-24用户最终配置（验收前检查点）

用户明确采用Qwen4/CAM++4/emo1：loader continuous缺省owner数4，根与Docker模板Qwen process/preload=true、S16，派生audio4/global64；CAM++设备数组4份且process，emo数组1份，宿主物理npu7/容器逻辑npu0。VAD、生成策略、每owner预算、阈值及HTTP1保持原配置。本轮B4与增强驻留A4及最终完整guard/保护待root验收，未提前宣布通过或归档。此前N6真实OOM与N5超200秒等历史段落保留原冻结语境，不能作为当前配置成绩；Docker仍待构建/验收。

### 2026-11-24最终组合验收完成

当前正式默认continuous/process Qwen4@npu7、CAM4 process、emo1、S16，总音频4/global64，HTTP1。用户已确认替代最终范围，5.3/5.6通过，29/29：正式A4/B4/增强驻留A4均4/4，墙钟186.44/200.10/183.47秒；B4 P50/P95=183.26/199.45秒、严格无警告2/4（12warning/11discard），冷加载短clip暖身105.21秒单列。最低HBM采样free15406MB；CAM等待p50/p95/max0/1/25，PCM峰65，无推理OOM。source1299/379subtests、实际40SO保护373/244subtests、归档清单兼容源49/保护49通过。所有CPU/NPU/控制器退出，service正常exit0/34.89秒、端口≥5秒、PCM为空、shared1601397/8007保留。背景used15908→15910MB仅+2，root独立相近背景/归属审查通过，raw recovery unknown仍保留。

本change完成后仅同步其六份delta并归档；旧N6共驻OOM、N5补测及三个旧open change任务不改。没有更改KV/allocator/阈值/生成策略或依赖/权重，没有Docker构建/上线，质量未认证。报告docs/qwen-multi-instance-20261124/FINAL4-REPORT.md，实际大原始日志final-m4-cam4-emo1-v1保留；上方“待验收/27of29/正式默认6”仅属此前历史检查点。测试控制器稳定清单在docs/.../test-inputs.json，active change归档不再使CLI路径失效。

2026-11-24：`2026-11-23-qwen-multi-instance-serving`已29/29完成，六能力主规格已同步并严格校验，归档于`openspec/changes/archive/2026-11-23-qwen-multi-instance-serving`。当前正式默认Qwen4/CAM4/emo1；旧N6 OOM/N5结果和三个旧open change保留，不外推长期稳定/人工质量或Docker上线。所有测试服务已释放。


### 增强模型启动预加载（2026-11-24）

功能开启且设备数组非空的 CAM++、emotion2vec 在 HTTP 就绪前，按数组顺序完成全部实例的加载尝试；当前配置为 CAM4/emo1。关闭功能或空数组跳过，部分失败使用健康实例，某池全部失败仍允许其他能力服务，所需能力请求沿用原不可用错误。状态显示实际健康数及失败原因，GET 不触发加载；请求开关仅决定本次推理分支，后续请求复用已加载实例。启动耗时及权重内存占用提前发生，需重启服务生效；此步骤不执行语音 forward 暖身，首个推理仍可能有编译开销。无需新增配置或修改设备/实例数。历史懒加载描述代表此前版本。


### 2026-11-24 异常复读折叠恢复

continuous已确认的异常尾部循环保留前文及一次完整循环，不再因恢复文本不足8汉字/4单词丢弃。只折叠已确认尾部，纯填充词/空正文/无效前缀和非复读length_limit仍沿原策略；正常EOS、在线检测阈值、设备/取消契约不改。折叠成功仍是completed_with_warnings，action=repetition_collapsed、recovered=true，原段index/bg/ed保留，不视为人工准确率通过。周期回溯须同时保护真实前文标点/词界。报告docs/qwen-repetition-recovery-20261124/REPORT.md，根配置只读审计CONFIG-AUDIT.md；当前continuous专用模板可精简4项不等于相关实现全无用途，不擅自删除queue/VAD过滤开关。本轮不覆盖历史课程JSON、不宣称原2.02秒段已恢复。


### 增强能力设备数组统一配置（2026-11-24）

CAM++、emotion2vec启用状态现在只来自各自设备数组：非空数组开启，每元素一实例；`[]`关闭。数组缺省为空。`showSpk`/`showRoleIdentify`选择CAM++执行，`showEmotion`选择emo执行，均不能开启空数组能力。模型在HTTP就绪前按非空数组预加载，请求复用健康实例；A请求不执行增强，已加载权重继续驻留。预加载不执行语音forward暖身。

旧 `[features].open_v2_role`、`open_v2_emotion` 已删除，残留无论true/false都会在任何worker/模型创建前报迁移错误。迁移办法：原true删除旧键并保留设备数组；原false删除旧键，并把对应的 `speech_campplus_devices` 或 `emotion2vec_devices` 改为 `[]`。删除空 `[features]` 表。不允许只删原false键却保留非空数组，以免开启原已关闭的模型。

单项关闭请求仍在音频准入前返回4015/4016，双请求且两项均关闭返回4018，优先于容量错误，拒绝计数不变。非空数组全部加载失败是模型故障，不是配置关闭，不返回4018。数组改变需重启源码服务；保护构建和镜像在后续构建纳入。本次不构建上线；三个镜像内置小模型的必需资产预检保持，空数组不表示可以删除权重。

正式continuous根/Docker模板省略两项legacy组批参数和两个固定增强batch配置；legacy显式参数支持、实际batch1及非1校验保持。两个 `prefetch_queue_depth=0`、全部 `online_repetition_*` / `repetition_*`、`retry_count`、队列及VAD参数保持；预取0是在上游反压等待，不是无限预取。早先章节中涉及旧功能开关的已日期标记说明仅代表当时版本，以本节为当前规则。

### CANN8.2源码容器单卡可见性

源码容器同时使用ASCEND_VISIBLE_DEVICES=7（runtime设备选择）与ASCEND_RT_VISIBLE_DEVICES=7（CANN进程可见性），配置写逻辑npu:0。实测特权模式仅前者仍可能看到8卡；预检在放置权重/创建worker前要求仅1卡，错误映射不能带病启动。宿主原生npu:7与多卡检查保持原义。

### 2026-11-25 CANN8.2源码容器交付完成

2026-11-24-docker-source-cann82已完成真实构建、最终CPU1419/379subtests及工具82、ABI/原生/NPU物理7核验、双增强完整B4 4/4（P50/P95/墙钟182.44/196.25/198.65s，严格无警告2/4；16warning=9恢复/7丢弃）。最终asr_offline_910b:v3_20261125，ID ef7b8add6a42c0297dc66eaa58718a67df8fba688d1738f1c314fe59b8565c5a；cold HTTP约416～419s，clip暖身30.20s单列。正常parent0、VAD8正常、出生PID/端口/PCM释放，共享服务保持；本轮中间候选/helper/builder/registry/cache/staging已清，原11镜像/14容器保留。Docker严格四文件、无Nginx/Cython，config必需只读挂载。报告docs/docker-source-cann82-20261125/REPORT.md。该短时可用性不等于零丢段/人工准确率/长期生产认证。按流程同步本change四delta并独立归档；旧open change历史不改。

### 2026-11-25 ITN缺省关闭

源码Settings.itn_enabled及ItnConfig.enabled缺省false，根/Docker模板[itn].enabled=false；显式true仍可开启数字规范化，关闭不影响独立标点卫生及语气词清理。/data/asr_offline_910b部署配置已false并已重建容器确认healthy。本次仅同步源码缺省及模板，现有已构建镜像不因源码提交自动更新。

上述ITN缺省关闭回归：77passed/15subtests，CPU隔离无NPU/共享缓存写入或owned遗留；证据logs/itn-default-off-20261125-110151。
