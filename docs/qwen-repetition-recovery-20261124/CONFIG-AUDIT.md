# 根配置只读审计（85bd900）

范围：xjtu910b7:/home/zhangs/asr_nvidia/config.toml，80个显式键；当前continuous/process、N4/S16、CAM4/emo1预加载。未编辑项目文件、未启动任何模型、未改Git。

## 可优先从当前模板精简的4项

| 配置 | config.toml行 | 实际使用/可删理由 | 限制 |
|---|---:|---|---|
| qwen_asr.batch_wait_timeout_ms=100 | 72 | core/qwen_asr_client.py132–152先构造legacy BatchScheduler(137–138读取wait)，continuous随即替换成ContinuousScheduler；仅audio_frame_budget/token_budget转移，不转移wait | legacy仍生效，删键后默认10；不是删除全局支持 |
| qwen_asr.batch_max_duration_ratio=32.0 | 52 | 同上143–144；continuous调度器不接此参数 | legacy仍生效，删键后默认4；当前源码仍校验1–64 |
| model_device_pool.speech_campplus_batch_size=1 | 132 | core/config.py573–576校验范围1～1；686–687同；模型池不读取此getter，core/models.py616 actual_batch_size固定1 | 属固定值伪配置，删TOML仍按缺省1校验，并非移除校验代码 |
| model_device_pool.emotion2vec_batch_size=1 | 133 | 同上，core/config.py690–691 | 同上 |

## 仅continuous下不控制推理，但不能称完全未引用

qwen_asr.retry_count=2（config.toml66）：core/qwen_asr_client.py64记录，557–600只在legacy _execute_batch使用。continuous _execute_sequence458–506仅对accepted=False/capacity_rejected重试，固定range(4)即原尝试+3重试，30s内，delay0.1/0.5/1.0，不读取retry_count。api/routes/asr_v3.py161仍将settings.qwen_retry_count写入请求遥测。若仅删TOML，默认2→1会改变遥测标签；应把它列为当前推理无效/易误解项，并说明清理遥测语义需要另行改代码。当前复读异常不是由该项控制重试。

## 当前关闭或部署模式下不影响处理，但不是死配置

- qwen_asr.queue_max_size=64（63）：core/qwen_asr_queue.py258先检查wait_timeout_s<=0并拒绝，265才检查等待队列容量。现在queue_wait_timeout_s=0所以此容量不参与准入。开启正等待后会生效；删键缺省仍64。
- qwen_asr.queue_wait_timeout_s=0（64）：有效，决定满载立即拒绝而非等待，不能列入无效配置；读取queue.py368，执行258–269。
- vad.vad_snr_thres、vad_segment_decibel_thres、vad_decibel_frame_sample_ratio、vad_decibel_low_frame_ratio_thres、vad_low_volume_rms_dbfs_thres、after_start、before_end：三项filter_enabled当前false（99/105/109），core/vad.py122–129没有进入段级过滤；138–157才使用scope及采样参数。功能开关false必须保留（删除默认true会重新开启）。阈值可暂时省略，但未来开启会按缺省回填；尤其after_start/before_end删除后从-1/-1变缺省3/3，因此不是一般意义的行为等价清理。
- 顶层instance_count=1（3）：宿主直接main.py启动不读取；docker/start.sh27–41、111等实际控制HTTP服务进程数量，不能因本次宿主启动未用就删实现。与qwen_asr.local_instance_count不同。

## 看似冗余但有实际用途，应保留

- model=""（54）：core/qwen_asr_client.py61、230、339–340、378、398；可选显示名，空值回退模型目录名。删空键与当前等价，属于省略默认值，不是死字段。
- enabled=true（53）：api/routes/asr_v3.py47拒绝关闭能力，main.py102预加载条件，deployment.py36环境预检。有效。
- 两增强prefetch_queue_depth=0（130/131）：core/models.py912/919传getter，447、450–451计算等待容量，697控制prefetch排队；0表示即时交接/反压，不是未生效。
- max_batch_audio_frames/max_batch_tokens：client96–99用于continuous worker，同时149–152用于continuous scheduler。不能因名称含batch误删。
- repetition_*后处理、online_repetition_*生成中检测分别消费，不能把二者看成同一套冗余字段。
- [itn]五项经core/config.py142–152构造ItnConfig，utils/itn.py125–127传normalizer，api/routes/asr_qwen_common.py903消费。
- speech_rate.rate_factor经api/routes/asr_qwen_common.py919参与语速计算。
- features.open_v2_role/open_v2_emotion虽名字含v2，但core/models.py911/918实际控制v3增强池和预加载，不是下线v2残留。

## 总结

没有发现根TOML完全不读取的字段。推荐清单4项可从continuous专用模板先精简；retry_count另列当前推理无效但遥测仍消费；queue_max_size和7个VAD阈值/scope是条件关闭；instance_count是Docker入口专用。删除显式键后Settings多数仍回填默认，不能宣称相关功能/代码已删除。
