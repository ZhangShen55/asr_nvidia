# NPU 迁移测试覆盖矩阵

本表对应 change `2026-11-17-npu-910b-local-qwen-asr` 的 0、2、3、5、6、9、10、11 节任务。它记录**已落地的测试及其证据边界**，不是以测试数量代替实机验收。实施日期为2026-11-17，2026-11-18更新完整回归与课堂证据；最终以验证报告中的冻结源码回归、实机结果和服务关闭记录为准。

## 证据分层与执行记录

| 层次 | 含义 | 能证明与不能证明 |
| --- | --- | --- |
| C1 | CPU 上的替身、受控异常、静态配置/脚本契约 | 能验证协议、错误归属、调度和调用边界；不能证明 NPU 算子可执行、实际落卡或模型质量。 |
| C2 | 实际 CPU 数学、线程、spawn/IPC、临时文件、CPU 子服务 | 能验证真实等待/取消/进程和文件收尾、CPU 数学参考；线程事件模拟的设备完成仍不是 NPU 完成。 |
| N1 | 本次在 `xjtu910b7` 的物理 `npu:7` 上实际运行的原始记录 | 仅证明该源码/输入/配置/时间范围内的实测；卡 7 有既有服务，不能称独占性能。 |
| 待确认 | 工装已实现，但未执行、尚在运行或缺少最终关闭证据 | 不记为通过，不据此勾选对应验收任务。 |

CPU 回归解释器为 `/home/zhangs/conda-envs/asr/bin/python`，替身测试使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 隔离设备自动初始化。典型复算命令为：

```bash
cd /home/zhangs/asr_nvidia
TORCH_DEVICE_BACKEND_AUTOLOAD=0 /home/zhangs/conda-envs/asr/bin/python -m pytest tests/ -q --disable-warnings
```

最终1/1默认配置的完整CPU回归为 **449 passed、146 subtests passed、2 warnings，85.18秒**。日志`/tmp/asr-apply-20261117-a99trpfv/pytest-final-g1-default-20261118-b25uqnne.log` SHA256=`093f20f359c0c9ff11d878d40f83be638e75b0e3c0dfb8fa6920c4ab0ae2c3b0`；[cpu-regression-final-g1-default.json](cpu-regression-final-g1-default.json) SHA256=`16d8976b167e3970c494fa160643c83347a0755bb0a23c880bcbd9bf94db0c9f`，记录exit0、无剩余自有子进程/无强制回收。该次覆盖main关闭失败退出码修复、3项真实CPU入口测试、18项wrapper测试以及1/1 getter/host/docker配置断言。此前mainexit版本449/146、84.47秒的日志SHA=`a6125ae5cc6698beeffc552ab11f9a6cf073f9a3e87a50d5f9248bf59a730f47`保留作历史，不拿旧配置结果覆盖新默认。

Docker测试L116仅去掉17个空格，AST指纹前后相同且已被新全量回归覆盖。最终编译版本另外完成实际NPU API/稳态与关闭验收，见下表；不以CPU计数替代硬件结论。

历史阶段日志`current-unit-3.log`的366 passed/138 subtests、生命周期3 passed、VAD两文件+生命周期26 passed/2 subtests及各CPU子集均是过程记录；与449有重叠，**不得相加当作独立用例总数**。同样不能用任何CPU计数替代编译版本NPU API/稳态验收。

全局dtype竞态修复及CPU前端保护边界的最新相关回归为 **60 passed、30 subtests passed（16.48秒，1 warning）**：`test_native_loading_dtype.py`、`test_model_loading.py`、`test_inference_backends.py`、`test_npu_pooling.py`、`test_model_pool.py`、`test_model_pool_route.py`、`test_audio_admission_lifecycle.py`。这包含最新原生9个case、共享guard/Qwen前端10个case，使用上述CPU隔离环境执行，没有加载NPU或启动接口。

## 按测试文件对应任务

下表文件均位于 `tests/`。保留的 `v2`、`v118`、`tensorrt` 文件名是历史命名，不代表仍运行旧公开接口或 TensorRT 后端。

| 测试文件 | 关键 case / 验证内容 | 层次 | 对应任务 |
| --- | --- | --- | --- |
| `test_v3_contract_snapshot.py` | `test_success_body_matches_frozen_business_snapshot`、八种增强组合、4008/4009/4011/4012 固定响应；受控时钟保持 `gpu_time_ms` 原主链路起止，`bg/ed` 两位小数字符串及计时字符串类型 | C1 | 0.1；2.3/2.4/2.6；6.4；11.1 |
| `test_v3_route.py` | 只注册 v3，旧 v2 不再公开，OpenAPI 增强默认 false | C1 | 0.1；6.4；11.1 |
| `test_request_defaults.py` | 八种开关分类、表单与内部默认值一致 | C1 | 0.1；2.8；11.1 |
| `test_qwen_v2.py` | 前缀解析/语种占比/具体语种透传；段级与全局并发独立；聚类及短段继承；损坏片段隔离、CAM/emo 降级、Qwen 4012；保持确定性复读修复 | C1/C2 | 0.1；2.3/2.4/2.6/2.8；3.4/3.5；6.4；11.1/11.2 |
| `test_local_qwen_runtime.py` | BF16 与整数/布尔输入边界；懒加载与状态不加载；进程精度配置先于目标卡 context；单实例串行、等待者取消、重复取消、完成交付边界取消、超时等待原工作结束；多实例隔离及关闭错误不隐藏 | C1/C2 | 2.1–2.8；5.2；6.4；10.9/10.10；11.2/11.6 |
| `test_model_loading.py` | 10个case：共享RLock在读取saved dtype前获取；异常/嵌套恢复；实际HF tiny checkpoint两个并发BF16 from_pretrained、HF构造失败与native真实CPU Linear精确FP32 statecopy；验证HF浮点参数但保留其有意FP32 buffers；Qwen lazy import等待同guard、只保护CPU processor不锁设备generate；实际Whisper/Qwen Hann前端在BF16窗口后逐值匹配FP32参考，int32 mask不降精度 | C1/C2 | 2.1/2.2/2.5；3.2；5.2；10.9；11.2 |
| `test_native_loading_dtype.py` | 9个case：实际HF local_torch_dtype(BF16)窗口阻塞真实FunASR CAMPPlus构造；临时小checkpoint原值逐值保留；显式BF16构造必须statecopy前float；CPUAutoModel工厂的真实Linear/BN同guard；参数/浮点缓冲错dtype拒绝，opaque句柄不伪称FP32；CAM真实Kaldi fbank、emo真实ALiBi factory/cache及CPU AutoModel内部fbank在BF16窗口后逐值匹配FP32参考；NPU形状假体forward不持有全局guard | C1/C2 | 2.1/2.2；3.2/3.4/3.5；5.2；6.2；10.9；11.2 |
| `test_memory_clip_pipeline.py` | 实际 PCM16/mono WAV 字节可解码；客户端接收内存字节且不要求文件路径 | C1/C2 | 0.1；2.2/2.7；6.4；11.1 |
| `test_audio_admission.py` | A/B 容量及同类 FIFO；等待者取消；grant 后取消、截止时间竞争、交付前 timeout rollback；重复取消不跳过 cleanup | C1/C2 | 2.8；10.9/10.13；11.2/11.4 |
| `test_audio_admission_lifecycle.py` | 读取线程、增强推理、兼容裁剪和 VAD 阶段取消/重复取消；实际线程未结束时继续持有音频与模型租约；迟到临时文件仍清除 | C1/C2 | 2.8；10.9；11.2/11.6 |
| `test_audio_admission_startup.py` | 无效容量配置在 VAD/模型加载前失败 | C1 | 0.1；2.8；10.9；11.2/11.6 |
| `test_audio_admission_status.py` | 请求前计数为零；活动与等待采用同一快照；禁用增强不隐式加载 | C1 | 5.1/5.2；10.9；11.4 |
| `test_audio_admission_telemetry.py` | 拒绝原因分开计数，元数据拷贝，遥测窗口有界 | C1 | 5.1/5.5；11.4 |
| `test_v2_audio_admission_route.py` | 能力检查先于容量拒绝；各退出路径恰好一次释放，先清理再释放；等待取消与混合容量恢复 | C1/C2 | 0.1；2.8；6.4；10.9/10.13；11.2/11.6 |
| `test_v2_capabilities.py` | 全部开关矩阵；禁用全局池即使已有模型也不能加载/租用；短情绪路径不能绕过能力检查 | C1 | 0.1；3.5；6.2；10.12；11.1 |
| `test_model_pool.py` | CAM/emo 非对称容量、FIFO 等待及取消、设备数组次序；坏实例隔离、全失败/空数组显式失败 | C1/C2 | 3.1/3.2；6.2；10.9/10.12；11.2 |
| `test_model_pool_route.py` | 业务经池租约使用实际发放设备；speaker/emotion 输出契约及请求级降级保持 | C1 | 0.1；3.2/3.5；6.2；10.12 |
| `test_inference_backends.py` | NPU 不调用 CUDA；CPU/NPU 四种组合；混合池独立；状态查询不加载；NPU 加载失败不回退 CPU；坏实例唤醒等待者；加载及设备工作中取消；close 失败保留所有权、其他实例继续关、可重试且不能复用；重复取消等全部 close 完成 | C1/C2 | 3.1/3.2/3.6；5.2；6.2；10.9/10.12；11.2/11.6 |
| `test_npu_lifecycle_faults.py` | 加载锁等待者取消不影响加载 owner；真实 Runner 的 `.cpu()` 结果读取路径用受控假体阻塞后抛 ACL 式错误；`_condition` 阻塞释放窗口时重复取消仍持有租约，最终 busy/waiting 归零 | C1/C2 | 3.2；10.9；11.2/11.6 |
| `test_npu_pooling.py` | T=1/20/99/100/101/199/200/201/1201 的 avg/max 尾块及全负值 CPU 数学对照；实例局部绑定；set target 先于 context/allocator；HF32 配置进程一次/并发幂等；同步错误转实例故障；精确 GELU 的 schema 参数、CPU 数学、权重和全局对象不变 | C1/C2 | 3.1–3.5；5.3；10.7/10.8/10.9/10.10；11.2/11.3 |
| `test_local_model_runtime.py` | 状态保留既有业务字段并增加 NPU 模型明细；主机/容器配置结构、模型路径和设备映射；状态合约测试隔离真实预检/worker，另有部署预检测试承担其责任 | C1 | 0.1/0.2/0.5；5.1/5.2/5.5；6.2；10.12 |
| `test_vad_pool.py` | 真正 spawn 的 CPU 假模型 worker：固定容量/FIFO/并行；任务错误不杀健康 worker；CPU worker 实际 `os._exit` 崩溃与设备错误替身都按原设备补位；参数透传、进程命名和取消等待 | C1/C2 | 3.2；5.1；6.2；10.9/10.12；11.2/11.6 |
| `test_vad_pool_static.py` | 固定数组、取消排队；晚到 1 MiB IPC 结果需排空，正常 stop 不被迫 terminate；Queue/Process/Event 引用回收；第二个 spawn 失败时已启动子进程也关闭；强制 TERM/KILL 即使回收成功仍失败且其他 worker 继续关闭；pool_stop 非零退出不冒充成功，故障补位的 reap reason 独立记录 | C1/C2 | 6.2；10.9；11.2/11.6 |
| `test_vad.py` | 原有首尾/全段过滤范围；缺少音频 tensor 时先加载再过滤 | C1/C2 | 0.1；6.5；11.1 |
| `test_vad_volume.py` | dBFS/10ms frame、短帧、采样分母、RMS/SNR/能量规则及禁用边界；过滤掉的段不发起下游工作 | C2 | 0.1；6.5；11.1 |
| `test_vad_reference.py` | 三个原来依赖缺失真实文件的 case 改用可追溯 synthetic PCM/标记，另五个算法 case 保留；覆盖 parse、scan 计数、percentile/context | C1/C2 | 0.3；6.5；11.1/11.5 |
| `test_vad_quality_validation.py` | 合成音频/人工构造结构的 overlap、baseline-only、复读扫描及新旧日志解析；不是人工课堂质量验收 | C1/C2 | 0.3；6.5；10.17；11.1 |
| `test_config_contract.py` | 设备数组、空数组、非法值、旧键提示；容量与低音量阈值边界；旧 Qwen VAD 键不覆盖 `[vad]` | C1 | 3.1/3.6；6.2；10.12；11.1 |
| `test_tensorrt_config.py` | 历史四个 TRT 键被忽略/告警；只接受 CPU/NPU、拒绝 CUDA；不再执行 engine 校验 | C1 | 3.1/3.6；6.1；10.12/10.13 |
| `test_docker_profile.py` | CPU 预检不导入 NPU；缺资产先失败；NPU 可见编号用替身核对且不加载 Qwen；旧 CUDA profile 先于硬件失败 | C1 | 0.2/0.5；3.6；6.3；10.9/10.15 |
| `test_npu_environment.py` | 版本 metadata、attrs/CANN 编译器路径、环境重建拒绝覆盖已有前缀；原始 checkpoint/七类顺序边界用工厂替身验证；PyAV CPU 解码/GBK 路径；VAD benchmark 同伴失败收尾 | C1/C2 | 0.2/0.4；3.5；6.5；10.1–10.3/10.11；11.1/11.6 |
| `test_docker_deployment.py` | 18项配置/构建/完整CPU wrapper用例；父PID TERM、nginx QUIT、超时组TERM/KILL且非零；子退出7/信号退出143/未知127均不可洗白；早退0/非0、启动与PID登记窗口、flag-check→sleep窗口、INT后TERM保留130、cleanup期间重复TERM均覆盖；只管理自有进程 | C1/C2 | 0.5；3.6；6.3；10.13–10.15；11.6 |
| `test_main_shutdown_exit.py` | 3项：真实CPU Uvicorn进程正常TERM退出0、lifespan关闭故障必须非零且其他closer仍依序执行；CLI传同一个app对象并恢复TERM handler。随机自有端口最终释放，未初始化NPU；不能用Uvicorn吞异常后的0当成功 | C1/C2 | 6.3；10.9/10.15；11.2/11.6 |
| `test_resource_diagnostics.py` | npu-smi 两行表/进程表准确 HBM；未知不填零；只对物理卡 7 闸门；本 PID/后代归属与他人占用分开；Qwen 嵌入后不访问独立健康端点 | C1/C2 | 0.5；5.3–5.5；10.10；11.4/11.6 |
| `test_npu_resource_status.py` | 无 torch/未初始化 NPU 时不导入、不初始化；已初始化时只采配置目标卡；失败不改采其他卡、不伪造零；缺峰值 API 明示不完整 | C1 | 0.5；5.2/5.3/5.5；10.9/10.10；11.4 |
| `test_status_sampling.py` | 慢诊断在真实线程运行时 asyncio 心跳仍前进；采样异常不丢业务字段；空 snapshot 不触发重采 | C1/C2 | 5.1/5.4/5.5；11.4 |
| `test_model_memory_pressure_diagnostics.py` | 阶段计划、PSS/RSS 斜率与卡 7 AICore；显式 PID 与 HTTP 状态 PID 必须相符；未知 HBM 不升压；GBK 展示与 ASCII 上传名 | C1/C2 | 0.4/0.5；5.3/5.4；10.10/10.17；11.4/11.6 |
| `test_audio_manifest.py` | 递归 SHA/重复源分组、稳定 ID、GBK 字节往返、坏 header 不冒充 decode；无 MP3 probe 明示原因；不写回输入、不跟随软链 | C2 | 0.3/0.4；11.1/11.5 |
| `test_audio_normalization.py` | 合规 PCM 字节不变；CPU 内存重采样保持时长；坏流不伪装成功，无 ffmpeg CLI 依赖 | C2 | 0.1/0.4；2.2；11.1 |
| `test_audio_utils.py` | CPU 立体声转 mono、兼容裁剪写入二维音频 | C2 | 0.1；2.2；11.1 |
| `test_segment_clip_lifecycle.py` | 内存字节上传；兼容片段在上传前删除；转写失败/整体异常仍清理；emo 并行不依赖临时片段 | C1/C2 | 2.2/2.7；6.4；11.2/11.6 |
| `test_cache_lifecycle.py` | 同名上传隔离、重复清理幂等、启动扫描只清服务前缀 | C2 | 0.4；11.4/11.6 |
| `test_validate_npu_models.py` | path_hex 只读片段/SHA、禁止截短、manifest 选择；预登记容差、数值无穷显式记录、七类/Top1 gate；VAD 边界 gate；TorchAir 缺失原因和 plan-only 不加载 | C1/C2 | 0.3/0.4；3.4/3.5；10.7/10.8；11.1/11.3/11.5 |
| `test_benchmark_qwen_candidates.py` | 独立片段、混合语言/长度、分桶尾批/mask/原次序；冷/稳态分开；每段错误不冒充成功；加载中断/close 错误/重复终止信号留证据；硬锁卡 7 | C1/C2 | 0.4/0.5；2.2/2.4；10.4/10.6；11.2/11.3/11.6 |
| `test_qwen_cancellation_tool.py` | 与 NPU 工装同一调度/观察逻辑，真实 CPU 线程分别模拟冷加载、串行等待及关闭；原工作结束后再释放；失败/外部停止/close 错误不隐藏 | C1/C2 | 2.5/2.6/2.8；10.9；11.2/11.6 |
| `test_validate_npu_service.py` | Popen 部分启动失败仍 finally；清理期间首个/重复信号不打断 wait；实际 CPU http.server 释放随机自有端口；TERM/KILL 假体超时即使最终回收也判失败；compact 保留卡 7 他人 PID/RSS/FD | C1/C2 | 0.5；5.3；10.9/10.10；11.2/11.4/11.6 |
| `test_text_hygiene.py` | 前导/纯标点处理、语言简称、占比及其他字段保持 | C1 | 0.1；2.3；6.5；11.1 |
| `test_text_postprocess.py` | 语气词/英文替换边界、元数据不变、语言别名归并 | C1 | 0.1；6.5；11.1 |
| `test_repetition.py` | 时长阈值、复读压缩及正常长句/堆叠词保留，配置禁用/阈值边界 | C1 | 0.1；6.5；11.1 |
| `test_feature_utils.py` | 中英法计词、标点、空串与语速计算 | C1 | 0.1；6.5；11.1 |
| `test_itn.py` | 原有 ITN 确定性行为回归；不作为设备性能或转写质量证据 | C1 | 0.1；6.5；11.1 |
| `test_log_rotation.py`、`test_progress_logging.py` | 原有日志轮转/进度契约保留于全量回归；不作为 NPU 前向证据 | C1/C2 | 6.5；11.5 |

## 故障、取消、重复取消与资源所有权

以下均是已经存在的 CPU 测试。标为 NPU 形状的替身只模拟接口与错误，不会初始化 NPU。错误码断言继续采用既有 Qwen 4011/4012/4009、CAM 4015、emo 4016；不把设备故障转成无语音 4008。

| 阶段 | 已有证据 | 所有权及最终状态断言 | 真机证据边界 |
| --- | --- | --- | --- |
| 等待加载锁 | `test_npu_lifecycle_faults.py::test_cancelled_load_lock_waiter_does_not_cancel_or_duplicate_loading_owner` | 真 CPU 加载 owner 阻塞于线程事件；第二请求等待 `_load_lock` 时取消。owner 未被取消、工厂只调用一次，完成后模型只登记一份并由 close 回收。 | 没有对真实 NPU 加载锁等待窗口做精确故障注入；Qwen runtime 内部 threading.Lock 等待未另行直接争用，生产 client 通过实例队列阻止同实例并发加载。 |
| 加载执行/登记 | `test_inference_backends.py::test_cancel_during_load_registers_and_later_closes_resource`；`test_audio_admission_lifecycle.py::test_cancel_during_context_load_waits_for_thread_then_cleans_partial_file` | 取消不能丢失迟到的模型句柄/文件；已启动的线程结束后完成登记或清除，再传播取消。 | 实机 `qwen-cancellation.json` 已证明真实冷加载执行中的重复取消/最终关闭；这不等同直接争用内部加载锁。 |
| 并发构造的全局 dtype | `test_model_loading.py` 的实际HF tiny checkpoint并发/异常；`test_native_loading_dtype.py` 的真实CAMPPlus构造、精确statecopy、CPUAutoModel | HF BF16初始化窗口受进程共享RLock保护；native构造明确FP32并在载权重前float，结束恢复原全局dtype；参数/浮点缓冲的实际dtype校验后才报precision。实际线程争用不再把FP32 checkpoint先舍入为BF16；HF有意FP32缓冲不被强制降精度。 | 本行单测为真HF/tiny checkpoint与真实CPU CAMPPlus，未运行NPU。另有新源码`e2e-cold-mixed-i2-fixed`冷A+B通过，见实机表；旧串行模型对比不能冒充此并发路径通过。 |
| 已加载模型的CPU前端/临时常量 | `test_native_loading_dtype.py::test_cam_fbank_waits_for_hf_window_and_matches_original_fp32_features`、`test_emotion_bias_factory_cache_is_instance_local_and_cannot_round_in_bf16_window`、`test_cpu_automodel_generate_keeps_its_internal_fbank_fp32`；`test_model_loading.py` 的实际Qwen前端/Hann case及lazy import case | 仅把隐式采用default dtype的CPU fbank、processor与实例ALiBi factory/cache纳入同一FP32 guard；等待BF16窗口后精确匹配原FP32输出，避免“先BF16舍入再cast为FP32”的假精度。emo适配是实例局部、幂等，保留原cache、请求device与输出dtype；CPU FunASR与HF首次lazy import也在同guard内。`test_npu_network_forward_does_not_hold_the_global_loading_guard` 和Qwen对应case证明设备forward不被该全局锁串行化。 | 本行是实际CPU数学/线程与NPU形状假体；没有将它们冒充真实冷A+B。另有新源码cold-i2实机通过；最终默认暖档位另记。 |
| 准入/实例/VAD 队列 | `test_audio_admission.py` 的 FIFO、cancel/grant/timeout race；`test_model_pool.py` FIFO；`test_local_qwen_runtime.py::test_single_instance_serializes_and_cancelled_waiter_does_not_execute`；VAD 排队取消 | 已取消等待者不再执行，不占用新实例；等待计数回零；其他请求继续；获 grant 但调用方尚未取得租约时仍回滚。 | CPU 模拟与真实 CPU spawn；未用共享卡上的破坏性故障测试替代它们。 |
| 设备工作执行 | `test_inference_backends.py::test_repeated_cancel_holds_lease_until_delayed_device_complete`；`test_local_qwen_runtime.py` 的重复取消/超时；音频生命周期的推理取消 | 使用真正阻塞 CPU 线程模拟设备 enqueue 后尚未完成。busy、模型锁和音频租约保持，完成后才归还；timeout 不能把仍执行的工作当作已结束。 | CPU 用例本身不能证明 NPU 完成；本次另有真实 Qwen 取消工装的 start/finish-after-sync 与最终归零记录，见下表；未做破坏性设备故障。 |
| 结果读取/完成交付 | `test_npu_lifecycle_faults.py::test_device_to_cpu_copy_fault_waits_for_copy_and_sync_then_quarantines`；`test_local_qwen_runtime.py::test_cancellation_at_worker_completion_is_not_swallowed`；缺前缀/不支持语种错误 | `.cpu()` 假体阻塞期间租约不释放；ACL 式复制故障后先走同步，再隔离实例，唤醒等待者并归零；future 完成边界取消不能被吞。 | `.cpu()` 故障是受控假体，没有真机注入 memcpy/驱动损坏。 |
| 释放窗口 | `test_npu_lifecycle_faults.py::test_release_condition_wait_repeated_cancel_keeps_lease_until_release_finishes`；准入 repeated cancellation/idempotent release | body 完成后用实际 condition 锁阻塞归还；重复 cancel 时实例仍 busy，close 不能提前前进。放开 condition 后取消传播、close 完成、busy/waiting 归零。 | 精确锁窗口只在 CPU 验证，不冒充硬件竞态覆盖。 |
| 实例故障隔离 | `test_inference_backends.py` 的 no-fallback、坏实例唤醒；`test_model_pool.py` 全失败/部分失败；Qwen 多实例独立；VAD crash/devicefail | 失败实例不能再次租用；其他实例仍服务；无可用实例快速失败。CPU VAD 子进程崩溃是真 `os._exit`，设备损坏是 `InstanceFailure` 替身。 | 未在真实 NPU 上制造 OOM、ACL context 损坏、断卡或杀共享服务。 |
| 关闭/关闭失败 | `test_inference_backends.py::test_close_failure_keeps_ownership_rejects_leases_and_can_retry`、`test_failed_inference_cleanup_is_quarantined_until_close_retry`、`test_repeated_cancel_cannot_interrupt_other_instance_closes`；Qwen close 对应 case；`test_vad_pool_static.py::test_forced_worker_cleanup_is_failure_and_other_workers_still_close` | 成功实例释放；失败实例/设备/锁/失败信息保留，closing 状态拒绝重用；继续关其他实例，聚合失败上抛，允许重试。重复取消不能跳过剩余 close。VAD 超时后强制回收也保留失败；继续清理所有其余 worker，空 stop 重试不得抹掉超时历史。 | close 故障是 CPU 注入；正常实机关闭已另有 short5 证据，不能替代 CPU 的关闭失败注入覆盖。 |
| VAD/部分启动/IPC 关闭 | `test_vad_pool_static.py` 晚到 1 MiB 结果、部分 spawn 失败；`test_validate_npu_service.py` 部分 Popen 失败 | reader 排空到子进程退出，Queue close/join_thread、Process close 与 Event 引用释放；未启动完也回收此前取得的所有权。 | 不能单凭 CPU 无 warning 宣称 NPU TBE/resource_tracker 警告已解决。 |
| 测试批次中止/端口 | `test_validate_npu_service.py` 首次/重复信号、CPU 自有端口与 timeout；候选/取消工具外部 stop case | finally 内信号不得遗弃关闭任务；只操作记录的自有 PID/进程组。超时后 TERM/KILL 即使清理成功也判验收失败，不按端口/名称批量杀进程。 | 实机 short5 已取得真实正常关闭证据，见下表；精确强制/重复信号窗口仍由 CPU 注入覆盖。业务成功不能单独替代关闭检查。 |

## 合成参考与真实音频来源

`test_vad_reference.py` 的三个 fixture 相关 case 使用临时生成的 **4 秒、16 kHz、mono、PCM16、220 Hz 确定性正弦**：四个指定时段中两个低幅度，注释文档含两个独立“细小声音”标记、说明行和缺逗号修复。它验证人工文档解析、scan 计数/结构、percentile 和相邻时段 context。原来的 **658 段/46 个标记**属于本项目不存在的被忽略课堂 fixture，本次没有重建、伪造或宣称重新验收这份真实标注；其余五个算法 case 保留。synthetic 的四段/两个标记不是人工语音 ground truth。

真实课堂音频由 [audio-manifest.json](audio-manifest.json) 记录 `sample_id`、原始绝对路径 `path_hex`、源 SHA256、格式与 header 时长；GBK 文件名不重命名、不覆盖。候选工具另记录 offset、duration、PCM SHA256，默认采用互不重叠片段；源 SHA256 若仅来自 manifest，报告明确区别 declared 与重新 verified。52 WAV/2 MP3 的存在、header 探测、实际片段解码、完整解码和人工标注是不同等级的证据。没有人工标注的真实课堂片段不能用于宣称 WER/CER、情绪准确率或有效对话召回达标。

## 实机执行记录与待完成范围

以下路径均是远程只读原始记录，统一前缀为 `/tmp/asr-apply-20261117-a99trpfv/`。临时日志在形成最终报告时应复制/摘要到可追溯交付目录，不能只保留本表中的临时路径。

| 范围 | 已有 N1 记录/观察 | 当前可采用的结论及限制 | 任务 |
| --- | --- | --- | --- |
| 四模型首次前向 | `model-smoke.log` | 已出现 CAM embedding、emo 七类、native VAD 段和 Qwen raw 前缀及关闭资源记录。是短样本前向，不代表原始精度问题已解决或业务质量达标。 | 0.2；2.1–2.3；3.3/3.5；11.5 |
| CAM 严格 FP32 | `model-validation-final-eager.jsonl`；详见 [local-model-comparison.md](local-model-comparison.md) 的 SHA/物理行号与原命令 | 正式两个来源×1/3/5 秒六片全部满足原预登记 FP32 门槛，最差 maxabs=5.36442e-6，0.3 聚类 partition 与 CPU 同为两簇。BF16 六片失败，FP16 五片通过但 S1-1s 超 .01 门槛，均不进入主链；原 HF32 失败与修正 probe 保留。CPU FP32 参考，无 CUDA golden，无人工说话人标签。 | 3.4；10.7/10.8；11.1/11.3 |
| emo 精确 GELU | `model-validation-final-eager.jsonl`；`emo-layer-probe.jsonl`；[local-model-comparison.md](local-model-comparison.md) | 正式六片 FP32 七概率/labels/Top1 全通过，最差 maxabs=4.90993e-6。FP16 六片通过自身 .01 门槛，但稳态中位20.399ms高于FP32的18.465ms，未观察到收益；BF16 S1-3s 超 .02 且 Top1 改变。未改变容差或原权重/CPU GELU 语义，无人工情绪准确率结论。 | 3.5；10.7；11.1/11.3 |
| VAD 三路同参 | `model-validation-final-eager.jsonl`；[local-model-comparison.md](local-model-comparison.md) | 正式两来源六片 AutoModel CPU、native CPU、native NPU 全部段数/边界一致，误差0。三路每片 elapsed 明细已记录；短片里 NPU 不慢于CPU，仍不足确定完整录音/同卡争用下的最优配比。nested kwargs 保持既有实际语义，不借迁移修阈值/段上限。 | 0.2；10.9；11.1/11.5 |
| Qwen eager / 分桶 / attention / 图候选 | `qwen-eager.json`、`qwen-batch.json`、`qwen-eager-attention.json`、`qwen-torchair-eager-attention.json`；[qwen-backend-selection.md](qwen-backend-selection.md) | 本次已完成对应短片段候选：默认eager/SDPA与batch均有成功原始记录；eager-attention也有输出但部分文本改变、没有人工质量参考，不能仅凭微小吞吐差换主链；TorchAir+eager-attention 42条全部失败（failed，非skip/通过）。正式保持默认SDPA+原生eager+BF16，batch/图未加入生产；后续1/2实例标定是同后端容量比较。 | 2.1–2.5；10.4–10.6；11.3 |
| NPU 安全取消 | `qwen-cancellation.json`；清单 `asr-qwen-cancel-manifest.json`；文件 SHA256 `b9ee8bca2eb0e6ac011e87cfc300f060cc541c741163585b1214bea6f1feeba8` | 真实 NPU 工装结果 passed：`cold_load_cancel_repeated`、`two_requests_one_instance`、`aclose_cancel_repeated_while_active` 三阶段通过。observer actual_started=actual_finished=4、peak_actual_active=1、最终 actual_active=0；client busy=waiting=0、loaded=false、close_finished=true、cleanup=passed。此为单实例真实安全取消/收尾证据，不等于 OOM/ACL 损坏或每个 Python 指令窗口均被注入。 | 2.5/2.6；10.9；11.2/11.6 |
| dtype/lazy import修复后的冷A+B | `e2e-cold-mixed-i2-fixed/summary.json`、`request-cold-A.json`、`request-cold-B.json`、`status-drained.json`、`shutdown-after.json` | 冻结源码后同时发A与全增强B，Qwen实例数临时为2：A=27.563264秒、B=34.476145秒，B的两个中文段有角色/情绪输出；这是冷并发修复验收。最终宿主/容器据完整课堂对照改为单Qwen副本和segment/request 1/1门限，不能将该临时双副本写成默认。原始排空状态中CAM三实例/emo一实例均为FP32且HF32禁用，emo exact GELU=28、FP32 bias factory=1；Qwen两实例707个浮点参数各自均为BF16，execution_failures=0。所有模型池/VAD busy=waiting=0，VAD四worker ready且failed=0。四worker均exit0、forced=false、cleanup_complete=true（CPU1.829/2.043秒，NPU7为9.422/6.853秒），parent exit0、无监听/后代、5秒观察无超时/强制回收/应用或worker关闭失败。 | 2.1/2.2/2.5；3.2/3.4/3.5；5.2；10.9；11.2/11.6 |
| v3 短 HTTP 批次与真实正常关闭 | `e2e-short-5/` 的 32 个 `request-*.json`、`status-initial.json`、`status-drained.json`、`summary.json`、`ownership.json`、`shutdown-before.json`、`shutdown-after.json` | 本次实机执行确认八组合、语种、静音/空/损坏与连续请求、排空及整个批次关闭通过。四个 VAD worker 均 exit0、forced=false、cleanup_complete=true；CPU graceful 约 2.23/1.88 秒，NPU 约 7.12/5.62 秒。父进程 exit0、5 秒观察无孤儿/端口监听，外层 shutdown_timeout=false、application_shutdown_failed=false、worker_shutdown_failed=false。日志无此前 semaphore/TBE 警告；采用正常退出链，未加入私有 `_npu_shutdown`。这是该短批次范围内的通过，不外推长课/八路/破坏性故障。旧 short4 的 INT 无响应记录保留，不改写成通过。 | 0.1；2.4/2.6；5.1–5.3；10.17；11.1/11.4/11.6 |
| 真实超时错误路径及新关闭入口 | 实机记录 `e2e-4012`，最终文件路径由报告固化 | 已实际得到 4012，并等待设备工作完成；新 main TERM 入口在外层观察为 exit0/端口解除/无后代/无外层 timeout，且未再出现此前 30-semaphore/TBE 日志。当时约 30 秒总关闭不能用于推断任何单 worker 超时。随后 short5 的逐 worker `VAD_WORKER_SHUTDOWN` JSON 已确认各自正常退出；强制失败传播与非零退出门槛也已补齐。该错误请求只证明超时映射/原工作收尾，不等于所有失败类型真实注入。 | 2.6；10.9；11.2/11.6 |
| 完整长MP3 | `e2e-long-mp3/summary.json`、`request-full-0.json`、`status.jsonl`、`shutdown-after.json`；[classroom-threshold-review.md](classroom-threshold-review.md) | 完整2422.476秒，请求725.202986秒、RTF0.299364；最终349段/12070字符，日志360完成失败0不可替代最终计数。角色全spk_0、emo以平淡为主，复读告警/超门槛段/越界/重叠均0；104快照及多口径资源原始SHA、四worker正常关闭/父exit0/端口解除均已汇总。单条补充录音不是八份独立验收，也无人工质量结论。 | 0.4；2.2/2.4；5.3；6.5；9.4；10.17；11.1/11.4–11.6 |
| 八份独立完整WAV g1（1/1） | `e2e-full-g1-final/`；[classroom-threshold-review.md](classroom-threshold-review.md) | 八个不同SHA，4B+4A全量8/8；活动峰8/B4，P50=4104.338199秒、P95=4781.662453秒。最终3109段/100071字符，处理3210完成失败0。B簇数2/1/7/9，A的role/emo null为禁用增强的预期。全部最终时间合法/无重叠；一次2秒506→6字符的复读修复、修后超门槛0，五个正常长度循环候选及五对相邻相同文本保留为复听线索。682快照、父exit0/四worker exit0/forced=false、无端口/后代。没有由工程8/8推出人工98%召回。 | 9.1/9.3–9.5；10.17；11.1/11.4–11.6 |
| g1 F3缺口的原CPU参考（C2） | `/tmp/asr-b3-cpu-reference-3j24mrb9/reference.json`、`configuration-provenance.json`；[classroom-threshold-review.md](classroom-threshold-review.md) | 原checkpoint CPU4线程完整WAV：raw196/过滤189/最早1207.77秒与g1计数/最早输出一致，最终166边界全部属于CPU过滤候选。前20分钟所有10ms帧RMS低于−60dBFS，分钟RMS约−94dBFS；支持输入近量化噪声而非NPU新增缺口，能量仍不是人工语音标签。五项nested/YAML/CPU effective/Native构造逐项列明；SNR18及独立segment−40的主链过滤与FSMN内部−100分层。进程已退出/临时副本清理/原资产SHA保持，未NPU或服务。 | 9.4；10.9/10.17；11.1/11.5 |
| 最终保护构建CPU与NPU稳态 | [compiled-final-g1-steady-evidence.json](compiled-final-g1-steady-evidence.json)，SHA256=`82c9ad562bfa99839e4a1552b70b87fb5aa9f1e6ae5b0fa51dd7203bb76fc2aa`；phase `e2e-compiled-final-g1-steady` | 最终stage为`asr-cython-g1-default-20261118-nkmpk_2a`，snapshot=`4050173f89303bc5477078a23dbe8ac3d4be84f95f95dba6eb0ccb75a427a992`。3次全B预热后16次timed实际NPU请求通过，P50/P95=2.966116/3.499545秒；8资源/收尾checks全true，allocated/reserved增长0、RSS+5701632字节、线程/FD增长0，缓存及租约排空。父exit0、四worker exit0/forced=false，5秒无后代/监听，所有shutdown失败标志false。旧3个.so的4项CPU guard边界报告另存`/tmp/asr-compiled-frontguard-cpu-3hr7fbt3.json`，不将它冒充新stage实机；也不与已失败TorchAir候选混淆。 | 10.14/10.15；11.3/11.4/11.6 |
| 正式候选 / 独立来源 / 完整档位 | `model-validation-final-eager.jsonl`、隔离`model-validation-torchair.jsonl`；[local-model-comparison.md](local-model-comparison.md) | eager严格FP32及第二来源通过，低精度按各自gate记录；TorchAir六组合首forward全失败，其后续长度/稳态不写通过。另有完整MP3、同八源g1/g2（1/1与2/2）全量/关闭成功；配对3109段的文本/边界/角色/emo/语种逐值相同，两批每档只跑一次且共享卡，不宣称全局最优或人工准确率。最终选1/1、新CPU449及新保护构建稳态均有独立完成证据。 | 3.4/3.5；9.3–9.5；10.7/10.17；11.3–11.5 |

### 9.4课堂阈值证据的覆盖边界

[classroom-threshold-review.md](classroom-threshold-review.md)已包含真实完整输入、每源SHA/原始响应、0.3下簇分布及碎片/合并风险、40字符/秒长度门槛与真实触发、修后候选与异常计数、CPU参考及两个VAD参数层次。它足以形成“保留现阈值、明确适用范围”的实测评审；不提供人工说话人/情绪标签，也不以一致性宣称WER/CER、BCubed F1或有效对话召回达标。g2配对已检查相同八源，3109个业务分段及顶层parsed文本/语种逐值相同，差异0；[full-pair-output-comparison.json](full-pair-output-comparison.json)只记录计数/哈希，不复制完整课堂文本。一致性不代表准确率。tasks勾选与最终验收由实际完成证据决定，本矩阵不改任务状态。

## 冻结源码的原生/VAD只读复核

本次复核未运行NPU或启动服务。远程hash-only与已审阅的文件逐一一致：

| 文件 | SHA256 |
| --- | --- |
| `core/local_inference.py` | `c676b25b4abb86d9e4a589c7da88acd28430ede98925541fa07689c317d4bc01` |
| `core/models.py` | `19edd7f9c567f9c34b953607ef667f9a50515f14b192e45e83a64ede9ce9541d` |
| `core/vad_pool.py` | `3f7c52d272690ddbce226823843c7d73b3d04fd7f90d47f3cac3cace7f1053ed` |
| `core/model_loading.py` | `eaabd44ad9234cd2e335a2309cca07f22f1fc91f4e6d5417bfdbcfc047995d9c` |

复核结论：未发现新的阻断问题。`configure_npu_precision`先set_device再访问配置/context，主机证据中实际设备均为`npu:7`；容器logical0需另按物理7绑定核对。CAM/emo加载在同一guard内，FP32存储在statecopy前确定并校验实际参数及浮点缓冲；CPU fbank/ALiBi factory使用明确边界，设备forward不持全局guard。ModelPool关闭等待租约结束，继续尝试其他实例后聚合失败；失败runner/设备/锁与closing状态保留，坏实例quarantine不再出租。保留runner不等于保留其全部内部张量：原生runner的close在同步失败时也会finally清内部网络引用，pool仍保留失败、禁止复用且关闭不能报成功。VAD关闭继续排空IPC、显式释放Queue/Process/Event，强制或非零退出均上抛；不引入私有NPU shutdown。

VAD保留既有有效参数：权重配置中的`model_conf.max_single_segment_time=8000`仍决定8秒段上限；服务声明的12000以原来的nested `vad_kwargs`传递，没有在迁移中提升为顶层覆盖项。`NativeVadRunner`按原AutoModel方式合并model_conf与顶层配置，frontend与cache/reset行为也沿用原实现。将它改成真正生效的12秒会改变分段行为，需单独评审，不能当设备适配顺手修正。正式1/3/5秒三路对照在段数/边界上全相同，但这些短片不能独立证明8秒截断边界的长音频行为。

冷并发原始文件的SHA256为：summary `96f69ebd0103a3c2a7bdd35c34ee87a6c141fa52ac57f11ed2165938b86a8a59`、status-drained `250eb6a88651a90fc3315898367c87b4f97db5f5c3ecd0d03058e3ad0fe35fb9`、shutdown-after `934efb15cc733b9579799988435b2b7afed0c06b444f17f4e10bf6d3dc3d365c`、ownership `c21733c8b9022b31fdb2780a907f32b678c0ebd66e261fd3d7257d72d8ea7d7a`。ownership中的`source_snapshot_sha256`原值为null；上述源码指纹是复核时实读结果，不伪装成工装当时已经记录的快照字段。

一个非阻断的资源统计限制必须保留：cold-i2的CAM load allocator delta为7860.67MB、emo为646.06MB，Qwen两个实例分别4446.86/7805.32MB。并发加载时，同一进程其他模型的分配会进入before/after差值。这些原始数值是加载窗口估计，不能相加归因或当作各模型独占权重/峰值体积；总体/进程HBM及后续稳态采样另看资源报告。

## 尚不能据此宣布完成的事项

- 所有破坏性 NPU 故障（OOM、ACL context/stream 损坏、驱动/卡故障）均未真机注入；仅有受控 CPU/错误替身验证。禁止为补证据耗尽共享卡或终止既有服务。
- 精确加载锁等待、结果复制故障、释放 condition 竞争已补 CPU 回归，仍不能冒充同窗口的真实 NPU 故障覆盖。真实 Qwen 安全取消已有独立 observer/sync 证据；Qwen 内部加载锁直接争用、batch_decode 执行中精确取消以及全局 `_client_instance` 的关闭失败引用断言未分别注入。已有完成回调取消、每 runtime close 失败测试与“仅 close_task 成功才清全局引用”的源码核查不能写成这几个额外 case 已通过。
- 严格 FP32 的 HF32 与 GELU 问题已保存失败、定位和修正 probe；正式源码两个来源六片现已通过。CPU 参考不等于 CUDA golden，六片也不能提升为整网全场景或人工质量验证；完整录音与同卡并发另行记录。
- 未采用的生产微批/图编译路径应标不适用或候选待决，不填写成“生产已通过”。候选的混合长度/mask CPU 测试不替代真实 kernel 与长音频取消验证。
- 稳态记录应区分 allocated/reserved、进程 HBM、设备聚合 HBM、RSS/线程/FD、缓存文件和 worker/租约。现有工装保存相关原始采样；共享卡既有 PID 不归因给本项目；未做完整长时测试前不能宣称没有持续增长。
- 服务关闭是独立门槛：部分启动、信号、超时 CPU 测试不能替代本次 NPU 服务的退出记录。必须同时检查外层 harness 与内部逐 worker 的 `forced`、`cleanup_complete`、graceful 耗时/exitcode；外层 exit0/端口释放不证明内部没有超时。超时即验收失败，即使之后已按归属清理；TBE/resource_tracker 噪声也须与实际孤儿进程/端口分开分析，不能通过隐藏 stderr 或忽略失败获得通过。
- 新发现并复现的HF全局default dtype竞态已用共享guard、native实际FP32存储校验及明确的CPU前端/实例bias factory边界修复；设备forward仍在全局guard外。最新相关CPU子集60 passed/30subtests与其他子集重叠；新源码冷A+B已另由`e2e-cold-mixed-i2-fixed`实际通过，暖短测2/2是历史记录，最终完整课堂选择单副本1/1。串行候选的旧数值记录保留，不作为这轮并发通过的替代证据。
- 最终1/1源代码449/146 CPU回归与最终保护构建NPU稳态均已完成，分别关联新日志/JSON、stage/snapshot及正常关闭记录。上述通过范围不扩张为未真机注入的破坏性故障或缺失的人工质量指标；实际任务状态及归档以独立规格/实现审核为准。
