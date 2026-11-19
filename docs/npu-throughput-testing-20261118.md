# 完整音频吞吐测试工具

`scripts/benchmark_asr_throughput.py`只测现行v3。每次运行只负责一个并发档位、一个能力组合和一轮八文件；重复轮次使用不同输出目录，冷/暖态不得合并。NPU模型试验由负责人串行执行；CPU逻辑测试不会启动模型。

清单为`{"samples": [...]}`或列表，必须恰好八条完整文件。每项含`sample_id`（兼容`id`）、`path`/`path_hex`/`path_bytes_b64`、完整源`sha256`、`audio_seconds`（兼容`duration_seconds`或资产审计的`media.duration_seconds`）。历史GBK文件名按原字节读取，JSON转义保留surrogate往返，HTTP上传使用SHA派生的ASCII名称；不重命名源文件。工具逐文件校验SHA、拒绝重复；不自动选最短八条，不切片、不复制源文件。选样与VAD密度/语种依据另存测试报告。

```bash
python scripts/benchmark_asr_throughput.py --manifest /tmp/selected-eight.json \
  --concurrency 4 --mode B --config /tmp/throughput-b4.toml \
  --launch-service --url http://127.0.0.1:18083 --output /tmp/asr-b4-round1
```

暖态测试须显式添加`--warmup-audio /tmp/真实短片.wav`，由调用者另提供4～12秒真实片段；工具不切改原资产。预热在正式八文件batch_start之前单独请求，`warmup.json`保存模式、实际开关、原始响应、错误码、单调/UTC起止，不能把短片算第九个完整文件或混入N=8分位数。A只热Qwen，B/mixed/mixed-swap均以双增强true、Rolefalse触发所需增强池懒加载。预热同样受420秒/无进展诊断与有界取证约束，失败、取消或超线不进入正式轮次，owned资源照常finally关闭。不指定此参数时温态标记`unknown/cold_possible`，不能把首次懒加载轮称暖态。实际八模型健康容量仍须结合状态证据核实。

可额外指定`--expected-model-instances 8`校验预热后的新鲜状态快照：VAD须八健康，B/mixed两增强池均须配置八且实际成功八、失败零；A只要求VAD八，不因校验而加载增强。未指定时工具保持通用，仍保存`final_status`供报告核查。

在已source CANN的conda asr执行；保留原PYTHONPATH。`--launch-service`使用当前解释器与指定配置执行正式入口`python main.py --workers 1 --timeout-graceful-shutdown 180`，保留入口负责的TERM重放及设备收尾，禁止旁路为`python -m uvicorn`。登记父/后代PID及创建时间，所有退出统一finally关闭owned资源，关闭期限至少210秒。正式入口正常TERM应exit0，exit1或其它非零退出即使无剩余进程和监听也不能通过；纯CPU裸Python子进程测试可显式接受SIGTERM退出，但该策略不用于正式服务。不给`--launch-service`只访问外部已有URL，记录`external_nonowned`并保留外部服务；这种模式不能单独作为“本轮测试服务已关闭”的验收证据。运行时勿向输出目录写别的资源。

并发档位为1、2、4、8；每档仍完成同八文件、完成一条立即补下一条。`A`关闭增强，`B`同时showSpk/showEmotion，`mixed`前四B后四A，`mixed-swap`交换；showRoleIdentify固定false。1/2/4默认总8/B4，8纯B需要独立B8配置；工具不擅自改配置或绕过准入，预期拒绝另做功能测试。CPU测试可注入模拟请求和状态。

默认诊断门槛420秒、采样1秒，无进展120秒提前诊断；`--no-progress-s`调整有限无进展窗口。阶段完成计数参与进度指纹，不用增长的时长或HBM伪装进展。P50/P95或任一在途请求严格超过420秒即停止追加，保存`diagnostic-trigger.json`、有界`diagnostic-status.json`，取消未完成HTTP等待并安全关闭owned服务。慢状态接口不阻塞watchdog。诊断窗口默认5秒，服务关闭默认210秒；可通过同名秒参数调整，但不能放宽420秒或缩短5秒监听观察。接口超时、设备仍执行及服务关闭不是同一事件，强制回收明确失败。

`summary.json`保留八条全部状态，包括not_submitted、cancelled、failed和completed。`success_subset_latency`采用线性插值；N<8、诊断中止或关闭失败不可作为整轮验收通过。`elapsed_s`从该请求开始HTTP调用至完整响应，包含上传/服务排队，不含客户端尚未提交等待；整批时间包含后续排队。所有单调时钟均由同一客户端产生，UTC仅关联日志。吞吐仅完整成功轮计算，未完成不填造时延。`response-*.json`独立留原始响应，通用监控不保存音频。

每轮还保存`status.jsonl`及配置快照，原服务状态提供各阶段队列、批统计、健康/忙闲及资源证据。诊断后先定位卡点并小规模复验，再重跑原档位；禁止原样无限重跑。`shutdown.json`记录owned身份、强制回收、剩余进程和连续端口观察，强制或超时即使端口释放仍不通过。保留共享既有服务，不以设备HBM归零验收。最终完整矩阵、真实128容量、同源质量和保护构建实机仍需独立报告，工具存在不代表已经执行。
