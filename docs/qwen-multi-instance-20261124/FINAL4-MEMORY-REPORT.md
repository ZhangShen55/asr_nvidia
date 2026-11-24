# 最终Qwen4/CAM++4/emo1独立内存与池队列报告（2026-11-24）

原始目录：`/home/zhangs/asr_nvidia/logs/qwen-multi-instance-20261124/final-m4-cam4-emo1-v1`。仅在`run.json`确认`phases.shutdown.passed=true`且`remaining_pids=[]`后流式读取大JSONL；只用标准库读取文件，未运行测试、设备命令、HTTP、模型或服务。详细来源SHA与原值见同名JSON。

用户已采用最终Qwen4/CAM++4/emo1，S16/global64，HTTP1；本报告是独立证据核对，是否完成验收由root结合完整源码/保护/真实请求/关闭结果确定，不代表长期稳定、准确率认证或此前N6验收通过。

## 请求结果

| 阶段 | 完成 | P50/P95秒 | 墙钟秒 | 警告/丢弃 | 200秒目标 |
|---|---:|---:|---:|---:|---|
| A4 | 4/4 | 164.846/184.524 | 186.443 | 13/12 | True |
| B4 | 4/4 | 183.260/199.447 | 200.100 | 12/11 | None |
| A4-resident | 4/4 | 172.538/182.313 | 183.469 | 12/11 | True |

本轮`requested_phases_complete=true`、`final_n4_functional_matrix_complete=true`、integrity通过；保留原`formal_matrix_complete=false`，不将用户最终N4功能矩阵混同此前N6矩阵。A4及增强驻留A4均达到各自200秒目标，B4未配置200秒目标。真实准入probe通过（不纳入上述性能窗口）：N4总容量已满时第5条B/额外A均4013，B-only 4017在总容量与B容量同为4时不可达。任务是否勾选仍由root完整验收。

## HBM与采样缺口

NPU7单位保持原始字段MB。重复TTL快照按resourceDiagnostics.timestamp去重；峰值为观测峰，缺口不插值、不承诺捕捉瞬时峰。

| 阶段 | GET/独立资源样本 | 缺HBM轮询 | minFree/maxUsed MB | 资源最大间隔秒 | GET最大间隔秒 |
|---|---:|---:|---:|---:|---:|
| A4 | 232/39 | 0 | 21997/43539 | 8.370 | 5.193 |
| B4 | 222/36 | 0 | 18996/46540 | 7.101 | 4.981 |
| A4-resident | 205/36 | 0 | 15406/50130 | 8.075 | 5.000 |

## CAM真实身份、精度与分配

| CAM index | PID/create_time | READY精度/math | B4调用/成功 | B4累计推理包络秒 |
|---|---|---|---:|---:|
| 0 | 489009/1795487913.44 | fp32/fp32_ops_hf32_disabled | 700/700 | 58.739 |
| 1 | 557990/1795487934.6 | fp32/fp32_ops_hf32_disabled | 689/689 | 58.001 |
| 2 | 684770/1795487955.86 | fp32/fp32_ops_hf32_disabled | 699/699 | 58.268 |
| 3 | 820738/1795487977.15 | fp32/fp32_ops_hf32_disabled | 687/687 | 60.151 |

CAM身份取`model_workers[].worker_identity.pid/create_time`；READY同时记录原始FP32、HF32 disabled、浮点存储均FP32及test_double=false。每worker的当前allocator与READY/最后RPC完整快照在JSON中。上表为真实NN调用分配增量，先扣B-warm；累计包络包含前端、派发/IPC、设备等待和同步，不能直接称为核时，也不能跨段相加解释墙钟。

## B4等待、NN/skip覆盖与N5比较

| 池 | 调用/成功/skip | 累计等待秒 | 平均等待ms | 累计推理包络秒 | 平均推理包络ms |
|---|---:|---:|---:|---:|---:|
| speech_campplus | 2775/2775/0 | 67.526 | 24.334 | 235.158 | 84.742 |
| emotion2vec | 2226/2226/549 | 318.610 | 143.131 | 94.676 | 42.532 |

| B4指标 | 本轮Qwen4/CAM4/emo1 | 旧N5/CAM1/emo1 |
|---|---:|---:|
| CAM_waiting P50/P95/max | 0.0/1.0/25 | 254.0/255.0/255 |
| EMO_waiting P50/P95/max | 0.0/8.0/29 | 0.0/1.0/34 |
| PCM_active P50/P95/max | 64.0/64.0/65 | 256.0/256.0/256 |
| PCM_waiting P50/P95/max | 0.0/0.0/0 | 26.0/37.0/41 |
| PCM_bytes P50/P95/max | 27758592.0/33486352.0/36480064 | 82469056.0/101002144.0/104940736 |
| PCM满256的轮询点 | 0/222 | 240/348 |
| 墙钟秒 | 200.100 | 296.961 |

这里只复用既有`N5-MEMORY-REPORT.json`，未重读旧大日志。两配置同时改变Qwen份数和CAM份数，且各只有一轮，不能把差值全部归因CAM；等待分位数与PCM满点占比是采样点统计，不是时间加权占用率。emo仅执行原时长范围内段，范围外skip单列，不以skip冒充NN完成。

## 每owner当前allocator与增长

每格allocated/reserved MB，值取worker RPC/状态缓存，非同步全局原子快照。各owner身份与暖身一致。

| owner/PID | 暖身后 | A4后 | B4后 | 驻留前 | 驻留后 | reserved较暖身增长MB |
|---|---:|---:|---:|---:|---:|---:|
| 0/2307689 | 3906.119/4390 | 3897.368/5584 | 4001.168/5642 | 3906.447/5642 | 3927.011/6754 | 2364 |
| 1/2554711 | 3906.447/4390 | 3915.197/6632 | 3914.760/6688 | 3906.447/6688 | 3905.681/7376 | 2986 |
| 2/2663705 | 3906.119/4390 | 3913.229/7062 | 3889.246/7062 | 3906.447/7062 | 3912.354/7062 | 2672 |
| 3/2759994 | 3906.228/4390 | 3921.432/6028 | 3920.120/6936 | 3906.447/6936 | 3915.088/7520 | 3130 |

最终active_sequences、kv_bytes、kv_storage_bytes原值在JSON。当前reserved增长仍是缓存生命周期线索，不能据此称为泄漏、全部可安全释放或已彻底修复。不同owner的独立max峰不相加；frame/token预算与PCM上限不保证总HBM容量。

## Host/swap及关闭恢复

| 阶段 | host available最小GiB | swap used最大MiB | swap-in/out增量bytes |
|---|---:|---:|---:|
| A4 | 1750.677 | 0.000 | 0/0 |
| B4 | 1721.165 | 0.000 | 0/0 |
| A4-resident | 1721.529 | 0.000 | 0/0 |

host/swap是系统级共享负载观测，不能单独归因本服务。

shutdown passed=True，exit=0，forced=False，remaining_pids=[]，remaining_cache_files=[]；端口观察5秒。

本轮背景恢复原字段：`{"passed": null, "observation_complete": true, "hbm_observed": true, "status": "unknown", "before": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15908, "hbm_total_mb": 65536, "hbm_free_mb": 49628, "chip": 0, "bus_id": "0000:42:00.0"}, "after": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15910, "hbm_total_mb": 65536, "hbm_free_mb": 49626, "chip": 0, "bus_id": "0000:42:00.0"}, "used_delta_mb": 2, "free_delta_mb": -2, "external_process_changes": [{"pid": 1601397, "chip": 0, "before_mb": 12417, "after_mb": 12417, "delta_mb": 0}], "owned_normal_exit": true, "near_background": null, "scope": "physical NPU7 actual MB evidence plus owned exit; nonzero aggregate delta requires shared-load review, no replacement free threshold", "reason": "recovery not verified: inspect NPU7 numeric evidence, external changes and owned cleanup; unknown is not zero"}`。保留unknown原判定；+2MB等小差值也不擅自改为恢复通过或零。

既有N5背景恢复原字段：`{"passed": null, "observation_complete": true, "hbm_observed": true, "status": "unknown", "before": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15908, "hbm_total_mb": 65536, "hbm_free_mb": 49628, "chip": 0, "bus_id": "0000:42:00.0"}, "after": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15910, "hbm_total_mb": 65536, "hbm_free_mb": 49626, "chip": 0, "bus_id": "0000:42:00.0"}, "used_delta_mb": 2, "free_delta_mb": -2, "external_process_changes": [{"pid": 1601397, "chip": 0, "before_mb": 12417, "after_mb": 12417, "delta_mb": 0}], "owned_normal_exit": true, "near_background": null, "scope": "physical NPU7 actual MB evidence plus owned exit; nonzero aggregate delta requires shared-load review, no replacement free threshold", "reason": "recovery not verified: inspect NPU7 numeric evidence, external changes and owned cleanup; unknown is not zero"}`。它对应旧轮次，不混作本轮观测。

后续若继续优化，应分别验证Qwen owner缓存生命周期与CAM前端/IPC/NN包络；本报告不授权额外负载，也不证明长期稳定或缓存增长根因已消除。
