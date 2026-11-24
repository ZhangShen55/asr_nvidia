# N5独立内存与池队列分析（2026-11-24）

原始目录：`/home/zhangs/asr_nvidia/logs/qwen-multi-instance-20261124/followup-m5-s16-v1`。仅在run.json存在且shutdown.passed=true、remaining_pids为空后读取大JSONL。仅标准库文件读取与报告生成，未运行模型、设备命令、测试或服务，未改业务源码、根默认N6、配置或tasks。

本报告只代表这一次N5/S16复测，不替代此前N6共驻OOM验收，也不证明长期稳定；详细路径、SHA、字段与所有计数见同名JSON。

## 阶段结果

| 阶段 | 完成 | P50/P95秒 | 墙钟秒 | 警告/丢弃 |
|---|---:|---:|---:|---:|
| A5 | 5/5 | 173.702/183.176 | 183.829 | 16/14 |
| A4 | 4/4 | 181.309/194.165 | 195.440 | 14/10 |
| B4 | 4/4 | 284.912/296.236 | 296.961 | 13/11 |
| A5-resident | 5/5 | 173.323/203.229 | 206.119 | 14/11 |

四阶段请求均完成且阶段字段校验通过；A5-resident墙钟206.119秒超过既定200秒目标，故run.formal_matrix_complete=false，不能把本轮整体记为目标验收通过。

## NPU7与采样

单位保持原npu-smi字段MB；GET重复TTL快照已按resource timestamp去重。采样点比例不是时间占用比例。

| 阶段 | GET/独立资源样本 | min free / max used MB | 最大资源采样间隔秒 | 最大GET间隔秒 |
|---|---:|---:|---:|---:|
| A5 | 216/37 | 13398 / 52138 | 9.261 | 5.382 |
| A4 | 238/40 | 11426 / 54110 | 9.412 | 4.624 |
| B4 | 348/59 | 9191 / 56345 | 6.731 | 5.581 |
| A5-resident | 231/41 | 5131 / 60405 | 9.647 | 5.636 |

## B4真实池覆盖与PCM反压

- speech_campplus：真实调用2775、时长skip0、outcome增量{'success': 2775, 'empty': 0, 'failed': 0, 'cancelled': 0}；累积等待59438.663秒（均21419.338ms/调用），推理调用包络259.584秒（均93.544ms/调用）。
- emotion2vec：真实调用2226、时长skip549、outcome增量{'success': 2226, 'empty': 0, 'failed': 0, 'cancelled': 0}；累积等待95.330秒（均42.826ms/调用），推理调用包络82.912秒（均37.247ms/调用）。

B4 CAM waiting P50/P95/max=254.0/255.0/255，EMO=0.0/1.0/34。PCM active峰256段，240/348个轮询点满256段；PCM预算等待峰41，观测字节峰104940736 bytes。累积等待不可相加解释墙钟。CAM前端/IPC/传输/NN/同步未分拆，因此不能将其整段调用包络直接称为核时。

## 每owner当前allocator快照

每格allocated/reserved MB。owner身份与本轮暖身一致；这些是worker状态/RPC缓存快照，未把不同worker的max峰相加，也不将reserved-unused全判可释放。

| owner/PID | 暖身后 | A5后 | A4后 | B4后 | resident前 | resident后 |
|---|---:|---:|---:|---:|---:|---:|
| 0/4034033 | 3906.119/4390 | 3898.571/5672 | 3889.246/5690 | 3897.259/5790 | 3906.447/5790 | 3898.571/5790 |
| 1/101806 | 3905.900/4390 | 3967.224/5558 | 3929.198/6284 | 3928.979/6478 | 3889.246/6478 | 3942.105/6478 |
| 2/249830 | 3906.119/4390 | 3930.183/7186 | 3930.073/7770 | 3905.353/7886 | 3906.228/7886 | 3916.182/8574 |
| 3/379863 | 3905.900/4390 | 3920.995/6418 | 3914.760/6418 | 3913.557/6758 | 3906.119/6758 | 3923.620/7982 |
| 4/478865 | 3906.009/4390 | 3913.557/7580 | 3913.557/7580 | 3913.557/7580 | 3906.228/7580 | 3921.213/8372 |

保留缓存和阶段间当前reserved增长仅为生命周期线索，不能单独认定泄漏或保证安全清理。frame/token预算是接纳估计，不是总HBM容量保证。

## Host/swap与关闭恢复

| 阶段 | host available最小GiB | host swap used最大MiB | swap-in增量bytes | swap-out增量bytes |
|---|---:|---:|---:|---:|
| A5 | 1743.965 | 0.000 | 0 | 0 |
| A4 | 1744.900 | 0.000 | 0 | 0 |
| B4 | 1732.416 | 0.000 | 0 | 0 |
| A5-resident | 1731.013 | 0.000 | 0 | 0 |

上述host/swap为系统级观测，包含其它共享负载，不能据此单独归因本服务。

shutdown passed=True，exit=0，forced=False，remaining_pids=[]，remaining_cache_files=[]，端口观察5秒。

背景恢复原字段：`{"passed": null, "observation_complete": true, "hbm_observed": true, "status": "unknown", "before": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15908, "hbm_total_mb": 65536, "hbm_free_mb": 49628, "chip": 0, "bus_id": "0000:42:00.0"}, "after": {"index": 7, "available": true, "status": "observed", "hbm_used_mb": 15910, "hbm_total_mb": 65536, "hbm_free_mb": 49626, "chip": 0, "bus_id": "0000:42:00.0"}, "used_delta_mb": 2, "free_delta_mb": -2, "external_process_changes": [{"pid": 1601397, "chip": 0, "before_mb": 12417, "after_mb": 12417, "delta_mb": 0}], "owned_normal_exit": true, "near_background": null, "scope": "physical NPU7 actual MB evidence plus owned exit; nonzero aggregate delta requires shared-load review, no replacement free threshold", "reason": "recovery not verified: inspect NPU7 numeric evidence, external changes and owned cleanup; unknown is not zero"}`。数值未测或恢复未知必须保留unknown，不以新free阈值替代证据。

后续如需优化，应先将owner缓存生命周期与CAM前端/NN包络分别验证；本次报告不授权额外负载，不把单次N5成功外推N6、其它档位或长期运行。
