# 增强模型启动预加载验证（2026-11-24）

## 结果

已开启且设备数组非空的CAM++、emotion2vec现在于HTTP就绪前完成所有配置实例加载尝试，成功实例常驻。关闭/空数组跳过，部分或全部加载失败仍沿原增强池降级契约；请求复用模型池，仅按参数执行对应增强分支。此次只修改启动加载时机，不执行启动语音forward暖身，不改变设备、实例数、精度或推理策略。

## 验证

- CPU隔离回归：= 1301 passed, 2 skipped, 1 deselected, 2 warnings, 379 subtests passed in 258.64s (0:04:18) =。含新增9项启动预加载测试，以及相关生命周期、模型池、能力与状态回归。硬件模型探针文件和项目既有单个直接npu-smi探针不列入CPU范围；2个跳过为保护构建专属测试。无源文件漂移、无残留owned进程、无共享cache写删。设备/权重执行由guard阻止。
- 真机使用宿主物理NPU7，当前Qwen4/CAM4/emo1及原配置；先读取GET，再发送任何音频。请求前CAM健康4、emo健康1，实际推理调用均0，说明加载已在启动阶段完成。
- 连续两次真实v3双增强短音频请求均成功，两增强分支真实成功调用数增加；各实例load_attempts始终1，CAM worker出生身份不变。短片用于功能冒烟，不要求逐个CAM槽执行推理，不做吞吐或完整课程质量验收。

| 模型 | 请求前已加载实例 | 请求前调用 | 第一次请求后调用 | 第二次请求后调用 |
|---|---:|---:|---:|---:|
| speech_campplus | 4 | 0 | 1 | 2 |
| emotion2vec | 1 | 0 | 1 | 2 |

- 正常TERM后服务exit0、无强制收尾、owned身份退出，端口持续释放至少5秒，服务PCM缓存为空；既有PID1601397及8007服务保留，根配置未改变。
- 原始证据：`/home/zhangs/asr_nvidia/logs/enhancement-preload-20261124-v2`（before-any-audio.json、after-1/2.json、response-1/2.json、service.log、summary.json）；CPU：`/tmp/enhancement-preload-cpu-final-20261124`。精简证据及源/config/输入SHA见verification.json，原始响应不提交。

## 验证过程中的非通过轮次

首个相关子集4项旧状态夹具未隔离新启动入口，被CPU guard拦截，已补齐替身。第一次全量在旧准入mock测试停留，主动中断、owned归零；其独立9项复验通过。第二次全量1301通过/2跳过，唯一失败为既有直接npu-smi硬件探针被guard拦截；最终使用项目既有CPU排除范围重新完整通过。原日志保留在/tmp/enhancement-preload-cpu-target-20261124、cpu-full-20261124、cpu-admission-20261124、cpu-full-v2-20261124（后三个缩写目录均位于/tmp且以enhancement-preload-为前缀），不将中断/失败轮计为通过。

第一次真机请求与复用检查通过，但冒烟脚本误用close参数，未完成原父进程退出码采集；按ownership.json核对本次PID/出生身份后TERM回收，无强制、无残留及监听。该轮仅算功能证据，不算完整收尾通过；修复脚本后第二轮重新完成全部检查。首轮日志位于logs/enhancement-preload-20261124，recovery-shutdown.json保留退出码unknown，不追造为0。

## 使用与限制

重启服务后生效，无新增配置。启动会提前承担加载时间和权重内存；首次语音推理仍可能有编译开销。本验证不代表Docker镜像已构建/上线、全shape暖身、长时稳定性或文本质量认证。没有重跑旧压力矩阵。
