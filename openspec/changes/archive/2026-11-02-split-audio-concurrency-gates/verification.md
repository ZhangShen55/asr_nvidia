# 实施验证

日期：2026-11-02。运行环境：`/root/anaconda3/envs/asr/bin/python`，Python 3.11。

## 结果

- 定向回归 **142 个测试用例通过**，耗时 18.241 秒。其中一个诊断封装用例执行三个既有音频对齐、质量扫描及日志解析函数。
- 混合路由集成：`3A+4B` 拒绝新 B（4017），新 A 获准形成 `4A+4B`，随后新 A/B 均为 4013；结束后活动及等待数全部归零。
- OpenSpec 严格校验通过：`openspec validate 2026-11-02-split-audio-concurrency-gates --strict`。
- `git diff --check` 通过。

## 覆盖对应

| 规格场景 | 验证位置 |
|---|---|
| 新配置缺省 4、严格正整数、B 上限可大于总上限、就绪前拒绝非法配置 | `test_config_contract.py`、`test_audio_admission_startup.py` |
| 三个参数默认 false、全部分类组合、角色识别归 B | `test_request_defaults.py`、`test_v2_audio_admission_route.py` |
| 功能开关矩阵、4015/4016/4018、能力拒绝先于队列、已有模型也不能绕过关闭 | `test_v2_capabilities.py`、`test_v2_audio_admission_route.py` |
| 8/4 六种混合容量场景、总满优先、原子授予与幂等回收 | `test_qwen_v2.py`、`test_v2_audio_admission_route.py` |
| A 绕过受阻 B、最早可处理优先、同类 FIFO、共享等待容量及零容量 | `test_audio_admission.py` |
| 等待超时、取消、授予竞争、截止竞争、重复取消回收 | `test_audio_admission.py`、`test_v2_audio_admission_route.py` |
| 准备至清理持有名额、异常与清理失败、取消后等待线程和 VAD 收尾 | `test_audio_admission_lifecycle.py`、`test_v2_audio_admission_route.py` |
| 首次状态零值、混合活动/等待快照、关闭能力无计数变化、查询不加载模型 | `test_audio_admission_status.py`、`test_local_model_runtime.py` |
| 有限拒绝原因、分类统计、音频等待耗时、事件淘汰、近期明细上限 | `test_audio_admission_telemetry.py`、`test_resource_diagnostics.py` |
| 连续请求阶段隔离、取消状态、已准入失败不计为拒绝、新旧日志兼容 | `test_v2_audio_admission_route.py`、`test_vad_quality_validation.py` |
| 成功响应、启用模型段级失败回退、独立模型池、Qwen 信号量、缓存与 VAD 回归 | `test_qwen_v2.py`、模型池/缓存/片段/VAD 相关测试 |

## 复现

定向 unittest 模块如下；运行时将 `utils.asr_stats.STATS_FILE` 指向临时文件，避免写入工作区业务统计。

```bash
/root/anaconda3/envs/asr/bin/python -m unittest \
  tests.test_config_contract tests.test_request_defaults tests.test_audio_admission_startup \
  tests.test_v2_capabilities tests.test_qwen_v2 tests.test_audio_admission \
  tests.test_audio_admission_lifecycle tests.test_audio_admission_status \
  tests.test_audio_admission_telemetry tests.test_v2_audio_admission_route \
  tests.test_model_pool tests.test_model_pool_route tests.test_local_model_runtime \
  tests.test_cache_lifecycle tests.test_memory_clip_pipeline tests.test_segment_clip_lifecycle \
  tests.test_resource_diagnostics tests.test_model_memory_pressure_diagnostics \
  tests.test_progress_logging tests.test_vad_pool tests.test_vad_pool_static
```

额外诊断封装执行 `tests/test_vad_quality_validation.py` 的三个 `test_*` 函数，
需要 `tmp_path` 的两个函数传入临时目录的 `Path`。本次完整运行脚本及输出分别位于
`/tmp/asr-admission-regressions.py`、`/tmp/asr-admission-regressions.log`。

默认配置测试使用临时空 TOML，增强功能测试显式设置所需功能开关；
保留工作区已有 VAD 设备、功能开关等部署定制，未用修改部署参数的方式满足测试。
预期的模型加载失败、推理失败、清理失败及假 VAD worker 崩溃均为故障注入验证。

实现验收阶段未进行真实 GPU 模型负载或 Qwen 服务压测，未部署或重启业务服务。
配置修改在服务重启后生效，两类限制按进程独立执行。

## 归档验证

2026-11-02 已将增量规格同步到三份主规格，并完成归档；原有标题、Purpose 和未涉及条款保留。
`openspec validate --specs --strict` 验证 6 份主规格全部通过。
提交前将暂存内容导出到独立目录复测，142 个测试用例全部通过，耗时 18.432 秒；
本次提交仅包含音频并发实现及归档，工作区的其他开发改动保留。
