# CAM++/emotion2vec 4+4 实例完整音频并发测试报告

## 1. 测试结论

本次在 `cuda:0` 上分别加载 4 个 CAM++ 实例和 4 个 emotion2vec 实例，使用完整音频、`showSpk=true`、`showEmotion=true` 完成 2 / 4 / 8 / 16 / 32 路并发测试。

- 2、4、8、16 路全部成功，模型池正常并行工作。
- 8 路吞吐最高：`1.811 音频/分钟`。
- 16 路仍保持全部成功，吞吐为 `1.808 音频/分钟`，已接近本配置的稳定吞吐平台。
- 32 路仅 `15/32` 成功，`17` 个请求因 VAD 切分超时或排队超时失败，P95 尾延迟升至 `756.934 秒`。
- 当前推荐并发上限设为 16；32 路需要先调整 VAD 排队/超时和整体资源策略，不能直接作为稳定生产并发。

## 2. 测试条件

| 项目 | 配置 |
|---|---|
| Conda 环境 | `asr` |
| 本地服务 | `127.0.0.1:8084`，`instance_count=1` |
| Qwen-asr | `127.0.0.1:8806`，测试期间未停止、未重启 |
| 音频 | `test/audio_file/教师1.aac`，完整音频，时长约 `3385.722 秒`（56.43 分钟） |
| 请求开关 | `showSpk=true`、`showEmotion=true` |
| CAM++ | 4 实例，全部 `cuda:0` |
| emotion2vec | 4 实例，全部 `cuda:0`，fp16 |
| VAD | CPU，常驻 8 worker，上限 16 |
| 测试顺序 | 2、4、8、16、32 路，每档并发提交完整音频副本 |

本次实际使用的模型池配置为：

```toml
[model_device.local_model_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0", "cuda:0", "cuda:0", "cuda:0"]
```

## 3. 并发结果

吞吐按该档成功请求数除以该档 wall time 计算；失败请求不计入音频吞吐。

| 并发 | 成功/失败 | wall(s) | P50(s) | P95(s) | max(s) | 音频/分钟 | 段/秒 | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 2 | 2/0 | 89.139 | 87.688 | 89.138 | 89.138 | 1.346 | 11.622 | 全部成功 |
| 4 | 4/0 | 139.539 | 133.628 | 139.536 | 139.536 | 1.720 | 14.849 | 全部成功 |
| 8 | 8/0 | 265.115 | 254.491 | 265.113 | 265.113 | **1.811** | **15.631** | 全部成功 |
| 16 | 16/0 | 530.848 | 502.734 | 530.841 | 530.841 | 1.808 | 15.613 | 全部成功 |
| 32 | 15/17 | 757.084 | 281.439 | 756.934 | 757.059 | 1.189 | 10.263 | VAD 超时，失败率 53.1% |

32 路的 P50 受到 17 个约 281 秒失败请求影响，不能与全成功档位直接比较；该档应以成功率、P95 和失败边界为主要判断依据。

## 4. 模型池和 GPU 状态

测试结束前 `/get_status` 确认：

```json
{
  "speech_campplus": {
    "device": "cuda:0",
    "devices": ["cuda:0", "cuda:0", "cuda:0", "cuda:0"],
    "instances": 4,
    "configured_instances": 4,
    "allocated_mb": 106.35,
    "failed_instances": 0
  },
  "emotion2vec": {
    "device": "cuda:0",
    "devices": ["cuda:0", "cuda:0", "cuda:0", "cuda:0"],
    "instances": 4,
    "configured_instances": 4,
    "allocated_mb": 1258.47,
    "failed_instances": 0
  }
}
```

模型池自身显存登记合计约 `1364.82 MiB`。压测期间 GPU 0 总显存约 `28.4~29.1 GiB`，包含持续运行的 Qwen-asr；未出现 CUDA OOM。高并发采样中 GPU 0 利用率最高约 `96%`，4 个 CAM++ 实例曾同时 busy，说明实例池确实参与并发推理。

## 5. 32 路失败边界

服务日志在 32 路档记录了 17 条 VAD 阶段错误，内容为“VAD 切分超时，请稍后再试”或“VAD 切分排队超时，请稍后再试”，对应接口错误码为 `4014`。模型池状态中的 CAM++ / emotion2vec 均为 4/4 loaded，`failed_instances=0`，因此本轮 32 路失败点是 VAD/整条音频处理排队，而不是模型池实例加载失败。

## 6. 原始数据

- 压测原始 JSON：`/tmp/cam_emo_pool_4_cuda0_20261026/summary.json`
- 分档数据：`/tmp/cam_emo_pool_4_cuda0_20261026/concurrency_2.json`、`concurrency_4.json`、`concurrency_8.json`、`concurrency_16.json`、`concurrency_32.json`
- 压测脚本：[bench_cam_emo_model_pool.py](/home/zhangs/workspace/asr_nvidia/scripts/bench_cam_emo_model_pool.py)

测试结束后 8084 后端已关闭；Qwen-asr 的 8806 服务仍在运行。
