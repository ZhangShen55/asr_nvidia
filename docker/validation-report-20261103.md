# CUDA Cython Docker v3 验收报告

## 镜像与运行环境

- 镜像：`asr_offline:v3_20261103040410`
- Digest：`sha256:4226bc34f5310bf9c9045f6a0ce3e85dc35941e33f0b398eb31df14583ca7f98`
- 基础镜像：`docker.1ms.run/nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
- GPU：RTX 4090 / RTX 3090，NVIDIA Driver `580.126.09`
- CUDA：`12.4`
- TensorRT：`10.7.0.post1`
- 运行配置：`CONFIG_PATH=/config.toml`，宿主配置只读挂载；默认配置为 `audio_concurrency=8`、`audio_spk_emo_concurrency=4`、VAD `cpu,cpu`。

最终默认容器启动后 `/get_status` 返回 `cuda-tensorrt-fp16`，CAM++ 3 个实例和 emotion2vec 1 个实例均为 `effective_backend=tensorrt`、`actual_precision=fp16`、`health=healthy`；VAD 2 个 worker 均 ready。

## 接口验收

- `GET /get_status`：HTTP 200。
- `POST /v3/seacraft_asr` A 请求：HTTP 200，转写完成，`role=null`、`emotion=null`。
- `POST /v3/seacraft_asr` B 请求（`showSpk=true`、`showEmotion=true`）：HTTP 200，返回 `role=spk_0`、`emotion=平淡`。
- `POST /v2/seacraft_asr`：HTTP 404。
- B 并发 8 路：4 路成功、4 路返回 `code=4017`，消息为 `spk_emo` 队列已满；最终总槽位和 B 槽位均恢复为 0。

## 并发准入验收

`test/tensorrt-backends-20261102/docker-v3-admission-matrix.json` 记录了 `8/4` 配置下的原子准入结果：

| 已占用 | 新 B | 新 A |
| --- | --- | --- |
| 3A+4B（总计 7，B=4） | `4017 spk_emo_full` | 允许，占用到总计 8 |
| 5A+3B（总计 8） | `4013 total_full` | `4013 total_full` |
| 8A（总计 8） | `4013 total_full` | `4013 total_full` |

当 `open_v2_role=false`、`open_v2_emotion=false` 且使用非 CUDA 强制 profile 进行能力关闭验证时：A 请求正常转写并返回两个字段 `null`；携带增强参数的 B 请求返回 `code=4018`，不会加载或调用增强模型。结果见 `test/tensorrt-backends-20261102/docker-v3-disabled-capability.json`。CUDA 镜像通过内部 `ASR_DEPLOYMENT_PROFILE` 强制要求两个增强模型启用，关闭时会在启动闸门阶段拒绝就绪。

## 八份完整音频 TensorRT fp16 验收

使用最终镜像，临时将 B 容量设为 8 以进行质量和并发测试，提交 8 份固定完整音频；结果见 `test/tensorrt-backends-20261102/docker-v3-full8-acceptance.json`。

- 成功：`8/8`
- 失败：`0`
- `role` 空值段数：`0`
- `emotion` 空值段数：`0`
- P50：`94.995s`
- P95：`112.480s`
- 批次墙钟：`113.747s`
- 采样到的峰值：总音频并发 `8`、B 并发 `8`
- GPU0 采样显存：约 `24801–24809 MiB`
- 所有请求最终释放槽位，状态回到 `active_total=0`、`active_B=0`。

验收完成后已停止临时 `B=8` 容器，并恢复默认 `B=4` 配置启动容器。

## 镜像内容与自动化检查

新镜像初始文件清单确认：`core/`、`api/`、`utils/`、`entity/` 仅保留 `__init__.py` 和 Cython `.so`；无业务 `.c`、无 `__pycache__`、无测试文件；模型目录仅包含 VAD、CAM++、emotion2vec 三个必需目录及 fp16 TensorRT 资产。

- `docker compose config`：通过。
- TensorRT 资产、manifest、SHA256、GPU/CUDA/TensorRT 兼容性：通过。
- 全量 pytest：`253 passed, 21 warnings`。
- Cython runtime import smoke test：通过。

启动日志和状态快照分别保存在 `test/tensorrt-backends-20261102/docker-v3-startup.log` 与 `test/tensorrt-backends-20261102/docker-v3-status.json`。
