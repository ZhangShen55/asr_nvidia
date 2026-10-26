# Tasks

## 1. 配置项与设备解析

- [x] 1.1 在 `core/config.py` 新增 VAD 进程池相关配置读取：`vad_device`（默认 `"cpu"`）、`vad_pool_size`（默认 8）、`vad_pool_max`（默认 16）、`vad_queue_watermark`（默认 4）、`vad_collapse_timeout_s`（默认 120）、`vad_queue_timeout_s`（默认 120），并移除 `model_device_vad`；验证缺省配置下各默认值与规格一致
- [x] 1.2 在 `core/models.py` 增加 VAD 的设备解析路径：`vad_device` 为 `"cpu"` 时返回 CPU 设备串且不校验 CUDA 可用性，为 `"cuda:N"` 形式时复用现有 GPU 索引校验逻辑；验证 CPU 模式下在无 CUDA 可见的环境也能完成 VAD 设备解析
- [x] 1.3 在 `config.toml` 与 `config.docker.toml` 中同步新增上述 6 个配置项并移除 `[model_device].vad`（保留 `emotion2vec` 与 `speech_campplus`），验证两个文件结构一致且服务能以默认配置启动
- [x] 1.4 补充单元测试覆盖设备解析的三条路径（CPU 默认、显式 CPU、显式 GPU）与配置缺省时的默认值，验证测试全部通过

## 2. VAD worker 进程池

- [x] 2.1 新增 `core/vad_pool.py`：实现 worker 进程的创建与预热（每进程加载一份 FSMN-VAD 实例并跑一次预热推理）、请求分发（传递音频路径、回收分段结果）、以及单次调用的超时与异常处理（worker 崩溃时销毁重建并让受影响请求失败）；验证 4 路并发切分时墙钟不随请求数线性增长
- [x] 2.2 在 `core/vad_pool.py` 实现等待队列与按需扩容：等待队列深度超过 `vad_queue_watermark` 时拉起额外 worker 直到 `vad_pool_max`，空闲超过 `vad_collapse_timeout_s` 的额外 worker 被回收至 `vad_pool_size`；验证突发 16 路请求时 worker 数按需增长、空闲后回落
- [x] 2.3 实现 VAD 排队的超时控制：等待超过 `vad_queue_timeout_s` 时抛出可识别的超时异常；验证超时的请求返回错误码 4014 且其余请求不受影响
- [x] 2.4 为 worker 进程设置进程名 `asr-offline-vad`，验证在进程列表中可区分 worker 与主进程 `asr-offline-model`
- [x] 2.5 补充单元测试覆盖：worker 池的创建与回收、队列深度触发的扩容与回落、排队超时、worker 异常时的请求失败语义；验证测试全部通过

## 3. `core/vad.py` 切换到进程池

- [x] 3.1 将 `core/vad.py` 收敛为纯数据处理：保留 `_parse_vad_result`、`filter_short_segments` 与日志，把实例池管理（`_vad_instances` / `_vad_busy` / `_vad_pool_lock` / `_vad_semaphore`）与 funasr 调用迁移到 `core/vad_pool.py`，`detect_segments()` 改为向进程池提交音频路径并等待分段结果
- [x] 3.2 保证分段结果与变更前逐段一致：以三份完整音频为回归基线（储能学术英语实践.wav 753 段、传播学_马道全.aac 947 段、形式语言与自动机理论_康达周.aac 703 段），验证改造后三者的分段起止时间逐段吻合
- [x] 3.3 保持 VAD 切分参数语义不变（`max_single_segment_time`、`max_end_silence_time`、最短段时长过滤），验证改造前后同一音频的过滤后段数一致
- [x] 3.4 验证 `vad_device = "cuda:0"` 时 VAD 切分结果与 CPU 模式逐段一致，且此时 worker 出现在 `nvidia-smi` 的 GPU 进程列表中

## 4. 段级临时文件生命周期治理

- [x] 4.1 把 `api/routes/asr_qwen_common.py` 的段 clip 落盘从「一次性全量」改为「按需惰性创建」：`_cut` 不再在 VAD 之后批量执行，改为在该段即将发起转写时才创建文件；验证 60 分钟音频处理过程中同时存在的段文件数不超过在途转写段数
- [x] 4.2 在该段转写请求返回后立即删除该段 clip 文件，并从 `ctx.tmp_paths` 中摘除该路径以免重复删除；验证正常转写路径下段文件在请求结束前即已释放
- [x] 4.3 确认 role / emotion 使用的 numpy 数组路径不依赖段文件，删除文件不影响说话人与情绪计算；验证 `showSpk` 与 `showEmotion` 均为 true 时结果与变更前一致
- [x] 4.4 补充单元测试覆盖段文件的三条生命周期路径（转写成功即删、转写失败即删、整体异常时由 `finally` 兜底清理），验证测试全部通过

## 5. 状态接口与运行期可观测

- [x] 5.1 在 `core/status.py` 与 `api/routes/status.py` 中暴露 VAD 的实际运行设备、当前 worker 数与扩容上限；验证 `GET /get_status` 返回这些字段且 VAD 在 CPU 上时显存明细为 0
- [x] 5.2 在服务日志中记录 worker 池的创建、扩容、回收与异常事件，验证日志中可追溯池规模的变化过程

## 6. 效果与性能验证

- [x] 6.1 以 1 / 4 / 8 / 16 并发跑 `test/audio_file/` 中的完整音频（不截取），记录端到端耗时、段/秒与 VAD 阶段耗时占比；验证 VAD 阶段占比相比改造前（83.2%）显著下降
- [x] 6.2 对比改造前后的响应内容：分段数、分段起止时间、分段文本逐段比对，验证切分与转写结果无退化
- [x] 6.3 测量 4 / 8 / 16 worker 三档下的每实例 CPU 核数与聚合 CPU 占用，验证实测值与设计中的 2.2~2.5 核/实例一致，并据此给出容器 CPU 配额建议
- [x] 6.4 压测期间监控 `/tmp` 目录占用峰值，验证段文件改造后临时文件峰值占用相比改造前（219 MB/请求、32 路并发 6.8 GB）显著下降

## 7. 文档

- [x] 7.1 在 `docs/` 下更新压测报告，补充 VAD 进程池改造前后的效率对比、CPU 核数实测与临时文件占用变化
- [x] 7.2 在部署文档中写明 VAD 进程池的 CPU 配额契约（4 worker 建议 16 核、8 worker 建议 24~32 核）、配置项含义与调参建议

## Workflow follow-up

- 镜像构建与部署上线需用户明确确认后另行执行，本变更只完成后端运行、测试与效果比对。
- VAD 改造上线后，若 VAD 仍是高并发下的主要瓶颈，再单独开变更研究整文件切块并行 VAD 的等价性。
- 待 VAD 不再是瓶颈后，再评估是否把 Qwen 服务的 `--api-server-count` 从 4 提到 8。
