# CAM++ / emotion2vec 推理后端

四项参数与实例设备数组一起放在正式的 `[model_device_pool]` 表，修改后重启生效：

```toml
[model_device_pool]
speech_campplus_devices = ["cuda:0", "cuda:0", "cuda:0"]
emotion2vec_devices = ["cuda:0"]
speech_campplus_use_tensorrt = false
speech_campplus_tensorrt_precision = "fp16"
emotion2vec_use_tensorrt = false
emotion2vec_tensorrt_precision = "fp16"
```

两个开关必须为 TOML 布尔值，默认 false；精度只接受 `fp16` / `fp32`，默认 `fp16`。
启动时校验全部四项，包括当前未使用的精度配置；字符串 `"false"`、INT8 等非法值直接报字段错误。
精度项只选择 TensorRT engine，保留 PyTorch 的原有精度策略。两个模型可以独立启用或回滚。

每个数组元素代表一个实例，重复设备不合并。CUDA 实例按对应开关选择 TensorRT / PyTorch CUDA；
`npu:N` 固定为 torch_npu，`cpu` 固定为 PyTorch CPU，均忽略 TensorRT 开关及 engine 精度。
`features.open_v2_role` / `open_v2_emotion` 决定能力是否开放，请求参数决定是否调用；后端开关不覆盖它们。

## 安装与交付

CUDA TensorRT 运行环境安装 `pip install -r requirements-tensorrt.txt`；仅导出/构建环境安装
`requirements-tensorrt-export.txt`，服务不要求 ONNX。与目标 CANN 配套的 PyTorch/torch_npu
由 NPU 环境单独提供，不把 `torch_npu` 加入通用 NVIDIA 依赖。NPU 运行器使用原始网络、
FP32 和独立设备 API；当前只验证选择/依赖隔离，尚未进行 NPU 实机质量及性能验收。

模型目录保持原权重，并按以下结构存放可选资产：

```text
<model_paths.spk_model_dir 或 emotion_model_dir>/
  原始权重、config.yaml、tokens.txt（情绪模型）
  tensorrt/
    fp32/engine.plan
    fp32/manifest.json
    fp16/engine.plan
    fp16/manifest.json
    sources/                 # 带 SHA256 的导出/构建来源快照
```

加载时校验 schema、模型/精度、权重/配置/标签及导出来源 SHA256、engine SHA256、I/O/profile，
并检查实际实例 GPU 架构和 TensorRT/CUDA 版本。当前交付针对 RTX 4090 / SM89、
TensorRT 10.7.0.post1、PyTorch 2.6.0+cu124 / CUDA 12.4，不承诺在 RTX 3090、其他运行时或 NPU 上复用。
旧候选保留 `legacy-manifest.json`；原实验未记录精确 wall-clock 构建起点，正式 manifest
明确以原始 engine 文件 mtime 作为旧候选时间来源，不伪造精确构建时间。

`python scripts/check_tensorrt_assets.py --config config.toml` 可在没有 GPU/TensorRT 包时验证交付完整性；
它只要求已开放能力且选择 CUDA TensorRT 的模型资产，不代表实际运行时兼容检查。
Docker 默认不安装 TensorRT，可通过 `--build-arg ENABLE_TENSORRT=1` 选择安装；必须先核验
基础系统/glibc、CUDA 和 PyTorch 环境。当前 CentOS 7 / CUDA 12.1.1 基础镜像不等于本次已验证的
CUDA 12.4 主机环境，镜像的实际 engine 兼容性须在目标环境验收或重新构建，不能仅复制 engine 即宣称可用。

## 输入边界与资源

CAM++ 每段提取与 FunASR 一致的 80 维、去均值 fbank；输入 `[1,T,80]`，profile T=50/200/3000，
输出 `[1,192]`。不足 1 秒仍由调用侧跳过。构建保留 52 个 AveragePool 的
`ceil_mode=1` / `count_include_pad=0` 修正。emotion2vec 输入单声道 16 kHz 原始波形
`[1,N]`，profile N=16000/48000/160000，输出 `[1,9]` logits 和 `[1,7]` scores；
包含归一化、特征提取、去额外 token、时间池化、分类头和去 unuse 后的 Softmax。
1～10 秒外仍返回原有“平淡”，并且先校验功能开关。两个模型固定 batch=1，FP16 engine 的 I/O 仍为 FP32。

越界、非法输入与普通段级失败明确记录并沿用 null 契约，不补零/截断/跨 VAD 段合并，也不静默切后端。
每个 TensorRT 实例独立持有 context、stream、可写缓冲；执行、CPU 输出读取及同步完成后才结束租约。
重复取消仍等待已启动工作结束；上下文等实例故障从池中移除，唤醒等待者；加载失败不占健康容量。

## 状态与回滚

`/get_status` 的 `localModels.<模型>.instance_details` 提供 configured_device、configured_backend、
effective_backend、configured_engine_precision、actual_precision、health、failure、engine 和 memory。
未加载实例的实际后端/精度为 null，配置选择另列；查询不触发加载。A/B 音频状态、VAD 状态和模型池
busy/waiting 各自保持含义。配置 fp16 不会被当作 NPU/CPU 的实际计算精度。

TensorRT 完整原生分配无法仅用 PyTorch 分配器计数，因此池的 `allocated_mb` 在含 TensorRT 时为 null，
`torch_allocated_mb` 只表示 PyTorch 加载差额；每实例另列 context 大小、engine 文件大小及加载时设备总占用，
并说明无法获取原生分配总量。性能采样同时记录 nvidia-smi 的设备显存与 GPU 利用率。

engine 缺失、损坏、不兼容或初始化失败时实例不可用；全部不可用返回既有 CAM++ 4015 / emotion2vec 4016。
4018 仍仅表示同时请求的两项能力都被关闭。关闭故障模型的 `use_tensorrt` 并重启，即显式回滚该模型，
另一个模型的配置独立保留。

## 工具与验收

```bash
python scripts/export_local_tensorrt.py --model speech_campplus \
  --model-dir model/speech_campplus_sv_zh_en_16k-common_advanced --precision fp32 --device cuda:0
python scripts/export_local_tensorrt.py --model emotion2vec \
  --model-dir model/emotion2vec_plus_large --precision fp16 --device cuda:0
```

已有 engine 时默认拒绝覆盖；只有显式 `--rebuild` 才重建。`--adopt-from <原独立实验目录>` 用于核验旧候选
与原 engine/ONNX/权重证据后补齐 manifest，保留 engine 字节不变。先验证 FP32，再验证 FP16。

`scripts/validate_local_tensorrt.py` 对固定裁剪验证真实运行器、fbank、数值/标签、profile 和多实例隔离。
`scripts/bench_tensorrt_backends.py` 在临时端口、临时配置启动服务，预热后比较八门不同课程的完整 WAV：
CAM++ 3 实例、emotion2vec 1 实例、两个请求开关 true、总/B 上限 8、段并发 16、Qwen 全局并发 128。
保留正常服务懒加载与生产默认 B=4；基准使用八路共同起跑，并保存逐请求结果、状态/资源采样及原始耗时。
P50/P95 使用成功完整请求的客户端端到端耗时和线性插值，N=8；失败轮次明确不完整，不用单段模型加速替代接口收益。

## 本次验证结果与质量边界

实测及复算方法见 [八路完整课堂验证报告](cam-emo-tensorrt-backends-validation-20261102.md)。
同卡 Qwen、CAM++ 三实例和 emotion2vec 一实例下，PyTorch / TensorRT FP32 / FP16
的完整请求 P50 为 407.00 / 115.82 / 97.02 秒，P95 为 428.40 / 123.92 / 107.22 秒；
各组均 8/8 成功，属于一轮 N=8 的结果，不能直接外推生产尾延迟。

CAM++ 原 33 个裁剪及新增 48 个课堂片段之外，已比较八份完整课堂的输出聚类。
FP16 在共同输出片段上八门课的聚类划分均与当前 PyTorch FP16 一致（ARI=1）；
FP32 的高等传热学从 3 簇变成 2 簇。固定同一课堂 494 段输入后，纯 PyTorch FP32
同样为 2 簇且与 TensorRT FP32 的划分一致，说明精度本身也会改变聚类边界。
没有人工角色真值，不能据此判断 2 簇或 3 簇哪个更准确，也不承诺其他录音完全一致。

emotion2vec 原 27 个及新增 48 个片段在两种 engine 下均保持标签一致；针对整课差异
另选的 11 个边界片段则在 FP32 / FP16 下分别有 3 / 9 个标签翻转，
FP16 最大概率绝对差约 0.03974（3.974 个百分点）。这组是刻意选取的差异样本，
不是随机错误率。整课共同输出片段中，FP32 为 3/4428 个业务情绪不同，FP16 为 9/4430 个不同；
输出中含无需情绪模型推理的短段，“1～10 秒内”子集指标在报告中另列。
保留原业务映射和阈值，不为提高一致率而静默修改标签或聚类策略。

当前默认开关仍为 false。已完成服务集成和本机实测；NPU 实机及目标 Docker 镜像验收仍需独立环境验证。
