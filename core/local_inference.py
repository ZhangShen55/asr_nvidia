"""本地原始网络的 CPU/NPU 边界；设备依赖仅在需要时加载。"""
from contextlib import nullcontext
import importlib
from functools import partial
import os
import threading
import types
from pathlib import Path


class InstanceFailure(RuntimeError):
    """设备执行资源已损坏，模型池必须停止发放此实例。"""


def synchronize_npu(api, device):
    """同步是异步设备错误的暴露点；失败实例不能再回到空闲池。"""
    try:
        api.set_device(device)
        api.synchronize(device)
    except Exception as exc:
        raise InstanceFailure(f"NPU 同步失败 ({device}): {exc}") from exc


def backend_for_device(device: str) -> str:
    if device == "cpu":
        return "pytorch_cpu"
    if isinstance(device, str) and device.startswith("npu:") and device[4:].isdigit():
        return "torch_npu"
    raise ValueError(f"不支持的模型设备 {device!r}；请将旧 CUDA 配置迁移为 cpu 或 npu:N")


def npu_api():
    import torch

    module = importlib.import_module("torch_npu")
    return getattr(torch, "npu", None) or module.npu


_NPU_PRECISION_LOCK = threading.Lock()
_NPU_PRECISION_API = None
_NPU_PRECISION_PID = None
_NPU_PRECISION_STATE = None


def configure_npu_precision(api, device):
    """进程一次固定关闭 FP32 算子的 HF32 内部降精度；从不逐请求切换。

    卷积默认允许 HF32，而 FP32 参数/输入的 dtype 本身看不出内部降精度。
    BF16/FP16 参数仍保持原 dtype，此策略只约束本进程的 FP32 算子数学模式。
    先选目标卡，避免任何 option/context 的首次访问隐式初始化默认卡 0。
    """
    global _NPU_PRECISION_API, _NPU_PRECISION_PID, _NPU_PRECISION_STATE
    if backend_for_device(device) != "torch_npu":
        raise ValueError("NPU 精度初始化需要 npu:N")
    api.set_device(device)
    with _NPU_PRECISION_LOCK:
        if _NPU_PRECISION_API is api and _NPU_PRECISION_PID == os.getpid():
            return dict(_NPU_PRECISION_STATE)
        try:
            api.conv.allow_hf32 = False
            api.matmul.allow_hf32 = False
            state = {"mode": "fp32_ops_hf32_disabled", "conv_allow_hf32": api.conv.allow_hf32,
                     "matmul_allow_hf32": api.matmul.allow_hf32}
            if state["conv_allow_hf32"] is not False or state["matmul_allow_hf32"] is not False:
                raise RuntimeError(f"HF32 禁用未生效: {state}")
        except Exception as exc:
            raise InstanceFailure(f"NPU 精度初始化失败 ({device}): {exc}") from exc
        _NPU_PRECISION_API, _NPU_PRECISION_PID = api, os.getpid()
        _NPU_PRECISION_STATE = state
        return dict(state)


def npu_scope(api, device):
    # torch_npu.device.__enter__ may query current_device and initialize device 0
    # before switching; select the configured device before constructing context.
    try:
        configure_npu_precision(api, device)
        return api.device(device)
    except Exception as exc:
        raise InstanceFailure(f"NPU 设备选择失败 ({device}): {exc}") from exc


def device_scope(device):
    backend = backend_for_device(device)
    return npu_scope(npu_api(), device) if backend == "torch_npu" else nullcontext()


def waveform_tensor(audio, sample_rate):
    """模型输入契约；不隐式重采样、裁剪或补零。"""
    import numpy as np
    import torch

    if sample_rate != 16000:
        raise ValueError("本地模型要求 16 kHz 音频")
    if not isinstance(audio, np.ndarray) or audio.dtype != np.float32:
        raise ValueError("音频必须为 float32 numpy 数组")
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("音频必须为非空、有限值的单声道数组")
    return torch.from_numpy(np.ascontiguousarray(audio))


def campplus_features(waveform):
    import torch
    from core.model_loading import model_loading_guard

    # Kaldi mel-bank constants use process-default dtype before conversion to
    # waveform dtype. Casting the result cannot recover BF16-rounded constants.
    with model_loading_guard(dtype=torch.float32):
        from torchaudio.compliance.kaldi import fbank
        features = fbank(waveform.unsqueeze(0), num_mel_bins=80)
        return (features - features.mean(dim=0, keepdim=True)).unsqueeze(0).contiguous()


def segment_pooling(x, seg_len=100, stype="avg"):
    """等价于 ceil_mode 池化后展开；尾块不能补零后错误计数/取最大值。"""
    import torch

    if seg_len <= 0 or x.shape[-1] == 0:
        raise ValueError("池化长度和输入必须非空")
    if stype not in ("avg", "max"):
        raise ValueError("Wrong segment pooling type.")
    length = x.shape[-1]
    whole = length // seg_len
    parts = []
    if whole:
        blocks = x[..., :whole * seg_len].reshape(*x.shape[:-1], whole, seg_len)
        pooled = blocks.sum(dim=-1) / seg_len if stype == "avg" else blocks.amax(dim=-1)
        parts.append(pooled)
    if length % seg_len:
        tail = x[..., whole * seg_len:]
        pooled = tail.sum(dim=-1, keepdim=True) / tail.shape[-1] if stype == "avg" else tail.amax(dim=-1, keepdim=True)
        parts.append(pooled)
    pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    return pooled.unsqueeze(-1).expand(*pooled.shape, seg_len).reshape(*x.shape[:-1], -1)[..., :length]


def _native_seg_pooling(self, x, seg_len=100, stype="avg"):
    return segment_pooling(x, seg_len, stype)


def adapt_campplus_pooling(model):
    """只绑定本实例方法，不改安装包、共享类或 checkpoint。"""
    for module in model.modules():
        if callable(getattr(module, "seg_pooling", None)):
            module.seg_pooling = types.MethodType(_native_seg_pooling, module)
    return model


def _exact_gelu_forward(self, x):
    """原生 NPU gelu 忽略 approximate；用已注册 GeluV2 保持 none/erf 语义。"""
    import torch
    import torch.nn.functional as F

    if x.device.type == "npu":
        return torch.ops.npu.npu_gelu(x, approximate="none")
    return F.gelu(x, approximate="none")


def adapt_emotion_gelu(model):
    """只适配当前 emotion2vec 实例的 GELU(none)，保留 CPU 与所有权重。"""
    import torch

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.GELU) and module.approximate == "none":
            module.forward = types.MethodType(_exact_gelu_forward, module)
            count += 1
    model._npu_exact_gelu_count = count
    return model


def _fp32_bias_factory(factory, *args, **kwargs):
    import torch
    from core.model_loading import model_loading_guard

    # Only bias/cache construction is protected, not the surrounding device
    # forward. Keep the original factory and requested result dtype/device.
    with model_loading_guard(dtype=torch.float32):
        return factory(*args, **kwargs)


def adapt_emotion_dtype_boundaries(model):
    """Protect this instance's ALIBI factory from HF's temporary global BF16."""
    count = 0
    for module in model.modules():
        factory = getattr(module, "get_alibi_bias", None)
        if callable(factory):
            if not getattr(module, "_fp32_bias_factory_adapted", False):
                module.get_alibi_bias = partial(_fp32_bias_factory, factory)
                module._fp32_bias_factory_adapted = True
            count += 1
    model._fp32_bias_factory_count = count
    return model


def emotion_labels(model_dir):
    labels = (Path(model_dir) / "tokens.txt").read_text().splitlines()
    if len(labels) != 9 or [i for i, x in enumerate(labels) if x.startswith("unuse")] != [1, 6]:
        raise ValueError("emotion2vec 分类头要求 9 类，unuse 位于 1、6")
    return [x for x in labels if not x.startswith("unuse")]


def validate_fp32_network(network):
    """Validate actual storage, including floating buffers, before reporting FP32."""
    import torch

    if not isinstance(network, torch.nn.Module):
        raise ValueError("无法验证非 nn.Module 网络的 FP32 参数")
    floating = []
    mismatches = []
    for category, tensors in (("parameter", network.named_parameters()), ("buffer", network.named_buffers())):
        for name, value in tensors:
            if value.is_floating_point() or value.is_complex():
                floating.append(name)
                if value.dtype != torch.float32:
                    mismatches.append(f"{category}:{name}={value.dtype}")
    if mismatches:
        raise ValueError("模型声明 FP32 但实际参数/缓冲精度不符: " + ", ".join(mismatches))
    return bool(floating)


def load_native_network(name, model_dir):
    """直接加载原始网络，绕过 AutoModel 仅按 CUDA 可用性选设备的逻辑。"""
    import torch
    import yaml

    from core.model_loading import model_loading_guard

    with model_loading_guard(dtype=torch.float32):
        root = Path(model_dir)
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        if name == "speech_campplus":
            from funasr.models.campplus.model import CAMPPlus
            net = CAMPPlus(**cfg["model_conf"])
            weight = root / "campplus_cn_en_common.pt"
        elif name == "emotion2vec":
            from funasr.models.emotion2vec.model import Emotion2vec
            net = Emotion2vec(model_conf=cfg["model_conf"], vocab_size=9)
            weight = root / "model.pt"
        else:
            raise ValueError(f"未知本地增强模型: {name}")
        # Casting after load_state_dict would already have rounded FP32 weights if
        # construction accidentally used BF16. Promote storage before copying.
        net = net.float()
        state = torch.load(weight, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state:
                state = state[key]
        scopes = list(cfg.get("scope_map") or []) + ["module.", "none"]
        if len(scopes) % 2:
            raise ValueError("模型 scope_map 必须为源/目标前缀对")
        mapped = {}
        for key in net.state_dict():
            if key in state:
                mapped[key] = state[key]
            for source, target in zip(scopes[::2], scopes[1::2]):
                source = "" if source is None or str(source).lower() == "none" else str(source)
                target = "" if target is None or str(target).lower() == "none" else str(target)
                candidate = source + key[len(target):] if key.startswith(target) else None
                if candidate in state:
                    mapped[key] = state[candidate]
        missing = net.load_state_dict(mapped, strict=False).missing_keys
        # 发布权重不含训练重建 decoder；features_only 路径不执行它。
        required_missing = [key for key in missing if not (
            name == "emotion2vec" and key.startswith("modality_encoders.AUDIO.decoder.")
        )]
        if required_missing:
            raise ValueError(f"模型缺失推理参数: {required_missing}")
        if name == "speech_campplus":
            adapt_campplus_pooling(net)
        else:
            adapt_emotion_gelu(net)
            adapt_emotion_dtype_boundaries(net)
        if not validate_fp32_network(net):
            raise ValueError("本地原始网络缺少可验证的浮点参数/缓冲")
        return net.eval()


class InferenceRunner:
    healthy = True
    failure = None

    def close(self):
        pass


class TorchRunner(InferenceRunner):
    """CPU 原始 FunASR 兼容句柄；精度不能受旧 CUDA 开关隐式影响。"""
    def __init__(self, model, device, *, fp16=False):
        if device != "cpu":
            raise ValueError("NPU 请使用 TorchNpuRunner")
        self.model = model
        self.device = device
        self.effective_backend = "pytorch_cpu"
        import torch
        network = getattr(model, "model", model)
        # Compatibility/fake handles without nn.Module storage have no verified
        # precision; do not turn an assumption into a reported FP32 fact.
        self.actual_precision = ("fp32" if isinstance(network, torch.nn.Module)
                                 and validate_fp32_network(network) else None)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def generate(self, **kwargs):
        import torch
        from core.model_loading import model_loading_guard
        with model_loading_guard(dtype=torch.float32), torch.inference_mode():
            result = self.model.generate(**kwargs)
            for payload in result or []:
                embedding = payload.get("spk_embedding")
                if isinstance(embedding, torch.Tensor):
                    payload["spk_embedding"] = embedding.detach().float().cpu()
            return result

    def close(self):
        self.model = None


class TorchNpuRunner(InferenceRunner):
    """FP32 原生正确性基线；模型池独占本实例直到同步完成。"""
    effective_backend = "torch_npu"
    actual_precision = None
    math_mode = "fp32_ops_hf32_disabled"

    def __init__(self, name, model_dir, device):
        self.device = device
        self.name = name
        self.labels = emotion_labels(model_dir) if name == "emotion2vec" else None
        self.api = npu_api()
        with npu_scope(self.api, device):
            self.model = load_native_network(name, model_dir).to(device)
            if not validate_fp32_network(self.model):
                raise ValueError("NPU 原始网络缺少可验证的浮点参数/缓冲")
            self.actual_precision = "fp32"
            self.operator_adaptations = {
                "campplus_segment_pooling": name == "speech_campplus",
                "emotion2vec_exact_gelu_modules": getattr(self.model, "_npu_exact_gelu_count", 0),
                "emotion2vec_fp32_bias_factories": getattr(self.model, "_fp32_bias_factory_count", 0),
            }
            synchronize_npu(self.api, device)

    def generate(self, *, input, fs=16000, **kwargs):
        import torch
        import torch.nn.functional as F

        wave = waveform_tensor(input, fs)
        with npu_scope(self.api, self.device), torch.inference_mode():
            try:
                if self.name == "speech_campplus":
                    embedding = self.model(campplus_features(wave).to(self.device))
                    return [{"spk_embedding": embedding.detach().float().cpu()}]
                source = wave.to(self.device)
                if self.model.cfg.normalize:
                    source = F.layer_norm(source, source.shape)
                features = self.model.extract_features(source.view(1, -1), padding_mask=None)["x"]
                logits = self.model.proj(features.mean(dim=1))
                scores = torch.softmax(logits[:, [0, 2, 3, 4, 5, 7, 8]], dim=-1)
                return [{"labels": self.labels, "scores": scores[0].detach().float().cpu().tolist()}]
            except RuntimeError as exc:
                if any(text in str(exc).lower() for text in ("acl", "npu", "device-side", "out of memory")):
                    self.healthy = False
                    self.failure = str(exc)
                    raise InstanceFailure(str(exc)) from exc
                raise
            finally:
                synchronize_npu(self.api, self.device)

    def close(self):
        try:
            with npu_scope(self.api, self.device):
                synchronize_npu(self.api, self.device)
        finally:
            self.model = None


class NativeVadRunner(InferenceRunner):
    """固定 worker 的 FSMN 适配；复用 FunASR 前后处理，不走 CUDA 选卡。"""
    effective_backend = "torch_npu"
    actual_precision = "fp32"
    math_mode = "fp32_ops_hf32_disabled"

    def __init__(self, model_dir, device, vad_kwargs):
        import torch
        import yaml
        from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming
        from funasr.frontends.wav_frontend import WavFrontendOnline

        root = Path(model_dir)
        cfg = yaml.safe_load((root / "config.yaml").read_text())
        # 与 AutoModel 的 model_conf + 顶层 config 合并保持一致；不借迁移改变阈值。
        model_conf = dict(cfg["model_conf"])
        model_conf.update(cfg)
        net = FsmnVADStreaming(**model_conf)
        state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state:
                state = state[key]
        net.load_state_dict(state, strict=True)
        self.effective_backend = backend_for_device(device)
        self.math_mode = "fp32_ops_hf32_disabled" if self.effective_backend == "torch_npu" else "ieee_fp32"
        self.api = npu_api() if self.effective_backend == "torch_npu" else None
        self.device = device
        with npu_scope(self.api, device) if self.api is not None else nullcontext():
            self.model = net.eval().to(device)
            if self.api is not None:
                synchronize_npu(self.api, device)
        frontend_conf = dict(cfg["frontend_conf"])
        if (root / "am.mvn").is_file():
            frontend_conf["cmvn_file"] = str(root / "am.mvn")
        self.frontend = WavFrontendOnline(**frontend_conf)
        self.kwargs = dict(cfg, device=device, vad_kwargs=dict(vad_kwargs))

    def generate(self, *, input, cache=None, vad_kwargs=None, **kwargs):
        import torch
        options = dict(self.kwargs)
        options.update(kwargs)
        options["vad_kwargs"] = dict(vad_kwargs or options.get("vad_kwargs", {}))
        options.pop("frontend", None)
        with (npu_scope(self.api, self.device) if self.api is not None else nullcontext()), torch.inference_mode():
            try:
                results, _metadata = self.model.inference(
                    data_in=[input], key=[Path(str(input)).stem], frontend=self.frontend,
                    cache={} if cache is None else cache, **options,
                )
                return results
            except RuntimeError as exc:
                if self.api is not None and any(text in str(exc).lower() for text in ("acl", "npu", "device-side", "out of memory")):
                    raise InstanceFailure(str(exc)) from exc
                raise
            finally:
                if self.api is not None:
                    synchronize_npu(self.api, self.device)

    def close(self):
        try:
            with npu_scope(self.api, self.device) if self.api is not None else nullcontext():
                if self.api is not None:
                    synchronize_npu(self.api, self.device)
        finally:
            self.model = None
            self.frontend = None


def generate_local(model, device, **kwargs):
    """兼容替身/外部句柄，设备工作完成后才允许线程返回并归还租约。"""
    if isinstance(model, InferenceRunner):
        return model.generate(**kwargs)
    import torch
    with device_scope(device), torch.inference_mode():
        try:
            return model.generate(**kwargs)
        finally:
            if backend_for_device(device) == "torch_npu":
                synchronize_npu(npu_api(), device)
