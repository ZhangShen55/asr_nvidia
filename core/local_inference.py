"""本地增强模型的设备边界；可选依赖只在对应运行器中导入。"""
from contextlib import nullcontext
import importlib
from pathlib import Path


class InstanceFailure(RuntimeError):
    """执行资源已损坏，模型池必须停止发放此实例。"""


def backend_for_device(device: str, use_tensorrt: bool = False) -> str:
    if device == "cpu":
        return "pytorch_cpu"
    if device.startswith("npu:"):
        return "torch_npu"
    if device.startswith("cuda:"):
        return "tensorrt" if use_tensorrt else "pytorch_cuda"
    raise ValueError(f"不支持的模型设备: {device}")


def npu_api():
    import torch

    module = importlib.import_module("torch_npu")
    return getattr(torch, "npu", None) or module.npu


def device_scope(device):
    if device.startswith("cuda:"):
        import torch

        return torch.cuda.device(device)
    if device.startswith("npu:"):
        return npu_api().device(device)
    return nullcontext()


def waveform_tensor(audio, sample_rate):
    """engine/原始网络输入契约；不隐式重采样、裁剪或补零。"""
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
    from torchaudio.compliance.kaldi import fbank

    features = fbank(waveform.unsqueeze(0), num_mel_bins=80)
    return (features - features.mean(dim=0, keepdim=True)).unsqueeze(0).contiguous()


def emotion_labels(model_dir):
    labels = (Path(model_dir) / "tokens.txt").read_text().splitlines()
    if len(labels) != 9 or [i for i, x in enumerate(labels) if x.startswith("unuse")] != [1, 6]:
        raise ValueError("emotion2vec 分类头要求 9 类，unuse 位于 1、6")
    return [x for x in labels if not x.startswith("unuse")]


def load_native_network(name, model_dir):
    """绕过 FunASR AutoModel 中基于 CUDA 的设备回退，用于 NPU/导出。"""
    import torch
    import yaml

    root = Path(model_dir)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    if name == "speech_campplus":
        from funasr.models.campplus.model import CAMPPlus

        net = CAMPPlus(**cfg["model_conf"])
        weight = root / "campplus_cn_en_common.pt"
    else:
        from funasr.models.emotion2vec.model import Emotion2vec

        net = Emotion2vec(model_conf=cfg["model_conf"], vocab_size=9)
        weight = root / "model.pt"
    state = torch.load(weight, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "model"):
        if key in state:
            state = state[key]
    # emotion2vec 原始训练 checkpoint 使用 d2v_model.，配置给出 scope_map；
    # 按目标键映射，不把缺失推理参数静默留为随机初始化。
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
    # 发布权重不含训练重建 decoder；extract_features(features_only=True) 不执行它。
    required_missing = [key for key in missing if not (
        name == "emotion2vec" and key.startswith("modality_encoders.AUDIO.decoder.")
    )]
    if required_missing:
        raise ValueError(f"模型缺失推理参数: {required_missing}")
    return net.eval()


class InferenceRunner:
    healthy = True
    failure = None
    engine_info = None

    def close(self):
        pass


class TorchRunner(InferenceRunner):
    def __init__(self, model, device, *, fp16=False):
        self.model = model
        self.device = device
        self.effective_backend = backend_for_device(device)
        self.actual_precision = "fp16" if fp16 and device.startswith("cuda:") else "fp32"

    def __getattr__(self, name):
        # 保留既有诊断调用方读取 FunASR kwargs 等属性的能力。
        return getattr(self.model, name)

    def generate(self, **kwargs):
        import torch

        with device_scope(self.device):
            amp = (torch.amp.autocast("cuda", dtype=torch.float16)
                   if self.actual_precision == "fp16" else nullcontext())
            try:
                with torch.inference_mode(), amp:
                    result = self.model.generate(**kwargs)
                    for payload in result or []:
                        embedding = payload.get("spk_embedding")
                        if isinstance(embedding, torch.Tensor):
                            payload["spk_embedding"] = embedding.detach().float().cpu()
                    return result
            finally:
                if self.device.startswith("cuda:"):
                    torch.cuda.synchronize(self.device)

    def close(self):
        self.model = None


class TorchNpuRunner(InferenceRunner):
    """使用 torch_npu 原始网络；当前无 NPU 实机验证，默认 FP32。"""
    effective_backend = "torch_npu"
    actual_precision = "fp32"

    def __init__(self, name, model_dir, device):
        self.device = device
        self.name = name
        self.api = npu_api()
        with self.api.device(device):
            self.model = load_native_network(name, model_dir).to(device)
        self.labels = emotion_labels(model_dir) if name == "emotion2vec" else None

    def generate(self, *, input, fs=16000, **kwargs):
        import torch
        import torch.nn.functional as F

        wave = waveform_tensor(input, fs)
        with self.api.device(self.device), torch.inference_mode():
            try:
                if self.name == "speech_campplus":
                    embedding = self.model(campplus_features(wave).to(self.device))
                    return [{"spk_embedding": embedding.float().cpu()}]
                source = wave.to(self.device)
                if self.model.cfg.normalize:
                    source = F.layer_norm(source, source.shape)
                features = self.model.extract_features(source.view(1, -1), padding_mask=None)["x"]
                logits = self.model.proj(features.mean(dim=1))
                scores = torch.softmax(logits[:, [0, 2, 3, 4, 5, 7, 8]], dim=-1)
                return [{"labels": self.labels, "scores": scores[0].float().cpu().tolist()}]
            finally:
                self.api.synchronize(self.device)

    def close(self):
        with self.api.device(self.device):
            self.api.synchronize(self.device)
            self.model = None


def generate_local(model, device, **kwargs):
    """统一调用入口；兼容外部注入的旧式 FunASR 句柄。"""
    if isinstance(model, InferenceRunner):
        return model.generate(**kwargs)
    import torch

    with device_scope(device):
        amp = torch.amp.autocast("cuda", dtype=torch.float16) if device.startswith("cuda:") else nullcontext()
        with amp:
            return model.generate(**kwargs)
