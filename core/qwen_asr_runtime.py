"""Qwen 原始权重运行器；可选 NPU/Transformers 依赖仅在首次加载时导入。"""
from __future__ import annotations

import gc
import io
import re
import threading
import wave
from pathlib import Path


class LocalModelUnavailable(RuntimeError):
    pass


class LocalUnsupportedLanguage(ValueError):
    pass


def decode_wav_bytes(data: bytes):
    """只接受主链路产生的16k单声道PCM16，避免隐式重采样改变段内容。"""
    import numpy as np

    with wave.open(io.BytesIO(data), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (
            16000, 1, 2, "NONE"
        ):
            raise ValueError("Qwen 段音频必须为 16kHz 单声道 PCM16 WAV")
        frames = audio.readframes(audio.getnframes())
        if not frames or len(frames) != audio.getnframes() * 2:
            raise ValueError("Qwen 段音频为空或 PCM 数据不完整")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def move_inputs(inputs, *, device, dtype):
    """只转换浮点精度；token索引和mask不能被统一转成BF16。"""
    import torch

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(move(item) for item in value)
        return value

    return {name: move(value) for name, value in inputs.items()}


class LocalQwenRuntime:
    def __init__(self, model_dir, device, dtype="bfloat16", max_new_tokens=256):
        self.model_dir = str(model_dir)
        self.device = str(device)
        self.dtype = str(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.model = None
        self.processor = None
        self.api = None
        self.failure = None
        self.loading = False
        self.precision_policy = None
        self.parameter_dtype_check = None
        self.memory_allocated_mb = 0.0
        self.memory_reserved_mb = 0.0
        self._load_lock = threading.Lock()

    @property
    def loaded(self):
        return self.model is not None

    def ensure_loaded(self):
        if self.failure:
            raise LocalModelUnavailable(self.failure)
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            if self.failure:
                raise LocalModelUnavailable(self.failure)
            self.loading = True
            try:
                if not Path(self.model_dir).is_dir():
                    raise FileNotFoundError(f"Qwen 本地模型目录不存在: {self.model_dir}")
                if not re.fullmatch(r"npu:\d+", self.device):
                    raise ValueError(f"Qwen 本地推理需要明确的 npu:N 设备: {self.device}")
                import torch
                import torch_npu  # noqa: F401 - 必须注册NPU设备，不以CUDA可见性判定NPU

                self.api = torch.npu
                if not self.api.is_available() or int(self.device.split(":")[1]) >= self.api.device_count():
                    raise RuntimeError(f"Qwen NPU 设备不可用: {self.device}")
                kinds = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
                if self.dtype not in kinds:
                    raise ValueError(f"Qwen dtype 不支持: {self.dtype}")
                # 进程级固定FP32辅助算子的数学模式；helper先选目标卡，
                # 避免device上下文退出时恢复默认卡0，不在请求间来回切换。
                from core.local_inference import configure_npu_precision
                from core.model_loading import model_loading_guard, validate_parameter_dtype
                self.precision_policy = configure_npu_precision(self.api, self.device)
                with self.api.device(self.device):
                    before_allocated = self.api.memory_allocated(self.device)
                    before_reserved = self.api.memory_reserved(self.device)
                    # HF构造期间会改进程全局default dtype；与所有native/CPU
                    # 构造器和其它Qwen实例共用同一guard，完整保护权重加载。
                    with model_loading_guard():
                        # Transformers/FunASR懒导入也会触碰模块解析器，
                        # 必须与构造/加载一起在同一guard内完成。
                        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
                        processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
                        model = Qwen3ASRForConditionalGeneration.from_pretrained(
                            self.model_dir, dtype=kinds[self.dtype], local_files_only=True,
                        )
                        self.parameter_dtype_check = validate_parameter_dtype(model, kinds[self.dtype])
                    # 库有意使用的FP32缓冲区保持原dtype，不统一转为BF16。
                    model = model.to(self.device).eval()
                    self.api.synchronize(self.device)
                    self.processor, self.model = processor, model
                    # 这是加载窗口内的分配器差值，不冒充进程/整卡HBM精确归因。
                    self.memory_allocated_mb = max(0, self.api.memory_allocated(self.device) - before_allocated) / 2**20
                    self.memory_reserved_mb = max(0, self.api.memory_reserved(self.device) - before_reserved) / 2**20
            except Exception as exc:
                self.failure = f"{type(exc).__name__}: {exc}"
                raise LocalModelUnavailable(self.failure) from exc
            finally:
                self.loading = False

    def transcribe(self, audio_bytes, language=None):
        # 解码失败属于请求执行错误，不应被错误地登记为模型加载失败。
        audio = decode_wav_bytes(audio_bytes)
        self.ensure_loaded()
        import torch

        self.api.set_device(self.device)
        with self.api.device(self.device), torch.inference_mode():
            try:
                try:
                    # Qwen的CPU前端Hann window工厂没有显式dtype，
                    # 需避免另一HF构造器的临时全局BF16污染；只保护CPU准备，
                    # 下面的设备传输和NPU前向不持有全局guard。
                    from core.model_loading import model_loading_guard
                    with model_loading_guard(dtype=torch.float32):
                        inputs = self.processor.apply_transcription_request(audio=audio, language=language)
                except ValueError as exc:
                    if "language" in str(exc).lower():
                        raise LocalUnsupportedLanguage(str(exc)) from exc
                    raise
                inputs = move_inputs(inputs, device=self.device, dtype=self.model.dtype)
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                sequences = getattr(output, "sequences", output)
                generated = sequences[:, inputs["input_ids"].shape[1]:]
                text = self.processor.batch_decode(
                    generated, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )[0]
                # 保留语种标记供既有解析器使用，仅剥离已知结束控制token。
                text = re.sub(r"<\|(?:im_end|endoftext|eot_id)\|>", "", text).strip()
                if text and not re.search(r"language\s+[^<]*<asr_text>", text, re.I):
                    raise RuntimeError("Qwen 原始输出缺少语种前缀，不能当作无人声")
                return text
            finally:
                # 线程返回不代表NPU完成；调用方只有收到本方法结果才可归还实例。
                try:
                    self.api.synchronize(self.device)
                except Exception as exc:
                    self.failure = f"NPU synchronize failed: {type(exc).__name__}: {exc}"
                    raise

    def snapshot(self):
        return {
            "device": self.device, "dtype": self.dtype, "loaded": self.loaded,
            "loading": self.loading, "failure": self.failure,
            "precision_policy": self.precision_policy,
            "parameter_dtype_check": self.parameter_dtype_check,
            "memory_allocated_mb": self.memory_allocated_mb,
            "memory_reserved_mb": self.memory_reserved_mb,
            "memory_source": "pytorch_load_delta_estimate",
        }

    def close(self):
        if not self.loaded:
            return
        self.api.set_device(self.device)
        with self.api.device(self.device):
            self.api.synchronize(self.device)
            self.model = self.processor = None
            gc.collect()
            self.api.empty_cache()
        self.memory_allocated_mb = self.memory_reserved_mb = 0.0
