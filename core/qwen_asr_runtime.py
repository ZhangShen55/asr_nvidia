"""Qwen 原始权重运行器；可选 NPU/Transformers 依赖仅在首次加载时导入。"""
from __future__ import annotations

import gc
import io
import re
import threading
import wave
import time
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


def collate_independent_inputs(singles, *, pad_token_id=None):
    """保留各single全部特征后再补batch形状，不能在STFT前补PCM。"""
    import torch

    if not singles:
        raise ValueError("Qwen独立前端需要非空成员")
    if len(singles) == 1:
        # 单段保持处理器原始输入，避免为了batch改变已验收边界。
        return singles[0]
    expected = {"input_ids", "attention_mask", "input_features", "input_features_mask"}
    if type(pad_token_id) is not int:
        raise ValueError("Qwen多段collation需要明确的tokenizer.pad_token_id")
    for item in singles:
        if set(item) != expected:
            raise ValueError(f"Qwen processor输入键不符合锁定四字段: {sorted(item)}")
        if not all(isinstance(value, torch.Tensor) for value in item.values()):
            raise ValueError("Qwen processor四字段必须是Tensor")
        ids, text_mask = item["input_ids"], item["attention_mask"]
        features, feature_mask = item["input_features"], item["input_features_mask"]
        if (ids.ndim != 2 or ids.shape[0] != 1 or text_mask.shape != ids.shape
                or features.ndim != 3 or features.shape[0] != 1
                or feature_mask.ndim != 2 or feature_mask.shape != (1, features.shape[-1])):
            raise ValueError("Qwen single输入shape不符合B=1的text/features/mask契约")
        if ids.is_floating_point() or ids.dtype == torch.bool or not features.is_floating_point():
            raise ValueError("Qwen token索引必须为整数，音频特征必须为浮点")
        for mask in (text_mask, feature_mask):
            if mask.is_floating_point() or bool(((mask != 0) & (mask != 1)).any()):
                raise ValueError("Qwen mask必须为整数或bool二值Tensor")
        if not bool(text_mask.bool().any()):
            raise ValueError("Qwen single text没有有效token")
        for key in expected:
            if item[key].dtype != singles[0][key].dtype or item[key].device != singles[0][key].device:
                raise ValueError(f"Qwen成员{key}的dtype/device不同，不能静默转换")
        if features.shape[1] != singles[0]["input_features"].shape[1]:
            raise ValueError("Qwen成员特征通道数不同")
    lengths = [int(item["attention_mask"].bool().sum()) for item in singles]
    text_length = max(lengths)
    feature_length = max(item["input_features"].shape[-1] for item in singles)
    count = len(singles)
    reference = singles[0]
    output = {
        "input_ids": reference["input_ids"].new_full((count, text_length), pad_token_id),
        "attention_mask": reference["attention_mask"].new_zeros((count, text_length)),
        "input_features": reference["input_features"].new_zeros(
            (count, reference["input_features"].shape[1], feature_length)),
        "input_features_mask": reference["input_features_mask"].new_zeros((count, feature_length)),
    }
    for index, (item, length) in enumerate(zip(singles, lengths)):
        valid = item["attention_mask"][0].bool()
        output["input_ids"][index, text_length-length:] = item["input_ids"][0, valid]
        output["attention_mask"][index, text_length-length:] = item["attention_mask"][0, valid]
        original_length = item["input_features"].shape[-1]
        # 完整复制single原有末尾上下文及100帧补齐，只有额外batch补齐用零。
        output["input_features"][index, :, :original_length] = item["input_features"][0]
        output["input_features_mask"][index, :original_length] = item["input_features_mask"][0]
    return output


def prepare_independent_inputs(processor, audios, languages, *, metadata=None):
    """逐成员FP32前端，随后collation；批内一段的边界不取决于其它PCM长度。"""
    import torch
    from core.model_loading import model_loading_guard

    if not audios or len(audios) != len(languages):
        raise ValueError("Qwen前端音频/语种成员数必须一致且非空")
    with model_loading_guard(dtype=torch.float32):
        singles = [processor.apply_transcription_request(audio=audio, language=language)
                   for audio, language in zip(audios, languages)]
        tokenizer = getattr(processor, "tokenizer", None)
        inputs = collate_independent_inputs(singles, pad_token_id=getattr(tokenizer, "pad_token_id", None))
    if metadata is not None:
        metadata["frontend_mode"] = "independent_single_fp32_then_collate"
        metadata["single_processor_shapes"] = [
            {key: list(value.shape) for key, value in item.items() if hasattr(value, "shape")}
            for item in singles]
    return inputs


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
        value = self.transcribe_batch([audio_bytes], [language])[0]
        if isinstance(value, Exception):
            raise value
        return value

    def transcribe_batch(self, audio_bytes_list, languages=None):
        """独立音频列表组成物理批；不拼PCM，逐成员保留提示与输出。"""
        self.last_batch_timings = None
        started = time.perf_counter()
        audios = [decode_wav_bytes(data) for data in audio_bytes_list]
        languages = list(languages) if languages is not None else [None] * len(audios)
        if not audios or len(audios) != len(languages):
            raise ValueError("Qwen batch 音频和语言列表长度必须相同且非空")
        decoded = time.perf_counter()
        self.ensure_loaded()
        loaded = time.perf_counter()
        import torch

        self.api.set_device(self.device)
        timings = {"batch_size": len(audios), "decode_audio_ms": (decoded-started)*1000,
                   "load_ms": (loaded-decoded)*1000,
                   "generate_started": False, "generate_completed": False}
        with self.api.device(self.device), torch.inference_mode():
            try:
                # 使用锁定processor自身规则逐成员校验，非法提示不能污染其它请求。
                prepare_method = self.processor.apply_transcription_request
                prepare_function = getattr(prepare_method, '__func__', prepare_method)
                validator = getattr(prepare_function, '__globals__', {}).get('_prepare_language_inputs')
                if validator is None and getattr(prepare_function, '__module__', '').startswith('transformers.'):
                    raise RuntimeError('锁定processor缺少实际语言校验器，不能安全执行混合批')
                normalized = []
                valid_indices = []
                result_slots = [None] * len(audios)
                for index, language in enumerate(languages):
                    try:
                        if validator is not None:
                            validator(language, 1)
                        normalized.append(language)
                        valid_indices.append(index)
                    except ValueError as exc:
                        result_slots[index] = LocalUnsupportedLanguage(str(exc))
                timings['requested_batch_size'] = len(audios)
                timings['batch_size'] = len(valid_indices)
                timings['valid_member_indices'] = valid_indices
                if not valid_indices:
                    return result_slots
                valid_audios = [audios[index] for index in valid_indices]
                try:
                    inputs = prepare_independent_inputs(
                        self.processor, valid_audios, normalized, metadata=timings)
                except ValueError as exc:
                    if "language" in str(exc).lower():
                        raise LocalUnsupportedLanguage(str(exc)) from exc
                    raise
                prepared = time.perf_counter()
                if inputs["input_ids"].shape[0] != len(valid_audios):
                    raise RuntimeError("Qwen processor batch 维度与成员数不一致")
                timings["processor_shapes"] = {key: list(value.shape) for key, value in inputs.items()
                                                if hasattr(value, "shape")}
                timings["processor_ms"] = (prepared-loaded)*1000
                inputs = move_inputs(inputs, device=self.device, dtype=self.model.dtype)
                transferred = time.perf_counter()
                timings["transfer_wall_ms"] = (transferred-prepared)*1000
                timings["generate_started"] = True
                output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
                timings["generate_completed"] = True
                generated_at = time.perf_counter()
                timings["generate_wall_ms"] = (generated_at-transferred)*1000
                sequences = getattr(output, "sequences", output)
                generated = sequences[:, inputs["input_ids"].shape[1]:]
                texts = self.processor.batch_decode(
                    generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                if len(texts) != len(valid_audios):
                    raise RuntimeError("Qwen 输出成员数与输入不一致")
                results = []
                output_error_indices = []
                for member_index, text in zip(valid_indices, texts):
                    text = re.sub(r"<\|(?:im_end|endoftext|eot_id)\|>", "", text).strip()
                    if text and not re.search(r"language\s+[^<]*<asr_text>", text, re.I):
                        # 模型已实际处理此成员；仅隔离该输出，不能删掉物理dispatch记录。
                        error = RuntimeError("Qwen 原始输出缺少语种前缀，不能当作无人声")
                        results.append(error)
                        output_error_indices.append(member_index)
                    else:
                        results.append(text)
                timings["output_error_indices"] = output_error_indices
                timings["output_error_count"] = len(output_error_indices)
                timings["read_decode_wall_ms"] = (time.perf_counter()-generated_at)*1000
                timings["generated_padded_tokens"] = int(generated.shape[1])
                for index, text in zip(valid_indices, results):
                    result_slots[index] = text
                return result_slots
            finally:
                sync_started = time.perf_counter()
                try:
                    # 设备级同步仍保留；wall包络不冒充纯核时间。
                    self.api.synchronize(self.device)
                except Exception as exc:
                    self.failure = f"NPU synchronize failed: {type(exc).__name__}: {exc}"
                    raise
                finally:
                    timings["synchronize_wall_ms"] = (time.perf_counter()-sync_started)*1000
                    timings["total_wall_ms"] = (time.perf_counter()-started)*1000
                    self.last_batch_timings = timings

    def snapshot(self):
        return {
            "device": self.device, "dtype": self.dtype, "loaded": self.loaded,
            "loading": self.loading, "failure": self.failure,
            "last_batch_timings": getattr(self, "last_batch_timings", None),
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
