import asyncio
import torch
from funasr import AutoModel

from core.config import settings

# 单例缓存
_model_asr = None
_model_emotion = None
_model_whisper = None
_model_spk_v2 = None
_model_emotion_v2 = None

# 线程锁
_model_lock = asyncio.Lock()


def resolve_whisper_device() -> tuple[str, int]:
    """根据配置的 device 解析 Whisper 使用的设备类型与 GPU 索引。"""
    configured = settings.device.strip().lower()
    if configured == "cpu":
        return "cpu", 0
    if not configured.startswith("cuda:") or not configured[5:].isdigit():
        raise RuntimeError("device 必须是 cpu 或 cuda:<index>")

    index = int(configured[5:])
    if not torch.cuda.is_available():
        raise RuntimeError(f"配置要求使用 GPU {settings.device}，但 CUDA 不可用")
    visible = torch.cuda.device_count()
    if index >= visible:
        raise RuntimeError(
            f"设备 {settings.device} 索引越界，可见 CUDA 设备数量为 {visible}"
        )
    return "cuda", index


async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion, _model_whisper

    async with _model_lock:
        if settings.open_spk and _model_asr is None:
            _model_asr = AutoModel(
                model=settings.asr_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                punc_model=settings.punc_model_dir,
                vad_model=settings.vad_model_dir,
                spk_model=settings.spk_model_dir,
                vad_kwargs={"max_single_segment_time": 30000, "max_end_silence_time": 800},
                sentence_timestamp=True,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_emotion and settings.open_spk and _model_emotion is None:
            _model_emotion = AutoModel(
                model=settings.emotion_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_mul_lang and _model_whisper is None:
            # 按需导入：未开启小语种能力时，镜像无需包含 Whisper 运行时。
            from faster_whisper import WhisperModel

            device_type, device_index = resolve_whisper_device()
            _model_whisper = WhisperModel(
                settings.whisper_model_dir,
                compute_type="int8",
                device=device_type,
                device_index=device_index,
            )


def get_asr_model():
    return _model_asr


def get_emotion_model():
    return _model_emotion


def get_whisper_model():
    return _model_whisper


def _resolve_device(index: int) -> str:
    """把 GPU 索引解析为 torch 设备串。"""
    if not torch.cuda.is_available():
        raise RuntimeError("v2 链路要求使用 GPU，但 CUDA 不可用")
    visible = torch.cuda.device_count()
    if index < 0 or index >= visible:
        raise RuntimeError(f"GPU 索引 {index} 越界，可见 CUDA 设备数量为 {visible}")
    return f"cuda:{index}"


async def load_v2_models_if_needed(need_role: bool = False, need_emotion: bool = False):
    """懒加载 v2 链路使用的独立模型实例（CAM++ / emotion2vec）。

    - CAM++ 与 emotion2vec 仅在配置 open_spk 与 open_emotion 同时为 true 时加载，
      且各自落在 [model_device] 指定的 GPU 上；
    - VAD 由 core/vad.py 的实例池单独管理。
    """
    global _model_spk_v2, _model_emotion_v2

    async with _model_lock:
        role_ready = settings.open_spk and settings.open_emotion
        if role_ready and need_role and _model_spk_v2 is None:
            device = _resolve_device(settings.model_device_speech_campplus)
            _model_spk_v2 = AutoModel(
                model=settings.spk_model_dir,
                device=device,
                disable_update=True,
                disable_pbar=True,
            )

        if role_ready and need_emotion and _model_emotion_v2 is None:
            device = _resolve_device(settings.model_device_emotion2vec)
            # 精度对半即可，业务上有结果就行；权重占用从约 1.9GB 降到约 1GB。
            _model_emotion_v2 = AutoModel(
                model=settings.emotion_model_dir,
                device=device,
                fp16=True,
                disable_update=True,
                disable_pbar=True,
            )


def get_spk_model():
    return _model_spk_v2


def get_emotion_v2_model():
    return _model_emotion_v2
