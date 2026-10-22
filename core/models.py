import asyncio
import torch
from funasr import AutoModel

from core.config import settings

# 单例缓存
_model_asr = None
_model_emotion = None
_model_whisper = None

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
