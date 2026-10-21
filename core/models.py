import asyncio
from funasr import AutoModel

from core.config import settings

# 单例缓存
_model_asr = None
_model_emotion = None

# 线程锁
_model_lock = asyncio.Lock()


async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion

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

def get_asr_model():
    return _model_asr


def get_emotion_model():
    return _model_emotion
