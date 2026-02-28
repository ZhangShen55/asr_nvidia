import asyncio
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
from faster_whisper import WhisperModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from funasr import AutoModel
from pyannote.audio import Pipeline as PyannotePipeline

from core.config import settings
from utils.feature_utils import id2label

# 单例缓存
_model_asr = None
_model_emotion = None
_model_online = None
_model_whisper = None
_model_speaker = None
_punct_pipeline = None

# 五何
_model_bert = None
_tokenizer = None

# 线程锁
_model_lock = asyncio.Lock()


def device() -> torch.device:
    return torch.device(settings.device if torch.cuda.is_available() else "cpu")


async def load_models_if_needed():
    """
    根据配置开关懒加载模型。
    """
    global _model_asr, _model_emotion, _model_online, _model_whisper, _model_speaker, _punct_pipeline

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

        if settings.open_online and _model_online is None:
            _model_online = AutoModel(
                model=settings.asr_online_model_dir,
                device=settings.device,
                ngpu=settings.ngpu,
                disable_update=True,
                disable_pbar=True
            )

        if settings.open_mul_lang and _model_whisper is None:
            _model_whisper = WhisperModel(
                settings.whisper_model_dir,
                compute_type=settings.compute_type,
                device="cuda" if torch.cuda.is_available() else "cpu",
                device_index=int(settings.device.split(":")[-1]) if ":" in settings.device else 0
            )

        if settings.open_mul_spk and _model_speaker is None:
            _model_speaker = PyannotePipeline.from_pretrained(settings.pyannote_model_yml)
            _model_speaker.to(device())

        if settings.open_online and _punct_pipeline is None:
            _punct_pipeline = pipeline(
                task=Tasks.punctuation,
                model=settings.asr_online_punc_model_dir,
                disable_update=True,
                device=settings.device
            )


def get_asr_model():
    return _model_asr


def get_emotion_model():
    return _model_emotion


def get_online_model():
    return _model_online


def get_whisper_model():
    return _model_whisper


def get_speaker_model():
    return _model_speaker


def get_punct_pipeline():
    return _punct_pipeline


# ---------- 五何分类 ----------
def _ensure_bert_loaded():
    global _model_bert, _tokenizer
    if _model_bert is None or _tokenizer is None:
        _model_bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_dir
        ).to(device()).eval()
        _tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name_or_path=settings.bert_model_tokenizer
        )


def predict_fivewh(text: str) -> tuple[str, int, float]:
    """
    教师提问5何（是何、为何、若何、由何、如何、非提问） bert预测（中文）
    """
    _ensure_bert_loaded()
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device())
    with torch.no_grad():
        logits = _model_bert(**inputs).logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)

    return id2label[predicted.item()], predicted.item(), confidence.item()
