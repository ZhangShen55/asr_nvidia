import time
import tempfile
import logging
import torchaudio
from typing import Optional

from fastapi import APIRouter, Depends
from faster_whisper import WhisperModel

from entity.data import AsrRequestParams, get_asr_params
from core.config import settings
from core.models import get_emotion_model, get_whisper_model, get_speaker_model
from core.concurrency import acquire_gpu_slot, transcribe_with_gpu_lock
from utils.audio_utils import crop_audio
from utils.pynanote_speaker import diarize_text
from utils.feature_utils import calculate_speech_rate, build_speed_info
from utils.asr_stats import update_stat, update_fail_task

from api.routes.asr_common import (
    AsrContext,
    map_emotion,
    cleanup_temp_files,
    prepare_asr_context,
    run_paraformer_asr,
)

logger = logging.getLogger(__name__)

router = APIRouter()

COMMON_LANGS = ["fr"]


async def _try_whisper_asr(ctx: AsrContext) -> Optional[dict]:
    """小语种 Whisper 转写；不适用时返回 None，由调用方继续走 Paraformer。"""
    request = ctx.request
    if request.language == "auto" or request.language not in COMMON_LANGS:
        return None

    if not settings.open_mul_lang or get_whisper_model() is None:
        logger.error("open_mul_lang 未开启或 whisper 模型未加载")
        update_fail_task()
        return {"msg": "未开启小语种识别或模型未就绪", "code": 4003}

    filename = ctx.filename
    tmp_path = ctx.tmp_path
    tmp_paths = ctx.tmp_paths
    audio_tensor = ctx.audio_tensor
    sample_rate = ctx.sample_rate
    audio_total_s = ctx.audio_total_s
    task_id = ctx.task_id
    load_audio_time_ms = ctx.load_audio_time_ms

    model_whisper: WhisperModel = get_whisper_model()
    start_gpu = time.time()
    text = ""
    output_segments = []

    async with acquire_gpu_slot(task_id=task_id):
        segments, info = await transcribe_with_gpu_lock(
            model_whisper, tmp_path,
            language=request.language,
            beam_size=5,
            word_timestamps=request.wordTimestamps
        )
    gpu_time_ms = (time.time() - start_gpu) * 1000

    if request.showSpk:
        if not settings.open_mul_spk or get_speaker_model() is None:
            logger.error(f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型 文件名:{filename} 转写失败")
            update_fail_task()
            return {"msg": f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型", "code": 4003}

        speaker_result = get_speaker_model()(tmp_path)
        finally_result = diarize_text(segments, speaker_result)

        spk_id_counts = {}
        for segment, spk, sentence, word_text in finally_result:
            spk_id = int(spk.split("_")[-1]) if spk is not None else 0
            spk_id_counts[spk_id] = spk_id_counts.get(spk_id, 0) + 1
        teacher_id = max(spk_id_counts, key=spk_id_counts.get)
        student_ids = {spk_id: i for i, spk_id in enumerate([i for i in spk_id_counts if i != teacher_id], 1)}

        for segment, spk, sentence, word_text in finally_result:
            spk_id = int(spk.split("_")[-1]) if spk is not None else 0
            role = "teacher" if spk_id == teacher_id else f"student{student_ids[spk_id]}"
            text += sentence + " "
            speed = calculate_speech_rate(sentence, round(segment.start, 2), round(segment.end, 2), settings.speech_rate_factor)
            output_segments.append({
                "segment_text": sentence,
                "bg": f"{segment.start:.2f}",
                "ed": f"{segment.end:.2f}",
                "speed": speed,
                "segment_words": [
                    {
                        "bg": f"{float(word['bg']):.2f}",
                        "ed": f"{float(word['ed']):.2f}",
                        "word_text": f"{word['word_text'].strip()}"
                    }
                    for word in (word_text[0] if word_text else []) if word is not None
                ],
                "role": role
            })

        if request.showEmotion:
            if not settings.open_emotion or get_emotion_model() is None:
                logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                update_fail_task()
                return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

            model_emotion = get_emotion_model()
            for seg in output_segments:
                seg_len_ms = float(seg["ed"]) * 1000 - float(seg["bg"]) * 1000
                if seg_len_ms > 10000 or seg_len_ms < 1000:
                    seg["emotion"] = "平淡"
                    continue
                cropped_segment = crop_audio(audio_tensor, float(seg["bg"]) * 1000, float(seg["ed"]) * 1000, sample_rate)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                    seg_path = segf.name
                    tmp_paths.append(seg_path)
                torchaudio.save(seg_path, cropped_segment, sample_rate)

                res_emotion = model_emotion.generate(seg_path, granularity="utterance", extract_embedding=False)
                max_score = max(res_emotion[0]['scores'])
                emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                seg["emotion"] = map_emotion(emotion_label)

        update_stat("offline")
        return {
            "language": request.language,
            "segments": output_segments,
            "text": text.strip(),
            "speed_info": build_speed_info(output_segments, total_duration=audio_total_s),
            "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
            "gpu_time_ms": f"{gpu_time_ms:.2f}"
        }

    text = ""
    output_segments = []
    for seg in segments:
        text += seg.text.strip() + " "
        output_segments.append({
            "segment_text": seg.text.strip(),
            "bg": f"{seg.start:.2f}",
            "ed": f"{seg.end:.2f}",
            "segment_words": [
                {
                    "bg": f"{word.start:.2f}",
                    "ed": f"{word.end:.2f}",
                    "word_text": f"{word.word.strip()}"
                } for word in (seg.words if seg.words is not None else []) if word is not None
            ]
        })

    update_stat("offline")
    return {
        "language": request.language,
        "segments": output_segments,
        "text": text.strip(),
        "speed_info": build_speed_info(output_segments, total_duration=audio_total_s),
        "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
        "gpu_time_ms": f"{gpu_time_ms:.2f}"
    }


@router.post("/v1.1.7/seacraft_asr")
async def api_asr_v17(request: AsrRequestParams = Depends(get_asr_params)):
    """语音转写（含小语种 Whisper）+ 身份判定 + 情绪 + 音轨分离"""
    err, ctx = await prepare_asr_context(request)
    if err:
        return err

    try:
        whisper_result = await _try_whisper_asr(ctx)
        if whisper_result is not None:
            return whisper_result
        return await run_paraformer_asr(ctx)
    finally:
        cleanup_temp_files(ctx.tmp_paths)
