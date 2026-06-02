import os
import re
import time
import uuid
import shutil
import tempfile
import logging
import asyncio
import torchaudio
import numpy as np
from copy import deepcopy
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException
from fastapi import UploadFile, Form

from entity.data import AsrRequestParams, get_asr_params
from core.config import settings
from core.models import (
    get_asr_model, get_emotion_model, get_online_model, get_whisper_model, get_speaker_model
)
from core.concurrency import acquire_gpu_slot, generate_with_gpu_lock, transcribe_with_gpu_lock
from utils.audio_utils import preprocess_audio, write_audio_bytes_to_temp_file, crop_audio
from utils.pynanote_speaker import diarize_text
from utils.feature_utils import extract_features, identify_teacher, convert_role_ids, calculate_speech_rate, build_speed_info
from utils.asr_stats import update_stat, update_fail_task, add_queued_task, remove_queued_task
from utils.character_utils import safe_concat
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

router = APIRouter()

# emotion2vec 原始标签 -> 业务情绪标签 映射
EMOTION_LABEL_MAP = {
    "中立/neutral": "平淡",
    "其他/other": "平淡",
    "<unk>": "平淡",
    "开心/happy": "积极",
    "吃惊/surprised": "兴奋",
    "生气/angry": "强调",
    "难过/sad": "思考",
    "厌恶/disgusted": "思考",
    "恐惧/fearful": "疑问",
}


def _map_emotion(label: str) -> str:
    """将 emotion2vec 原始标签映射为业务情绪标签；未知或空值归为平淡。"""
    if not label:
        return "平淡"
    return EMOTION_LABEL_MAP.get(label, "平淡")


def _generate_task_id() -> str:
    return str(uuid.uuid4())


def _cleanup_temp_files(tmp_paths: List[str]):
    """统一清理临时文件"""
    for p in tmp_paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"删除临时文件失败: {p}, 错误: {e}")


@router.post("/v1.1.8/seacraft_asr")
async def api_asr_mul(request: AsrRequestParams = Depends(get_asr_params)):
    """
    语音转写 + 小语种(whisper) + 身份判定 + (可选)情绪识别 + (可选)音轨分离
    保持原接口契约不变
    """
    if request.audioFile is None:
        logger.error("音频为空")
        return {"msg": "音频文件不能为空", "code": 4001}

    logger.info(f"request: \n{request}")
    # funasr 识别参数
    param_dict = {
        "batch_size_s": 300,
        "language": "auto",
        "spk_model": "open"
    }
    local_param_dict = deepcopy(param_dict)

    # 处理 hotWords
    if not request.hotWords:
        hotword_str = ""
    else:
        if len(request.hotWords) == 1 and isinstance(request.hotWords[0], str) and "," in request.hotWords[0]:
            request.hotWords = request.hotWords[0].split(",")
        hotword_str = " ".join(request.hotWords)
    local_param_dict["hotword"] = "" if settings.ban_hotword else hotword_str

    # 读取音频
    try:
        start_time = time.perf_counter()
        content = await request.audioFile.read()
        suffix = request.audioFile.filename.split(".")[-1].lower()
        filename = request.audioFile.filename
        if suffix == "m4a":
            suffix = "mp4"
        audio_bytes = await preprocess_audio(content, suffix, force_resample=False)
        if len(audio_bytes) < 1024:
            logger.error(f"音频文件过小，疑似空白音频：{request.audioFile.filename}, 大小：{len(audio_bytes)}")
            update_fail_task()
            return {"msg": "音频文件内容为空，请重新录制", "code": 4006}
        load_audio_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"加载音频文件耗时：{load_audio_time_ms:.2f} ms")
    except Exception as e:
        logger.error(f"读取音频文件发生错误，文件名:{request.audioFile.filename}, 错误信息：\n{e}")
        return {"msg": "读取音频文件发生错误", "code": 4002}

    tmp_paths: List[str] = []
    common_langs = ["fr"]  # 上交大 fr
    audioFile_bytes = content

    try:
        # 保存为临时文件（供 whisper / 说话人分离使用）
        tmp_path = write_audio_bytes_to_temp_file(audioFile_bytes, file_name=filename, suffix=f"{suffix}")
        tmp_paths.append(tmp_path)

        # 读取音频 tensor
        audio_tensor, sample_rate = await asyncio.to_thread(torchaudio.load, tmp_path, backend="ffmpeg")

        # 音频总时长（秒），用于 speed_info 按整段音频切分时间窗口
        audio_total_s = audio_tensor.shape[-1] / sample_rate if sample_rate else 0.0

        task_id = f"{_generate_task_id()}_{filename}"

        # ---------- 小语种：优先 whisper ----------
        if request.language != "auto" and request.language in common_langs:
            if not settings.open_mul_lang or get_whisper_model() is None:
                logger.error("open_mul_lang 未开启或 whisper 模型未加载")
                update_fail_task()
                return {"msg": "未开启小语种识别或模型未就绪", "code": 4003}

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

            # 有/无音轨分离
            if request.showSpk:
                if not settings.open_mul_spk or get_speaker_model() is None:
                    logger.error(f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型 文件名:{filename} 转写失败")
                    update_fail_task()
                    return {"msg": f"open_mul_spk={settings.open_mul_spk} 未开启小语种音轨分离模型", "code": 4003}

                speaker_result = get_speaker_model()(tmp_path)
                finally_result = diarize_text(segments, speaker_result)

                # 统计 spk 次数 -> teacher
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

                # 情绪识别（可选）
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
                        seg["emotion"] = _map_emotion(emotion_label)

                update_stat("offline")
                return {
                    "language": request.language,
                    "segments": output_segments,
                    "text": text.strip(),
                    "speed_info": build_speed_info(output_segments, total_duration=audio_total_s),
                    "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                    "gpu_time_ms": f"{gpu_time_ms:.2f}"
                }
            else:
                # 不分离音轨：聚合纯 whisper 输出
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

        # ---------- Paraformer 路线 ----------
        start_time = time.perf_counter()
        model_asr = get_asr_model()

        if request.showSpk:
            if not settings.open_spk or model_asr is None:
                logger.error(f"open_spk={settings.open_spk} 未开启音轨分离模型 文件名:{filename} 转写失败")
                update_fail_task()
                return {"msg": f"open_spk={settings.open_spk} 未开启音轨分离模型", "code": 4003}

            try:
                async with acquire_gpu_slot(task_id=task_id):
                    rec_results = await generate_with_gpu_lock(
                        model_asr, input=audio_bytes, is_final=True, **local_param_dict
                    )
            except (asyncio.TimeoutError, IndexError, HTTPException):
                update_fail_task()
                return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

            gpu_time_ms = (time.perf_counter() - start_time) * 1000

            if len(rec_results) == 0:
                update_fail_task()
                return {"text": "", "sentences": [], "code": 0}
            elif len(rec_results) == 1:
                rec_result = rec_results[0]
                text = rec_result["text"]
                if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                    logger.error(f"音频文件为空或未检测到任何人声，可能是静音，文件名:{filename} 转写失败")
                    return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

                segments = []
                for segment in rec_result["sentence_info"]:
                    segment_words = []
                    emotion = None
                    if request.wordTimestamps:
                        words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                        timestamps = segment["timestamp"]
                        for i, word_text in enumerate(words):
                            if i < len(timestamps):
                                segment_words.append({
                                    "bg": f"{timestamps[i][0] / 1000:.2f}",
                                    "ed": f"{timestamps[i][1] / 1000:.2f}",
                                    "word_text": word_text
                                })

                    if request.showEmotion:
                        if not settings.open_emotion or get_emotion_model() is None:
                            logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                            update_fail_task()
                            return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

                        seg_len_ms = segment["end"] - segment["start"]
                        if seg_len_ms > 10000:
                            emotion = "平淡"
                        else:
                            # 音频片段识别
                            cropped = crop_audio(audio_tensor, segment["start"] + 0.1, segment["end"] + 0.1, sample_rate)
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                                seg_path = segf.name
                                tmp_paths.append(seg_path)
                            torchaudio.save(seg_path, cropped, sample_rate)
                            res_emotion = get_emotion_model().generate(seg_path, granularity="utterance", extract_embedding=False)
                            max_score = max(res_emotion[0]['scores'])
                            emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                            emotion = _map_emotion(emotion_label)

                    speed = calculate_speech_rate(segment["text"], segment["start"] / 1000, segment["end"] / 1000, settings.speech_rate_factor)
                    item = {
                        "segment_text": segment["text"],
                        "bg": f"{segment['start'] / 1000:.2f}",
                        "ed": f"{segment['end'] / 1000:.2f}",
                        "speed": speed,
                        "segment_words": segment_words,
                        "role": segment.get("spk")
                    }
                    if request.showEmotion:
                        item["emotion"] = emotion if emotion is not None else "平淡"
                    segments.append(item)

                # 身份识别（老师/学生）：由请求参数 showRoleIdentify 控制，默认 true
                if request.showRoleIdentify:
                    spk_features = extract_features(segments)
                    teacher_role, scores, student_roles = identify_teacher(spk_features)
                    if teacher_role is None:
                        teacher_role = max(spk_features, key=lambda x: spk_features[x]["keyword_count"])
                    segments = convert_role_ids(segments, teacher_role, student_roles)
                else:
                    # 关闭时保留 Paraformer 原始 spk ID，格式化为 spk_X 字符串
                    for seg in segments:
                        raw = seg.get("role")
                        if raw is not None:
                            seg["role"] = f"spk_{raw}"

                ret = {
                    "language": request.language,
                    "segments": segments,
                    "text": text,
                    "speed_info": build_speed_info(segments, total_duration=audio_total_s),
                    "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                    "gpu_time_ms": f"{gpu_time_ms:.2f}",
                }
                update_stat("offline")
                return ret

        # ---------- 不开音轨分离 ----------
        try:
            local_param_dict["spk_model"] = None
            async with acquire_gpu_slot(task_id=task_id):
                rec_results = await generate_with_gpu_lock(
                    model_asr, input=audio_bytes, is_final=True, **local_param_dict
                )
        except (asyncio.TimeoutError, IndexError, HTTPException):
            update_fail_task()
            return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

        gpu_time_ms = (time.perf_counter() - start_time) * 1000
        if len(rec_results) == 0:
            update_fail_task()
            return {"text": "", "sentences": [], "code": 0}
        elif len(rec_results) == 1:
            rec_result = rec_results[0]
            text = rec_result["text"]
            if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                logger.error(f"音频文件为空或未检测到任何人声, 可能是静音文件, 文件名:{filename} 转写失败")
                return {"msg": "音频文件为空或未检测到任何人声,可能是静音文件", "code": 4008}

            segments = []
            for segment in rec_result["sentence_info"]:
                segment_words = []
                if request.wordTimestamps:
                    words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                    timestamps = segment["timestamp"]
                    for i, word_text in enumerate(words):
                        if i < len(timestamps):
                            segment_words.append({
                                "bg": f"{timestamps[i][0] / 1000:.2f}",
                                "ed": f"{timestamps[i][1] / 1000:.2f}",
                                "word_text": word_text
                            })
                emotion = None
                if request.showEmotion:
                    if not settings.open_emotion or get_emotion_model() is None:
                        logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename} 转写失败")
                        update_fail_task()
                        return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

                    seg_len_ms = segment["end"] - segment["start"]
                    if seg_len_ms > 10000 or seg_len_ms < 1000:
                        emotion = "平淡"
                    else:
                        cropped = crop_audio(audio_tensor, segment["start"] + 0.1, segment["end"] + 0.1, sample_rate)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                            seg_path = segf.name
                            tmp_paths.append(seg_path)
                        torchaudio.save(seg_path, cropped, sample_rate)
                        res_emotion = get_emotion_model().generate(seg_path, granularity="utterance", extract_embedding=False)
                        max_score = max(res_emotion[0]['scores'])
                        emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                        emotion = _map_emotion(emotion_label)

                speed = calculate_speech_rate(segment["text"], segment["start"] / 1000, segment["end"] / 1000, settings.speech_rate_factor)
                item = {
                    "segment_text": segment["text"],
                    "bg": f"{segment['start'] / 1000:.2f}",
                    "ed": f"{segment['end'] / 1000:.2f}",
                    "speed": speed,
                    "segment_words": segment_words
                }
                if request.showEmotion:
                    item["emotion"] = emotion if emotion is not None else "平淡"
                segments.append(item)
            ret = {
                "language": request.language,
                "segments": segments,
                "text": text,
                "speed_info": build_speed_info(segments, total_duration=audio_total_s),
                "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
                "gpu_time_ms": f"{gpu_time_ms:.2f}"
            }
            update_stat("offline")
            return ret
        else:
            update_fail_task()
            return {"msg": "未知错误", "code": 4005}

    finally:
        # 统一清理所有临时文件
        _cleanup_temp_files(tmp_paths)
