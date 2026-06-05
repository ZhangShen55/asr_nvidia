import os
import re
import time
import uuid
import tempfile
import logging
import asyncio
import torchaudio
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from fastapi import HTTPException

from entity.data import AsrRequestParams
from core.config import settings
from core.models import get_asr_model, get_emotion_model
from core.concurrency import acquire_gpu_slot, generate_with_gpu_lock
from utils.audio_utils import (
    preprocess_audio,
    write_audio_bytes_to_temp_file,
    crop_audio,
    extract_audio_clip,
    plan_audio_chunks,
    load_audio_tensor,
)
from utils.feature_utils import (
    extract_features,
    identify_teacher,
    calculate_speech_rate,
    build_speed_info,
)
from utils.asr_stats import update_stat, update_fail_task

logger = logging.getLogger(__name__)

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


@dataclass
class AsrContext:
    request: AsrRequestParams
    content: bytes
    audio_bytes: bytes
    suffix: str
    filename: str
    tmp_paths: List[str] = field(default_factory=list)
    tmp_path: str = ""
    audio_tensor: Any = None
    sample_rate: int = 0
    audio_total_s: float = 0.0
    task_id: str = ""
    local_param_dict: dict = field(default_factory=dict)
    load_audio_time_ms: float = 0.0


def map_emotion(label: str) -> str:
    """将 emotion2vec 原始标签映射为业务情绪标签；未知或空值归为平淡。"""
    if not label:
        return "平淡"
    return EMOTION_LABEL_MAP.get(label, "平淡")


def generate_task_id() -> str:
    return str(uuid.uuid4())


def cleanup_temp_files(tmp_paths: List[str]):
    """统一清理临时文件"""
    for p in tmp_paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"删除临时文件失败: {p}, 错误: {e}")


async def prepare_asr_context(request: AsrRequestParams) -> Tuple[Optional[dict], Optional[AsrContext]]:
    """读取并预处理音频，构建 ASR 上下文。失败时返回 (error_response, None)。"""
    if request.audioFile is None:
        logger.error("音频为空")
        return {"msg": "音频文件不能为空", "code": 4001}, None

    logger.info(f"request: \n{request}")

    param_dict = {
        "batch_size_s": 300,
        "language": "auto",
        "spk_model": "open",
    }
    local_param_dict = deepcopy(param_dict)

    if not request.hotWords:
        hotword_str = ""
    else:
        if len(request.hotWords) == 1 and isinstance(request.hotWords[0], str) and "," in request.hotWords[0]:
            request.hotWords = request.hotWords[0].split(",")
        hotword_str = " ".join(request.hotWords)
    local_param_dict["hotword"] = "" if settings.ban_hotword else hotword_str

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
            return {"msg": "音频文件内容为空，请重新录制", "code": 4006}, None
        load_audio_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"加载音频文件耗时：{load_audio_time_ms:.2f} ms")
    except Exception as e:
        logger.error(f"读取音频文件发生错误，文件名:{request.audioFile.filename}, 错误信息：\n{e}")
        return {"msg": "读取音频文件发生错误", "code": 4002}, None

    tmp_paths: List[str] = []
    tmp_path = write_audio_bytes_to_temp_file(content, file_name=filename, suffix=f"{suffix}")
    tmp_paths.append(tmp_path)

    audio_tensor, sample_rate = await asyncio.to_thread(load_audio_tensor, tmp_path)
    audio_total_s = audio_tensor.shape[-1] / sample_rate if sample_rate else 0.0
    task_id = f"{generate_task_id()}_{filename}"

    ctx = AsrContext(
        request=request,
        content=content,
        audio_bytes=audio_bytes,
        suffix=suffix,
        filename=filename,
        tmp_paths=tmp_paths,
        tmp_path=tmp_path,
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        audio_total_s=audio_total_s,
        task_id=task_id,
        local_param_dict=local_param_dict,
        load_audio_time_ms=load_audio_time_ms,
    )
    return None, ctx


async def run_paraformer_asr(ctx: AsrContext) -> dict:
    """Paraformer 转写（含音轨分离、长音频分块、情绪识别）。"""
    request = ctx.request
    filename = ctx.filename
    audio_bytes = ctx.audio_bytes
    tmp_path = ctx.tmp_path
    tmp_paths = ctx.tmp_paths
    audio_tensor = ctx.audio_tensor
    sample_rate = ctx.sample_rate
    audio_total_s = ctx.audio_total_s
    task_id = ctx.task_id
    local_param_dict = ctx.local_param_dict
    load_audio_time_ms = ctx.load_audio_time_ms

    start_time = time.perf_counter()
    model_asr = get_asr_model()

    if request.showSpk:
        if not settings.open_spk or model_asr is None:
            logger.error(f"open_spk={settings.open_spk} 未开启音轨分离模型 文件名:{filename} 转写失败")
            update_fail_task()
            return {"msg": f"open_spk={settings.open_spk} 未开启音轨分离模型", "code": 4003}

        need_chunk = (audio_total_s / 60) > settings.chunk_threshold_minutes
        overlap_s = settings.chunk_overlap_seconds

        if need_chunk:
            chunk_plans = plan_audio_chunks(
                audio_total_s,
                chunk_minutes=settings.chunk_minutes,
                min_last_minutes=settings.min_last_chunk_minutes,
                overlap_s=overlap_s,
            )
            logger.info(f"长音频分块处理：时长={audio_total_s:.0f}s，共 {len(chunk_plans)} 块，文件：{filename}")
        else:
            chunk_plans = [(0, audio_total_s)]

        all_segments = []
        total_text = ""
        gpu_time_ms = 0.0
        global_student_count = 0
        global_spk_offset = 0

        if request.showEmotion:
            if not settings.open_emotion or get_emotion_model() is None:
                logger.error(f"open_emotion={settings.open_emotion} 未开启情感分析模型 文件名:{filename}")
                update_fail_task()
                return {"msg": f"open_emotion={settings.open_emotion} 未开启情感分析模型", "code": 4003}

        chunk_meta = []
        for idx, (cs, ce) in enumerate(chunk_plans):
            is_first = idx == 0
            is_last = idx == len(chunk_plans) - 1
            a_start = cs if is_first else max(0.0, cs - overlap_s)
            a_end = ce if is_last else min(audio_total_s, ce + overlap_s)
            chunk_meta.append((a_start, a_end, cs, ce))

        if need_chunk:
            async def _cut_one(idx, a_start, a_end):
                path = await asyncio.to_thread(
                    extract_audio_clip, tmp_path, a_start, a_end - a_start
                )
                return idx, path

            cut_coros = [_cut_one(i, m[0], m[1]) for i, m in enumerate(chunk_meta)]
            cut_results = await asyncio.gather(*cut_coros, return_exceptions=True)

            chunk_paths: dict[int, str] = {}
            for result in cut_results:
                if isinstance(result, Exception):
                    logger.error(f"音频分块切割失败：{result}")
                    update_fail_task()
                    return {"msg": "音频分块切割失败，请重试", "code": 4007}
                idx, path = result
                tmp_paths.append(path)
                chunk_paths[idx] = path
        else:
            chunk_paths = {}

        for chunk_idx, (actual_start, actual_end, clean_start, clean_end) in enumerate(chunk_meta):
            is_first = chunk_idx == 0
            is_last = chunk_idx == len(chunk_plans) - 1

            if need_chunk:
                with open(chunk_paths[chunk_idx], "rb") as f:
                    chunk_bytes = f.read()
                chunk_label = f"{task_id}_chunk{chunk_idx}"
            else:
                chunk_bytes = audio_bytes
                chunk_label = task_id

            max_retry = settings.chunk_retry_count
            rec_results = None
            for attempt in range(max_retry + 1):
                try:
                    t_gpu = time.perf_counter()
                    async with acquire_gpu_slot(task_id=f"{chunk_label}_r{attempt}"):
                        rec_results = await generate_with_gpu_lock(
                            model_asr, input=chunk_bytes, is_final=True, **local_param_dict
                        )
                    gpu_time_ms += (time.perf_counter() - t_gpu) * 1000
                    break
                except (asyncio.TimeoutError, IndexError, HTTPException) as e:
                    if attempt < max_retry:
                        wait_s = 2 ** attempt
                        logger.warning(
                            f"块 {chunk_idx} 推理失败（第{attempt+1}次），"
                            f"{wait_s}s 后重试，错误：{e}"
                        )
                        await asyncio.sleep(wait_s)
                    else:
                        logger.error(f"块 {chunk_idx} 推理失败，已达最大重试次数 {max_retry}")
                        update_fail_task()
                        return {"msg": "请求过多或超时，请稍后再试", "code": 4004}

            if not rec_results:
                continue
            rec_result = rec_results[0]
            if "sentence_info" not in rec_result or not rec_result["sentence_info"]:
                logger.warning(f"块 {chunk_idx} 未检测到人声，跳过")
                continue

            total_text += rec_result.get("text", "")

            chunk_raw_spks = sorted(set(
                seg.get("spk") for seg in rec_result["sentence_info"]
                if seg.get("spk") is not None
            ))

            chunk_segments = []
            for segment in rec_result["sentence_info"]:
                abs_start_s = segment["start"] / 1000 + actual_start
                abs_end_s = segment["end"] / 1000 + actual_start

                if not is_first and abs_start_s < clean_start:
                    continue
                if not is_last and abs_start_s >= clean_end:
                    continue

                segment_words = []
                if request.wordTimestamps:
                    words = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z']+|\b[a-zA-Z']+\b", segment["text"])
                    for i, word_text in enumerate(words):
                        if i < len(segment.get("timestamp", [])):
                            ts = segment["timestamp"][i]
                            segment_words.append({
                                "bg": f"{ts[0] / 1000 + actual_start:.2f}",
                                "ed": f"{ts[1] / 1000 + actual_start:.2f}",
                                "word_text": word_text
                            })

                emotion = None
                if request.showEmotion:
                    seg_len_ms = segment["end"] - segment["start"]
                    if seg_len_ms > 10000 or seg_len_ms < 1000:
                        emotion = "平淡"
                    else:
                        cropped = crop_audio(audio_tensor, abs_start_s * 1000 + 0.1, abs_end_s * 1000 + 0.1, sample_rate)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as segf:
                            seg_path = segf.name
                            tmp_paths.append(seg_path)
                        torchaudio.save(seg_path, cropped, sample_rate)
                        res_emotion = get_emotion_model().generate(seg_path, granularity="utterance", extract_embedding=False)
                        max_score = max(res_emotion[0]['scores'])
                        emotion_label = res_emotion[0]['labels'][res_emotion[0]['scores'].index(max_score)]
                        emotion = map_emotion(emotion_label)

                speed = calculate_speech_rate(segment["text"], abs_start_s, abs_end_s, settings.speech_rate_factor)
                item = {
                    "segment_text": segment["text"],
                    "bg": f"{abs_start_s:.2f}",
                    "ed": f"{abs_end_s:.2f}",
                    "speed": speed,
                    "segment_words": segment_words,
                    "role": segment.get("spk"),
                }
                if request.showEmotion:
                    item["emotion"] = emotion if emotion is not None else "平淡"
                chunk_segments.append(item)

            if request.showRoleIdentify:
                spk_features = extract_features(chunk_segments)
                teacher_role, _, student_roles = identify_teacher(spk_features)
                if teacher_role is None:
                    teacher_role = max(spk_features, key=lambda x: spk_features[x]["keyword_count"])

                for seg in chunk_segments:
                    raw = seg["role"]
                    if raw == teacher_role:
                        seg["role"] = "teacher"
                    elif raw in student_roles:
                        idx = student_roles.index(raw)
                        seg["role"] = f"student{global_student_count + idx + 1}"
                    else:
                        seg["role"] = f"spk_{raw}"

                global_student_count += len(student_roles)
            else:
                spk_to_global = {
                    spk: global_spk_offset + i
                    for i, spk in enumerate(chunk_raw_spks)
                }
                for seg in chunk_segments:
                    raw = seg["role"]
                    seg["role"] = f"spk_{spk_to_global.get(raw, raw)}"
                global_spk_offset += len(chunk_raw_spks)

            all_segments.extend(chunk_segments)
            logger.info(f"块 {chunk_idx} 完成：{len(chunk_segments)} 段，clean=[{clean_start:.0f}s,{clean_end:.0f}s]")

        if not all_segments:
            update_fail_task()
            return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

        ret = {
            "language": request.language,
            "segments": all_segments,
            "text": total_text,
            "speed_info": build_speed_info(all_segments, total_duration=audio_total_s),
            "load_audio_time_ms": f"{load_audio_time_ms:.2f}",
            "gpu_time_ms": f"{gpu_time_ms:.2f}",
        }
        update_stat("offline")
        return ret

    # 不开音轨分离
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
                    emotion = map_emotion(emotion_label)

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
