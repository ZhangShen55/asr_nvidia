"""v2（Qwen3-ASR）转写链路：VAD 切分 → 并发转写 → 响应封装。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from api.routes.asr_common import AsrContext, map_emotion, prepare_asr_context
from core.config import settings
from core.models import get_emotion_v2_model, get_spk_model, load_v2_models_if_needed
from core.qwen_asr_client import (
    QwenAsrCallError,
    QwenAsrUnavailableError,
    QwenAsrUnsupportedLanguageError,
    get_qwen_client,
)
from core.qwen_asr_queue import get_request_concurrency
from core.vad import VadSegment, detect_segments
from utils.audio_utils import (
    extract_audio_clip,
    extract_audio_clip_array,
    extract_audio_clip_from_tensor,
    read_audio_array,
)
from utils.feature_utils import (
    build_speed_info,
    calculate_speech_rate,
    extract_features,
    identify_teacher,
)
from utils.itn import ItnConfig, normalize_segments
from utils.qwen_response import build_language_ratio, parse_qwen_text
from utils.asr_stats import update_fail_task, update_stat

logger = logging.getLogger(__name__)

# 本机 GPU 推理并发闸门：同一张卡上并发推理过多会更慢且更易 OOM，
# 因此限制并发路数而不是完全串行——完全串行会让 8 路并发时
# role/emotion 成为端到端时延的主瓶颈（实测 P50 增幅超过 100%）。
# 因为 Qwen 走另一个容器，整体仍与转写并行。
_local_gpu_semaphore = asyncio.Semaphore(max(1, settings.qwen_local_gpu_concurrency))

# 情绪判定的有效时长区间（秒）
EMOTION_MIN_DURATION_S = 1.0
EMOTION_MAX_DURATION_S = 10.0

# 说话人嵌入过短时（秒）判定不可靠，归并到时间上最近的说话人
SHORT_SPEAKER_SEGMENT_S = 1.0


@dataclass
class SegmentOutcome:
    """单个 VAD 段的处理结果。"""

    index: int
    segment: VadSegment
    text: str = ""
    language: Optional[str] = None
    embedding: Optional[torch.Tensor] = None
    emotion: Optional[str] = None
    role: Optional[str] = None


def _itn_language(language: Optional[str]) -> str:
    """把模型识别语种映射到 ITN 支持的语言标识。"""
    normalized = (language or "").strip().lower()
    if normalized in {"chinese", "zh", "mandarin"}:
        return "zh"
    if normalized in {"english", "en"}:
        return "en"
    return normalized or "auto"


async def _transcribe_segment(clip_path, language, concurrency, segment_semaphore):
    """请求 Qwen 转写一个片段，并剥离原生前缀。"""
    client = await get_qwen_client()
    await concurrency.acquire(segment_semaphore)
    try:
        raw_text = await client.transcribe(clip_path, language=language)
    finally:
        await concurrency.release(segment_semaphore)

    parsed = parse_qwen_text(raw_text)
    return parsed.text, parsed.language


async def _speaker_embedding(audio_array, sample_rate):
    """提取片段的说话人嵌入（CAM++）。

    直接喂 numpy 数组而不是文件路径：funasr 的文件分支里有一次额外的
    音频读取与重采样，实测单次调用 98.8ms → 49.3ms，且嵌入结果逐位相同。
    """
    model = get_spk_model()
    if model is None:
        return None

    def _run():
        with torch.cuda.device(settings.model_device_speech_campplus):
            result = model.generate(input=audio_array, fs=sample_rate)
        if not result:
            return None
        payload = result[0]
        if not isinstance(payload, dict):
            return None
        embedding = payload.get("spk_embedding")
        if embedding is None:
            return None
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().float().reshape(-1).cpu()
        return torch.tensor(embedding, dtype=torch.float32).reshape(-1)

    async with _local_gpu_semaphore:
        return await asyncio.to_thread(_run)


async def _segment_emotion(clip_path, duration_s):
    """推断片段情绪；超出 1~10 秒的片段记为平淡。"""
    if duration_s < EMOTION_MIN_DURATION_S or duration_s > EMOTION_MAX_DURATION_S:
        return "平淡"

    model = get_emotion_v2_model()
    if model is None:
        return None

    def _run():
        with torch.cuda.device(settings.model_device_emotion2vec):
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                result = model.generate(
                    input=clip_path,
                    granularity="utterance",
                    extract_embedding=False,
                )
        if not result:
            return None
        payload = result[0]
        labels = payload.get("labels") or []
        scores = payload.get("scores") or []
        if not labels or not scores:
            return None
        best = max(range(len(scores)), key=lambda i: scores[i])
        return map_emotion(labels[best])

    async with _local_gpu_semaphore:
        try:
            return await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001 - 情绪失败不影响转写
            logger.warning(f"[情绪] 推理失败，回退为 null: {e}")
            return None


def _normalize_embedding(embedding):
    """归一化说话人嵌入，返回单位向量。"""
    norm = torch.linalg.norm(embedding)
    if float(norm) == 0.0:
        return embedding
    return embedding / norm


def _assign_short_speakers(outcomes, usable, labels):
    """过短的片段说话人特征不稳定，归到时间上最近的已聚类片段。"""
    short_segments = [
        o
        for o in outcomes
        if o.embedding is not None and o.segment.duration_s < SHORT_SPEAKER_SEGMENT_S
    ]
    for outcome in short_segments:
        if not usable:
            continue
        nearest = min(
            usable,
            key=lambda o: abs(o.segment.start_s - outcome.segment.start_s),
        )
        nearest_label = labels.get(nearest.index)
        if nearest_label:
            labels[outcome.index] = nearest_label


def cluster_speakers(outcomes, threshold=None):
    """按余弦相似度把说话人嵌入聚类，返回 段索引 -> spk_N。

    采用平均链接层次聚类：反复合并平均相似度最高的两个簇，直到最高相似度低于阈值。
    阈值依据实测 sweep（教师1 / 课堂实录 / 储能学术英语实践各 10 分钟窗口，
    以 /v1.1.8 的说话人标签为参照计算 BCubed F1）：
    0.30 时单说话人音频聚为 1 簇（F1 = 1.0）、三说话人课堂录音聚为 3 簇（F1 = 0.816）；
    阈值升到 0.6 以上会把同一说话人拆成 6~63 簇，0.7 以上 F1 跌破 0.7。
    """
    limit = settings.qwen_speaker_similarity_threshold if threshold is None else threshold

    usable = [
        o
        for o in outcomes
        if o.embedding is not None
        and o.segment.duration_s >= SHORT_SPEAKER_SEGMENT_S
    ]
    if not usable:
        return {}

    matrix = torch.stack([_normalize_embedding(o.embedding) for o in usable])
    similarity = (matrix @ matrix.T).numpy()

    clusters: List[List[int]] = [[index] for index in range(len(usable))]

    while len(clusters) > 1:
        best_score = -1.0
        best_pair = (-1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                block = similarity[np.ix_(clusters[i], clusters[j])]
                score = float(block.mean())
                if score > best_score:
                    best_score = score
                    best_pair = (i, j)
        if best_pair[0] < 0 or best_score < limit:
            break
        i, j = best_pair
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)

    labels: Dict[int, str] = {}
    for index, cluster in enumerate(clusters):
        for member in cluster:
            labels[usable[member].index] = f"spk_{index}"

    _assign_short_speakers(outcomes, usable, labels)
    return labels


def _apply_role_identify(outcomes) -> None:
    """结合关键词特征把 spk_N 映射为 teacher / studentN。"""
    segments = []
    for outcome in outcomes:
        if outcome.role is None:
            continue
        segments.append(
            {
                "role": outcome.role,
                "segment_text": outcome.text,
                "bg": f"{outcome.segment.start_s:.2f}",
                "ed": f"{outcome.segment.end_s:.2f}",
            }
        )
    if not segments:
        return

    spk_features = extract_features(segments)
    teacher_role, _, student_roles = identify_teacher(spk_features)
    if teacher_role is None:
        teacher_role = max(spk_features, key=lambda x: spk_features[x]["keyword_count"])

    student_index = 0
    for outcome in outcomes:
        raw = outcome.role
        if raw is None:
            continue
        if raw == teacher_role:
            outcome.role = "teacher"
        elif raw in student_roles:
            outcome.role = f"student{student_index + 1}"
            student_index += 1
        else:
            outcome.role = raw


async def run_qwen_asr(ctx: AsrContext) -> dict:
    """v2 转写主链路。"""
    request = ctx.request
    started = time.perf_counter()

    need_role = bool(request.showSpk or request.showRoleIdentify)
    need_emotion = bool(request.showEmotion)

    await load_v2_models_if_needed(
        need_role=need_role,
        need_emotion=need_emotion,
    )

    segments = await detect_segments(ctx.tmp_path)
    if not segments:
        update_fail_task()
        return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

    async def _cut(index: int, segment: VadSegment):
        # 优先用已加载的音频张量切片（比逐段 ffmpeg 快 30 倍），无张量时回退 ffmpeg
        if ctx.audio_tensor is not None and ctx.sample_rate:
            path = await asyncio.to_thread(
                extract_audio_clip_from_tensor,
                ctx.audio_tensor,
                ctx.sample_rate,
                segment.start_s,
                segment.duration_s,
            )
            array, sample_rate = await asyncio.to_thread(
                extract_audio_clip_array,
                ctx.audio_tensor,
                ctx.sample_rate,
                segment.start_s,
                segment.duration_s,
            )
        else:
            path = await asyncio.to_thread(
                extract_audio_clip,
                ctx.tmp_path,
                segment.start_s,
                segment.duration_s,
            )
            array, sample_rate = await asyncio.to_thread(read_audio_array, path)
        ctx.tmp_paths.append(path)
        return index, path, array, sample_rate

    cut_results = await asyncio.gather(
        *[_cut(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=True,
    )
    clip_paths: dict = {}
    clip_arrays: dict = {}
    for result in cut_results:
        if isinstance(result, BaseException):
            logger.error(f"[V2] 音频裁剪失败: {result}")
            update_fail_task()
            return {"msg": "音频分块失败，请重试", "code": 4007}
        index, path, array, sample_rate = result
        clip_paths[index] = path
        clip_arrays[index] = (array, sample_rate)

    concurrency = get_request_concurrency()
    segment_semaphore = concurrency.new_segment_semaphore()

    async def _process(index: int, segment: VadSegment) -> SegmentOutcome:
        outcome = SegmentOutcome(index=index, segment=segment)
        clip_path = clip_paths[index]
        audio_array, sample_rate = clip_arrays[index]

        tasks = [
            _transcribe_segment(
                clip_path,
                request.language,
                concurrency,
                segment_semaphore,
            ),
        ]
        if need_role:
            tasks.append(_speaker_embedding(audio_array, sample_rate))
        if need_emotion:
            tasks.append(_segment_emotion(clip_path, segment.duration_s))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        text_result = results[0]
        if isinstance(text_result, BaseException):
            raise text_result
        outcome.text, outcome.language = text_result

        cursor = 1
        if need_role:
            embedding = results[cursor]
            if isinstance(embedding, BaseException):
                logger.warning(f"[说话人] 嵌入提取失败: {embedding}")
                embedding = None
            outcome.embedding = embedding
            cursor += 1
        if need_emotion:
            emotion = results[cursor]
            if isinstance(emotion, BaseException):
                logger.warning(f"[情绪] 推理失败: {emotion}")
                emotion = None
            outcome.emotion = emotion

        return outcome

    outcomes = await asyncio.gather(
        *[_process(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=True,
    )

    for result in outcomes:
        if isinstance(result, BaseException):
            if isinstance(
                result,
                (QwenAsrUnsupportedLanguageError, QwenAsrUnavailableError, QwenAsrCallError),
            ):
                update_fail_task()
                return {"msg": result.message, "code": result.code}
            logger.error(f"[V2] 转写失败: {result}")
            update_fail_task()
            return {"msg": "Qwen 服务调用失败，请稍后再试", "code": 4012}

    valid = [o for o in outcomes if isinstance(o, SegmentOutcome) and o.text.strip()]
    if not valid:
        update_fail_task()
        return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

    # 说话人聚类（整段音频统一聚类）
    if need_role:
        speaker_labels = cluster_speakers(valid)
        for outcome in valid:
            outcome.role = speaker_labels.get(outcome.index)
        if request.showRoleIdentify:
            _apply_role_identify(valid)

    detected_language = None
    for outcome in valid:
        if outcome.language:
            detected_language = outcome.language
            break

    output_segments = []
    for outcome in valid:
        start_s = outcome.segment.start_s
        end_s = outcome.segment.end_s
        output_segments.append(
            {
                "segment_text": outcome.text,
                "bg": f"{start_s:.2f}",
                "ed": f"{end_s:.2f}",
                "speed": calculate_speech_rate(
                    outcome.text,
                    start_s,
                    end_s,
                    settings.speech_rate_factor,
                ),
                "segment_words": [],
                "role": outcome.role,
                "emotion": outcome.emotion,
            }
        )

    language_ratio = build_language_ratio(
        (outcome.language, outcome.text) for outcome in valid
    )
    speed_info = build_speed_info(output_segments, total_duration=ctx.audio_total_s)

    itn_language = _itn_language(detected_language)
    config = getattr(settings, "itn_config", ItnConfig())
    output_segments, normalized_text, _ = normalize_segments(
        output_segments, itn_language, config
    )

    gpu_time_ms = (time.perf_counter() - started) * 1000
    update_stat("offline")
    return {
        "language": detected_language or request.language or "auto",
        "segments": output_segments,
        "text": normalized_text,
        "speed_info": speed_info,
        "load_audio_time_ms": f"{ctx.load_audio_time_ms:.2f}",
        "gpu_time_ms": f"{gpu_time_ms:.2f}",
        "language_ratio": language_ratio,
    }


__all__ = [
    "SegmentOutcome",
    "cluster_speakers",
    "run_qwen_asr",
]
