"""v2（Qwen3-ASR）转写链路：VAD 切分 → 并发转写 → 响应封装。"""

from __future__ import annotations

import asyncio
import logging
import time
from contextvars import ContextVar
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

# 分阶段耗时统计：按请求隔离（子任务继承请求上下文，统计字典为同一对象），
# 用于定位端到端时延的构成
_PHASE_STATS: ContextVar[Optional[Dict[str, float]]] = ContextVar(
    "qwen_phase_stats", default=None
)


def _phase_stats() -> Dict[str, float]:
    """取当前请求的耗时统计字典；不在请求上下文中时返回一个丢弃用的临时字典。"""
    stats = _PHASE_STATS.get()
    if stats is None:
        stats = {"spk_ms": 0.0, "spk_calls": 0, "emo_ms": 0.0, "emo_calls": 0}
    return stats


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
        started = time.perf_counter()
        try:
            return await asyncio.to_thread(_run)
        finally:
            stats = _phase_stats()
            stats["spk_ms"] += (time.perf_counter() - started) * 1000
            stats["spk_calls"] += 1


async def _segment_emotion(audio_array, sample_rate, duration_s):
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
                    # 直接喂 numpy 数组：funasr 的文件分支会额外做一次
                    # 音频读取与重采样，实测单次 28.9ms → 18.7ms，
                    # 输出标签与分数与文件分支逐位一致（25/25 抽样核对）。
                    input=audio_array,
                    fs=sample_rate,
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
            started = time.perf_counter()
            try:
                return await asyncio.to_thread(_run)
            finally:
                stats = _phase_stats()
                stats["emo_ms"] += (time.perf_counter() - started) * 1000
                stats["emo_calls"] += 1
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
        if o.segment.duration_s < SHORT_SPEAKER_SEGMENT_S
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

    clusters = _average_linkage_clusters(similarity, limit)

    labels: Dict[int, str] = {}
    for index, cluster in enumerate(clusters):
        for member in cluster:
            labels[usable[member].index] = f"spk_{index}"

    _assign_short_speakers(outcomes, usable, labels)
    return labels


def _average_linkage_clusters(similarity: np.ndarray, limit: float) -> List[List[int]]:
    """平均链接层次聚类（增量维护簇间平均相似度）。

    旧实现每轮合并都遍历所有簇对、再对两个簇的成员对逐个求均值，
    在 n 段音频上是 O(n^4)：395 段的长音频要跑约 100 秒，
    是单个长音频端到端时延的主要来源。

    这里改成增量维护：合并 A、B 时，新簇与任意簇 C 的平均相似度按
    ``(|A|·sim(A,C) + |B|·sim(B,C)) / (|A|+|B|)`` 更新（平均链接的
    Lance-Williams 更新式），每轮只需一次 argmax 与一次 O(n) 向量更新，
    复杂度降到 O(n^3)，395 段耗时 < 0.1 秒。聚类结果与旧实现一致
    （合并顺序、并列取第一对、阈值判定都保持不变）。
    """
    n = int(similarity.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    scores = similarity.astype(np.float64, copy=True)
    # 自比较不参与候选；对角线置为 -inf
    np.fill_diagonal(scores, -np.inf)
    sizes = np.ones(n, dtype=np.float64)
    clusters: List[List[int]] = [[index] for index in range(n)]

    while True:
        flat = int(np.argmax(scores))
        i, j = divmod(flat, n)
        best = float(scores[i, j])
        if not np.isfinite(best) or best < limit:
            break

        # 合并 j 到 i
        merged = (sizes[i] * scores[i, :] + sizes[j] * scores[j, :]) / (sizes[i] + sizes[j])
        merged[j] = -np.inf
        merged[i] = -np.inf
        scores[i, :] = merged
        scores[:, i] = merged
        sizes[i] += sizes[j]
        clusters[i] = clusters[i] + clusters[j]
        clusters[j] = []
        # 失效簇整体置为 -inf，后续 argmax 不会再选中
        scores[j, :] = -np.inf
        scores[:, j] = -np.inf

    return [cluster for cluster in clusters if cluster]


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
    stats = {"spk_ms": 0.0, "spk_calls": 0, "emo_ms": 0.0, "emo_calls": 0}
    _PHASE_STATS.set(stats)
    phase = {"vad_ms": 0.0, "cut_ms": 0.0, "segment_ms": 0.0, "cluster_ms": 0.0, "itn_ms": 0.0}

    need_role = bool(request.showSpk or request.showRoleIdentify)
    need_emotion = bool(request.showEmotion)

    await load_v2_models_if_needed(
        need_role=need_role,
        need_emotion=need_emotion,
    )

    vad_started = time.perf_counter()
    segments = await detect_segments(ctx.tmp_path)
    phase["vad_ms"] = (time.perf_counter() - vad_started) * 1000
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

    cut_started = time.perf_counter()
    cut_results = await asyncio.gather(
        *[_cut(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=True,
    )
    phase["cut_ms"] = (time.perf_counter() - cut_started) * 1000
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
        # 短于 1 秒的片段不参与聚类（特征不可靠），其说话人标签由
        # _assign_short_speakers 按时间就近继承，这里直接跳过 CAM++ 调用，
        # 长音频里这类片段约占 9%，省下的都是 GPU 与 GIL 时间。
        if need_role and segment.duration_s >= SHORT_SPEAKER_SEGMENT_S:
            tasks.append(_speaker_embedding(audio_array, sample_rate))
        if need_emotion:
            tasks.append(
                _segment_emotion(audio_array, sample_rate, segment.duration_s)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        text_result = results[0]
        if isinstance(text_result, BaseException):
            raise text_result
        outcome.text, outcome.language = text_result

        cursor = 1
        if need_role and segment.duration_s >= SHORT_SPEAKER_SEGMENT_S:
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

    segment_started = time.perf_counter()
    outcomes = await asyncio.gather(
        *[_process(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=True,
    )
    phase["segment_ms"] = (time.perf_counter() - segment_started) * 1000

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
        cluster_started = time.perf_counter()
        speaker_labels = cluster_speakers(valid)
        for outcome in valid:
            outcome.role = speaker_labels.get(outcome.index)
        if request.showRoleIdentify:
            _apply_role_identify(valid)
        phase["cluster_ms"] = (time.perf_counter() - cluster_started) * 1000

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
    itn_started = time.perf_counter()
    output_segments, normalized_text, _ = normalize_segments(
        output_segments, itn_language, config
    )
    phase["itn_ms"] = (time.perf_counter() - itn_started) * 1000

    gpu_time_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "[V2 耗时] 段数=%d 总计=%.0fms | VAD=%.0f 裁剪=%.0f 分段处理=%.0f "
        "(说话人 %.0fms/%d次, 情绪 %.0fms/%d次) 聚类=%.0f ITN=%.0f",
        len(segments),
        gpu_time_ms,
        phase["vad_ms"],
        phase["cut_ms"],
        phase["segment_ms"],
        stats["spk_ms"],
        int(stats["spk_calls"]),
        stats["emo_ms"],
        int(stats["emo_calls"]),
        phase["cluster_ms"],
        phase["itn_ms"],
    )
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
