"""v2（Qwen3-ASR）转写链路：VAD 切分 → 并发转写 → 响应封装。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from api.routes.asr_common import AsrContext, map_emotion, prepare_asr_context
from core.config import settings
from core.models import get_model_pool, load_v2_models_if_needed
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
from utils.itn import ItnConfig, normalize_segments, sanitize_segments
from utils.qwen_response import (
    abbreviate_language,
    build_language_ratio,
    count_text_chars,
    parse_qwen_text,
)
from utils.text_postprocess import (
    is_single_interjection,
    replace_english_interjections,
)
from utils.repetition import (
    RepetitionThresholds,
    max_reasonable_chars,
    repair_repetition,
)
from utils.asr_stats import update_fail_task, update_stat

logger = logging.getLogger(__name__)

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


class SegmentCutError(RuntimeError):
    """单个 VAD 段的音频裁剪失败。"""


def _phase_stats() -> Dict[str, float]:
    """取当前请求的耗时统计字典；不在请求上下文中时返回一个丢弃用的临时字典。"""
    stats = _PHASE_STATS.get()
    if stats is None:
        stats = {
            "spk_wait_ms": 0.0, "spk_ms": 0.0, "spk_calls": 0,
            "emo_wait_ms": 0.0, "emo_ms": 0.0, "emo_calls": 0,
            "qwen_wait_ms": 0.0, "qwen_ms": 0.0, "qwen_calls": 0,
        }
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


def _dominant_language(outcomes: list[SegmentOutcome]) -> Optional[str]:
    """Choose the highest-character-count Qwen language, with stable first-seen ties."""
    counts: Dict[str, int] = {}
    order: list[str] = []
    for outcome in outcomes:
        language = (outcome.language or "").strip()
        chars = count_text_chars(outcome.text)
        if not language or chars <= 0:
            continue
        key = abbreviate_language(language)
        if key not in counts:
            order.append(language)
            counts[key] = 0
        counts[key] += chars
    if not counts:
        return None
    return max(order, key=lambda language: counts[abbreviate_language(language)])


def _repetition_thresholds() -> RepetitionThresholds:
    """从配置读取复读检测阈值。"""
    return RepetitionThresholds(
        max_chars_per_second=settings.qwen_repetition_max_chars_per_second,
        min_chars=settings.qwen_repetition_min_chars,
        min_unit=settings.qwen_repetition_min_unit,
        min_times=settings.qwen_repetition_min_times,
    )


async def _apply_repetition_guard(text: str, duration_s: float) -> str:
    """检测并修复复读（生成幻觉循环）。

    Qwen3-ASR 偶尔会在某个 VAD 段上进入「同一短语无限重复」的循环，
    直到撞上 max_model_len 被截断（实测产出 1863 字符的复读串）。

    这里 MUST 在 ITN 规范化之前执行：复读是模型原始输出就带的，ITN 会
    改写数字从而改变字符数，让「字符数 ÷ 时长」的判据偏移。

    只对「长度异常且检出复读结构」的文本动手；长度异常但没有复读结构的
    文本保留原文（由 utils.repetition 记录告警），避免凭空截断有效内容。
    """
    if not settings.qwen_repetition_guard_enabled:
        return text
    thresholds = _repetition_thresholds()
    stripped = (text or "").strip()
    if not stripped:
        return text
    # 长度判据很便宜，绝大多数分段在这里就返回了（实测 < 0.01ms）。
    if len(stripped) <= max_reasonable_chars(duration_s, thresholds):
        return text
    # 复读结构的扫描是正则与字符串比较，异常长文本实测可达 280ms，
    # 放到线程里执行，避免拖住事件循环。
    fixed, changed = await asyncio.to_thread(
        repair_repetition, text, duration_s, thresholds
    )
    # 未触发修复时逐字节保持原文，避免 strip 之类的副作用。
    return fixed if changed else text


async def _transcribe_segment(clip_path, language, concurrency):
    """请求 Qwen 转写一个片段，并剥离原生前缀。"""
    client = await get_qwen_client()
    stats = _phase_stats()
    wait_started = time.perf_counter()
    await concurrency.acquire_global()
    stats["qwen_wait_ms"] += (time.perf_counter() - wait_started) * 1000
    call_started = time.perf_counter()
    try:
        raw_text = await client.transcribe(clip_path, language=language)
    finally:
        await concurrency.release_global()
        stats["qwen_ms"] += (time.perf_counter() - call_started) * 1000
        stats["qwen_calls"] += 1

    parsed = parse_qwen_text(raw_text)
    return parsed.text, parsed.language


async def _speaker_embedding(audio_array, sample_rate):
    """提取片段的说话人嵌入（CAM++）。

    直接喂 numpy 数组而不是文件路径：funasr 的文件分支里有一次额外的
    音频读取与重采样，实测单次调用 98.8ms → 49.3ms，且嵌入结果逐位相同。
    """
    def _run(model, device):
        with torch.cuda.device(device):
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

    pool = get_model_pool("speech_campplus")
    lease_started = time.perf_counter()
    async with pool.lease() as leased:
        wait_ms = (time.perf_counter() - lease_started) * 1000
        infer_started = time.perf_counter()
        try:
            model, device = leased if isinstance(leased, tuple) else (
                leased, settings.model_device_speech_campplus)
            return await asyncio.to_thread(_run, model, device)
        finally:
            infer_ms = (time.perf_counter() - infer_started) * 1000
            record_inference = getattr(pool, "record_inference", None)
            if record_inference is not None:
                record_inference(wait_ms, infer_ms)
            stats = _phase_stats()
            stats["spk_wait_ms"] += wait_ms
            stats["spk_ms"] += infer_ms
            stats["spk_calls"] += 1


async def _segment_emotion(audio_array, sample_rate, duration_s):
    """推断片段情绪；超出 1~10 秒的片段记为平淡。"""
    if duration_s < EMOTION_MIN_DURATION_S or duration_s > EMOTION_MAX_DURATION_S:
        return "平淡"

    def _run(model, device):
        with torch.cuda.device(device):
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

    try:
        pool = get_model_pool("emotion2vec")
        lease_started = time.perf_counter()
        async with pool.lease() as leased:
            wait_ms = (time.perf_counter() - lease_started) * 1000
            infer_started = time.perf_counter()
            try:
                model, device = leased if isinstance(leased, tuple) else (
                    leased, settings.model_device_emotion2vec)
                return await asyncio.to_thread(_run, model, device)
            finally:
                infer_ms = (time.perf_counter() - infer_started) * 1000
                record_inference = getattr(pool, "record_inference", None)
                if record_inference is not None:
                    record_inference(wait_ms, infer_ms)
                stats = _phase_stats()
                stats["emo_wait_ms"] += wait_ms
                stats["emo_ms"] += infer_ms
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
    stats = {
        "spk_wait_ms": 0.0, "spk_ms": 0.0, "spk_calls": 0,
        "emo_wait_ms": 0.0, "emo_ms": 0.0, "emo_calls": 0,
        "qwen_wait_ms": 0.0, "qwen_ms": 0.0, "qwen_calls": 0,
    }
    _PHASE_STATS.set(stats)
    phase = {"vad_ms": 0.0, "cut_ms": 0.0, "segment_ms": 0.0, "cluster_ms": 0.0, "itn_ms": 0.0}

    need_role = bool(request.showSpk or request.showRoleIdentify)
    need_emotion = bool(request.showEmotion)

    await load_v2_models_if_needed(
        need_role=need_role,
        need_emotion=need_emotion,
    )

    vad_started = time.perf_counter()
    # 音频已在 prepare_asr_context 中加载；低音量过滤在裁剪和 Qwen 请求前完成。
    # 保持无张量的单测/兼容调用仍使用原有单参数接口。
    if ctx.audio_tensor is not None and ctx.sample_rate:
        segments = await detect_segments(
            ctx.tmp_path,
            audio_tensor=ctx.audio_tensor,
            sample_rate=ctx.sample_rate,
        )
    else:
        segments = await detect_segments(ctx.tmp_path)
    phase["vad_ms"] = (time.perf_counter() - vad_started) * 1000
    if not segments:
        update_fail_task()
        return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

    concurrency = get_request_concurrency()
    segment_semaphore = concurrency.new_segment_semaphore()

    async def _cut(segment: VadSegment):
        """惰性裁剪：只在该段真正要处理时才落盘一个 clip 文件。

        旧实现是在 VAD 之后一次性把全部分段落盘，60 分钟音频 943 段约 219 MB，
        峰值占用与分段总数成正比；改成按需创建后，同时存在的文件数等于在途段数。
        """
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
        return path, array, sample_rate

    def _release_clip(path: Optional[str]) -> None:
        """删除段 clip，并从兜底清理列表里摘除，避免请求结束时重复删除。"""
        if not path:
            return
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001 - 删除失败不应影响转写结果
            logger.warning(f"[V2] 删除段 clip 失败: {path}, 错误: {exc}")
        finally:
            if path in ctx.tmp_paths:
                ctx.tmp_paths.remove(path)

    async def _transcribe_and_cleanup(clip_path, language):
        """转写返回即删段文件。

        role / emotion 走的是内存里的 numpy 数组，不依赖该文件，因此这里删除
        不会影响同段的说话人与情绪计算，两者仍与转写并行。
        """
        try:
            return await _transcribe_segment(clip_path, language, concurrency)
        finally:
            _release_clip(clip_path)

    async def _process(index: int, segment: VadSegment) -> SegmentOutcome:
        outcome = SegmentOutcome(index=index, segment=segment)
        # 段级名额先占住：在途段数（即同时存在的 clip 文件数）由它限流，
        # 否则全部分段会在裁剪阶段同时落盘，临时文件峰值又回到与分段总数成正比。
        await segment_semaphore.acquire()
        clip_path: Optional[str] = None
        cut_started = time.perf_counter()
        try:
            try:
                clip_path, audio_array, sample_rate = await _cut(segment)
            except Exception as exc:
                logger.error(f"[V2] 音频裁剪失败: {exc}")
                raise SegmentCutError(f"音频分块失败: {exc}") from exc
            finally:
                phase["cut_ms"] += (time.perf_counter() - cut_started) * 1000

            tasks = [_transcribe_and_cleanup(clip_path, request.language)]
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
            # 复读修复 MUST 在 ITN 规范化与标点卫生之前：见 _apply_repetition_guard
            outcome.text = await _apply_repetition_guard(outcome.text, segment.duration_s)

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
        finally:
            # 兜底：转写协程尚未接管该文件时（如取消）由这里删除
            _release_clip(clip_path)
            segment_semaphore.release()

    segment_started = time.perf_counter()
    outcomes = await asyncio.gather(
        *[_process(i, seg) for i, seg in enumerate(segments)],
        return_exceptions=True,
    )
    phase["segment_ms"] = (time.perf_counter() - segment_started) * 1000

    for result in outcomes:
        if isinstance(result, BaseException):
            if isinstance(result, SegmentCutError):
                update_fail_task()
                return {"msg": "音频分块失败，请重试", "code": 4007}
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
    # Remove standalone interjections before language statistics and speaker clustering.
    valid = [o for o in valid if not is_single_interjection(o.text)]
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

    detected_language = _dominant_language(valid)
    english_dominant = _itn_language(detected_language) == "en"

    output_segments = []
    for outcome in valid:
        start_s = outcome.segment.start_s
        end_s = outcome.segment.end_s
        raw_text = outcome.text
        if english_dominant:
            outcome.text = replace_english_interjections(outcome.text)
        output_segments.append(
            {
                "segment_text": outcome.text,
                "bg": f"{start_s:.2f}",
                "ed": f"{end_s:.2f}",
                "speed": 0,
                "segment_words": [],
                "role": outcome.role,
                "emotion": outcome.emotion,
                "_language": outcome.language,
                "_raw_text": raw_text,
            }
        )

    itn_language = _itn_language(detected_language)
    config = getattr(settings, "itn_config", ItnConfig())
    itn_started = time.perf_counter()
    output_segments, normalized_text, _ = normalize_segments(
        output_segments, itn_language, config
    )
    # 标点卫生：丢弃纯标点段、移除段首连续标点。
    # 该规则与 ITN 开关无关，规范化关闭时同样执行。
    output_segments, normalized_text = sanitize_segments(output_segments)
    for segment in output_segments:
        start_s = float(segment["bg"])
        end_s = float(segment["ed"])
        segment["speed"] = calculate_speech_rate(
            segment.get("segment_text", ""),
            start_s,
            end_s,
            settings.speech_rate_factor,
        )
    speed_info = build_speed_info(output_segments, total_duration=ctx.audio_total_s)
    language_ratio = build_language_ratio(
        (segment.get("_language"), segment.get("_raw_text"))
        for segment in output_segments
    )
    for segment in output_segments:
        segment.pop("_language", None)
        segment.pop("_raw_text", None)
    phase["itn_ms"] = (time.perf_counter() - itn_started) * 1000

    gpu_time_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "[V2 耗时] 段数=%d 总计=%.0fms | VAD=%.0f 裁剪=%.0f 分段处理=%.0f "
        "(Qwen 等待 %.0fms/调用 %.0fms/%d次, 说话人等待 %.0fms/推理 %.0fms/%d次, "
        "情绪等待 %.0fms/推理 %.0fms/%d次) 聚类=%.0f ITN=%.0f",
        len(segments),
        gpu_time_ms,
        phase["vad_ms"],
        phase["cut_ms"],
        phase["segment_ms"],
        stats["qwen_wait_ms"],
        stats["qwen_ms"],
        int(stats["qwen_calls"]),
        stats["spk_wait_ms"],
        stats["spk_ms"],
        int(stats["spk_calls"]),
        stats["emo_wait_ms"],
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
