"""v3（Qwen3-ASR）转写链路：VAD 切分 → 并发转写 → 响应封装。"""

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
from core.async_utils import await_completion, run_in_thread
from core.local_inference import generate_local
from core.models import get_model_pool, load_v3_models_if_needed, validate_v3_capabilities
from core.qwen_asr_client import (
    QwenAsrCallError,
    QwenAsrUnavailableError,
    QwenAsrUnsupportedLanguageError,
    get_qwen_client,
    release_qwen_request,
)
from core.qwen_asr_queue import get_request_concurrency
from core.segment_pipeline import get_pending_budget, run_bounded_segments, gather_owned_branches
from core.request_telemetry import update_active
from core.vad import VadSegment, detect_segments, get_vad_stats
from utils.audio_utils import (
    extract_audio_clip,
    extract_audio_clip_array,
    # 保留导入别名，兼容外部调用方和旧版测试的打桩入口；实际链路使用
    # extract_audio_clip_array + save_audio_clip_array，避免重复切片。
    encode_audio_clip_wav,
    read_audio_array,
    save_audio_clip_array,
    extract_audio_clip_from_tensor,
    write_audio_bytes_to_temp_file,
)
from core.progress import create_audio_progress
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

# 统一最短有效段；0.3～不足1秒CAM仍推理，emo保留原1秒下界。
SHORT_SPEAKER_SEGMENT_S = 0.3

# 分阶段耗时统计：按请求隔离（子任务继承请求上下文，统计字典为同一对象），
# 用于定位端到端时延的构成
_PHASE_STATS: ContextVar[Optional[Dict[str, float]]] = ContextVar(
    "qwen_phase_stats", default=None
)


class SegmentCutError(RuntimeError):
    """单个 VAD 段的音频裁剪失败。"""


@dataclass(frozen=True)
class V3Capabilities:
    need_role: bool
    need_emotion: bool

    @property
    def category(self) -> str:
        return "B" if self.need_role or self.need_emotion else "A"


def resolve_v3_capabilities(request) -> V3Capabilities:
    """只解析请求需求，不加载模型或占用名额。"""
    return V3Capabilities(
        need_role=bool(request.showSpk or request.showRoleIdentify),
        need_emotion=bool(request.showEmotion),
    )


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


def get_phase_stats() -> Dict[str, float]:
    """返回当前请求的阶段统计副本，供状态/诊断遥测读取。"""
    stats = _PHASE_STATS.get()
    return dict(stats or {})


def reset_phase_stats() -> Dict[str, float]:
    stats = {key: 0.0 for key in (
        "spk_wait_ms", "spk_ms", "spk_calls", "emo_wait_ms", "emo_ms", "emo_calls",
        "qwen_wait_ms", "qwen_ms", "qwen_calls", "vad_ms", "cut_sum_ms", "cut_wall_ms",
        "segment_ms", "cluster_ms", "itn_ms", "cleanup_ms", "audio_wait_ms",
        "cut_cpu_ms", "cut_dispatch_wait_ms", "pending_wait_ms",
    )}
    _PHASE_STATS.set(stats)
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
    fixed, changed = await run_in_thread(
        repair_repetition, text, duration_s, thresholds
    )
    # 未触发修复时逐字节保持原文，避免 strip 之类的副作用。
    return fixed if changed else text


class _QwenCredit:
    """单段Qwen逻辑名额，前置裁剪后可转交；所有退出路径只归还一次。"""
    def __init__(self, concurrency, segment_semaphore):
        self.concurrency = concurrency
        self.segment_semaphore = segment_semaphore
        self.held = True

    async def release(self):
        if not self.held:
            return
        self.held = False
        if self.segment_semaphore is None:
            await await_completion(self.concurrency.release_global())
        else:
            await await_completion(self.concurrency.release(self.segment_semaphore))


async def _transcribe_segment(
    clip_bytes, language, concurrency, filename="segment.wav", *,
    segment_semaphore=None, request_id=None, segment_index=None, qwen_credit=None,
):
    """请求 Qwen 转写一个内存 WAV 片段，并剥离原生前缀。"""
    stats = _phase_stats()
    client = None
    if qwen_credit is None:
        # 直接调用保留原接口：没有前置credit时自行拿名额。
        client = await get_qwen_client()
        wait_started = time.perf_counter()
        if segment_semaphore is None:
            await concurrency.acquire_global()
        else:
            await concurrency.acquire(segment_semaphore)
        qwen_credit = _QwenCredit(concurrency, segment_semaphore)
        stats["qwen_wait_ms"] += (time.perf_counter() - wait_started) * 1000
    call_started = time.perf_counter()
    try:
        # 已有credit时发现客户端失败/取消也在finally立刻释放，不等增强join。
        if client is None:
            client = await get_qwen_client()
        transcribe_bytes = getattr(client, "transcribe_bytes", None)
        if callable(transcribe_bytes):
            metadata = ({"request_id": request_id, "segment_index": segment_index}
                        if getattr(client, "supports_batch_metadata", False) else {})
            raw_text = await transcribe_bytes(
                clip_bytes, language=language, filename=filename, **metadata
            )
        else:
            # 兼容迁移期间的旧客户端/测试桩。正式 QwenAsrClient 提供
            # transcribe_bytes；旧实现只能接收路径，因此仅在这个兼容
            # 分支创建服务专属临时文件，并在调用返回/异常后立即删除。
            compat_path = None

            def _write_compat():
                nonlocal compat_path
                compat_path = write_audio_bytes_to_temp_file(
                    clip_bytes, file_name=filename, suffix=".wav", unique=True,
                )

            try:
                await run_in_thread(_write_compat)
                raw_text = await client.transcribe(compat_path, language=language)
            finally:
                if compat_path is not None:
                    try:
                        os.remove(compat_path)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning("删除 Qwen 兼容段缓存失败: %s, 错误: %s", compat_path, exc)
    finally:
        await qwen_credit.release()
        update_active(increment="qwen_done")
        stats["qwen_ms"] += (time.perf_counter() - call_started) * 1000
        stats["qwen_calls"] += 1

    parsed = parse_qwen_text(raw_text)
    return parsed.text, parsed.language


async def _speaker_embedding(audio_array, sample_rate):
    """提取片段的说话人嵌入（CAM++）。

    直接喂 numpy 数组而不是文件路径：funasr 的文件分支里有一次额外的
    音频读取与重采样，实测单次调用 98.8ms → 49.3ms，且嵌入结果逐位相同。
    """
    validate_v3_capabilities(need_role=True)
    def _run(model, device):
        result = generate_local(model, device, input=audio_array, fs=sample_rate)
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
            return await run_in_thread(_run, model, device)
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
    validate_v3_capabilities(need_emotion=True)
    samples = len(audio_array)
    if samples < round(EMOTION_MIN_DURATION_S * sample_rate) or samples > round(EMOTION_MAX_DURATION_S * sample_rate):
        return "平淡"

    def _run(model, device):
        result = generate_local(model, device, input=audio_array, fs=sample_rate,
                                granularity="utterance", extract_embedding=False)
        if not result:
            return None
        payload = result[0]
        labels = payload.get("labels") or []
        scores = payload.get("scores") or []
        if not labels or not scores:
            return None
        best = max(range(len(scores)), key=lambda i: scores[i])
        return map_emotion(labels[best])

    pool = get_model_pool("emotion2vec")
    lease_started = time.perf_counter()
    async with pool.lease() as leased:
        wait_ms = (time.perf_counter() - lease_started) * 1000
        infer_started = time.perf_counter()
        try:
            model, device = leased if isinstance(leased, tuple) else (
                leased, settings.model_device_emotion2vec)
            return await run_in_thread(_run, model, device)
        finally:
            infer_ms = (time.perf_counter() - infer_started) * 1000
            record_inference = getattr(pool, "record_inference", None)
            if record_inference is not None:
                record_inference(wait_ms, infer_ms)
            stats = _phase_stats()
            stats["emo_wait_ms"] += wait_ms
            stats["emo_ms"] += infer_ms
            stats["emo_calls"] += 1


async def _segment_branch_with_progress(branch, *args, completion_field):
    """已进入分支的终态计一次：返回、失败、时长skip或取消，非NN成功次数。

    已进入但仍等待pool时取消也计终态；Task尚未进入本协程即取消则不计。
    branch在此调用才创建，未启动取消不会遗留未await的模型协程。
    原helper的run_in_thread会等已启动线程真实DONE/同步后才传播取消。
    """
    try:
        return await branch(*args)
    finally:
        update_active(increment=completion_field)


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
        if round(o.segment.duration_s * 16000) < round(SHORT_SPEAKER_SEGMENT_S * 16000)
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
        and round(o.segment.duration_s * 16000) >= round(SHORT_SPEAKER_SEGMENT_S * 16000)
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
    """v3 转写主链路。"""
    request = ctx.request
    capabilities = resolve_v3_capabilities(request)
    validate_v3_capabilities(capabilities.need_role, capabilities.need_emotion)
    started = time.perf_counter()
    stats = reset_phase_stats()
    phase = {
        "vad_ms": 0.0,
        "cut_sum_ms": 0.0,
        "cut_wall_ms": 0.0,
        "cut_wall_started": None,
        "cut_wall_ended": None,
        "segment_ms": 0.0,
        "cluster_ms": 0.0,
        "itn_ms": 0.0,
    }
    vad_stats = {"low_volume_dropped": 0}

    need_role = capabilities.need_role
    need_emotion = capabilities.need_emotion

    update_active("model_loading")
    await load_v3_models_if_needed(
        need_role=need_role,
        need_emotion=need_emotion,
    )

    update_active("vad")
    vad_started = time.perf_counter()
    # 音频已在 prepare_asr_context 中加载；低音量过滤在裁剪和 Qwen 请求前完成。
    # 保持无张量的单测/兼容调用仍使用原有单参数接口。
    async def _detect_with_stats():
        if ctx.audio_tensor is not None and ctx.sample_rate:
            detected = await detect_segments(
                ctx.tmp_path, audio_tensor=ctx.audio_tensor, sample_rate=ctx.sample_rate,
            )
        else:
            detected = await detect_segments(ctx.tmp_path)
        # ContextVar 的赋值留在子任务中，显式带回本次 VAD 统计。
        return detected, get_vad_stats()

    segments, vad_stats = await await_completion(_detect_with_stats())
    phase["vad_ms"] = (time.perf_counter() - vad_started) * 1000
    stats["vad_ms"] = phase["vad_ms"]
    for key in ("vad_wait_ms", "vad_worker_ms", "low_volume_filter_ms"):
        value = vad_stats.get(key)
        if isinstance(value, (int, float)):
            stats[key] = value
    # 采样点判定避免浮点相减把恰好0.3秒误删，响应两位小数不参与判断。
    sr = int(getattr(ctx, "sample_rate", None) or 16000)
    minimum_samples = max(round(0.3 * sr), int(np.ceil(settings.vad_min_segment_duration_s * sr)))
    segments = [seg for seg in segments
                if round(seg.end_s * sr) - round(seg.start_s * sr) >= minimum_samples]
    update_active("segments", segments_total=len(segments))
    if not segments:
        update_fail_task()
        return {"msg": "音频文件为空或未检测到任何人声", "code": 4008}

    concurrency = get_request_concurrency()
    segment_semaphore = concurrency.new_segment_semaphore()

    progress = None
    pending_budget = get_pending_budget()

    async def _cut(segment: VadSegment):
        """从已加载张量裁剪，返回 CAM/emo 数组和内存 WAV 字节。"""
        def timed(function, *args):
            began = time.perf_counter()
            value = function(*args)
            return value, (time.perf_counter() - began) * 1000

        work_ms = 0.0
        if ctx.audio_tensor is not None and ctx.sample_rate:
            (array, sample_rate), slice_ms = await run_in_thread(
                timed, extract_audio_clip_array,
                ctx.audio_tensor,
                ctx.sample_rate,
                segment.start_s,
                segment.duration_s,
            )
            clip_bytes, encode_ms = await run_in_thread(
                timed, encode_audio_clip_wav, array, sample_rate
            )
            work_ms = slice_ms + encode_ms
        else:
            compat_began = time.perf_counter()
            # 兼容尚未预加载张量的旧调用方。正常 v3 请求总是走上面的
            # 内存张量分支；这里只为旧客户端/测试保留一次兼容裁剪，并
            # 在读取后立即登记并删除生成的路径。
            clip_path = None

            def _extract_compat():
                nonlocal clip_path
                clip_path = extract_audio_clip(ctx.tmp_path, segment.start_s, segment.duration_s)
                ctx.tmp_paths.append(clip_path)

            try:
                # 线程内先登记文件，取消时即使 await 没有返回也能准确清理。
                await run_in_thread(_extract_compat)
                array, sample_rate = await run_in_thread(read_audio_array, clip_path)
                array = array.reshape(-1).astype("float32")
                if len(array) == 0:
                    raise SegmentCutError("裁剪区间为空")
                clip_bytes = await run_in_thread(encode_audio_clip_wav, array, sample_rate)
            finally:
                if clip_path is not None:
                    try:
                        os.remove(clip_path)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        logger.warning("删除兼容段缓存失败: %s, 错误: %s", clip_path, exc)
                    else:
                        try:
                            ctx.tmp_paths.remove(clip_path)
                        except ValueError:
                            pass
            work_ms = (time.perf_counter() - compat_began) * 1000
        return clip_bytes, array, sample_rate, work_ms

    async def _process(index: int, segment: VadSegment) -> SegmentOutcome:
        outcome = SegmentOutcome(index=index, segment=segment)
        # 汇合/PCM预算独立于Qwen窗口，增强不再持有已完成Qwen的名额。
        pending_lease = None
        qwen_credit = None
        segment_failed = False
        try:
            # 先per-audio16再global128，首文件不能预切256 PCM挡住后ready文件。
            qwen_wait_started = time.perf_counter()
            await concurrency.acquire(segment_semaphore)
            qwen_credit = _QwenCredit(concurrency, segment_semaphore)
            stats["qwen_wait_ms"] += (time.perf_counter() - qwen_wait_started) * 1000
            # cut_sum保留原口径(pending+裁剪)，不混入前置Qwen credit等待。
            cut_started = time.perf_counter()
            # 正式输入已标准化16k单声道，包含float32数组、PCM16 WAV及头部。
            reserved_bytes = max(1, round(segment.duration_s * 16000)) * 6 + 4096
            pending_lease = await pending_budget.acquire(reserved_bytes)
            stats["pending_wait_ms"] += (time.perf_counter() - cut_started) * 1000
            pure_cut_started = time.perf_counter()
            cut_work_ms = 0.0
            try:
                now = time.perf_counter()
                if phase["cut_wall_started"] is None:
                    phase["cut_wall_started"] = now
                clip_bytes, audio_array, sample_rate, cut_work_ms = await _cut(segment)
                phase["cut_wall_ended"] = time.perf_counter()
                update_active(increment="cut_done")
            except Exception as exc:
                logger.error(f"[V3] 音频裁剪失败: {exc}")
                segment_failed = True
                raise SegmentCutError(f"音频分块失败: {exc}") from exc
            finally:
                phase["cut_sum_ms"] += (time.perf_counter() - cut_started) * 1000
                cut_wall_ms = (time.perf_counter() - pure_cut_started) * 1000
                stats["cut_cpu_ms"] += cut_work_ms
                stats["cut_dispatch_wait_ms"] += max(0.0, cut_wall_ms - cut_work_ms)

            # EOF或裁剪后的实际PCM才是三模型共同阈值；不足0.3秒无推理/无响应。
            if len(audio_array) < max(round(0.3 * sample_rate),
                                      int(np.ceil(settings.vad_min_segment_duration_s * sample_rate))):
                return outcome

            tasks = [
                _transcribe_segment(
                    clip_bytes,
                    request.language,
                    concurrency,
                    filename=f"{getattr(ctx, 'task_id', 'audio')}-{index}.wav",
                    segment_semaphore=segment_semaphore,
                    request_id=str(getattr(ctx, "task_id", "audio")), segment_index=index,
                    qwen_credit=qwen_credit,
                )
            ]
            # 短CAM成功embedding同样参与聚类，emo在助手内按1～10秒判断。
            if need_role:
                tasks.append(_segment_branch_with_progress(
                    _speaker_embedding, audio_array, sample_rate, completion_field="spk_done",
                ))
            if need_emotion:
                tasks.append(
                    _segment_branch_with_progress(
                        _segment_emotion, audio_array, sample_rate, segment.duration_s,
                        completion_field="emo_done",
                    )
                )

            results = await gather_owned_branches(tasks)

            text_result = results[0]
            if isinstance(text_result, BaseException):
                segment_failed = True
                raise text_result
            outcome.text, outcome.language = text_result
            # 复读修复 MUST 在 ITN 规范化与标点卫生之前：见 _apply_repetition_guard
            outcome.text = await _apply_repetition_guard(outcome.text, segment.duration_s)

            cursor = 1
            if need_role:
                embedding = results[cursor]
                if isinstance(embedding, BaseException):
                    segment_failed = True
                    logger.warning(f"[说话人] 嵌入提取失败: {embedding}")
                    embedding = None
                outcome.embedding = embedding
                cursor += 1
            if need_emotion:
                emotion = results[cursor]
                if isinstance(emotion, BaseException):
                    segment_failed = True
                    logger.warning(f"[情绪] 推理失败: {emotion}")
                    emotion = None
                outcome.emotion = emotion

            return outcome
        except BaseException:
            segment_failed = True
            raise
        finally:
            # 一个段的 Qwen、CAM++、emotion2vec 全部结束后统一推进一次。
            if progress is not None:
                progress.update(failed=segment_failed)
            update_active(increment="segments_done")
            # 裁剪错误、EOF短段、分支未启动时兜底；Qwen已归还则为空操作。
            try:
                if qwen_credit is not None:
                    await qwen_credit.release()
            finally:
                if pending_lease is not None:
                    await await_completion(pending_budget.release(pending_lease))

    segment_started = time.perf_counter()
    progress = create_audio_progress(
        len(segments), f"{getattr(ctx, 'filename', 'audio')} 有效段"
    )
    try:
        outcomes = await run_bounded_segments(
            segments, _process,
            # 每音频只预建有限元任务；PCM/实际分支开工仍由全局段和字节预算控制。
            # 总8音频时最多8*pending_limit元任务，不将它误报为全局PCM活动数。
            settings.qwen_pending_segment_limit,
            **({"fail_fast": True} if getattr(settings, "qwen_generation_mode", "legacy") == "continuous" else {}),
        )
    finally:
        # run_bounded_segments does not return/propagate cancellation until all
        # owned segment branches have drained. Only now may worker dedup entries go.
        try:
            progress.close()
        finally:
            await await_completion(release_qwen_request(str(getattr(ctx, "task_id", "audio"))))
    phase["segment_ms"] = (time.perf_counter() - segment_started) * 1000
    stats["segment_ms"] = phase["segment_ms"]
    if phase["cut_wall_started"] is not None and phase["cut_wall_ended"] is not None:
        phase["cut_wall_ms"] = (phase["cut_wall_ended"] - phase["cut_wall_started"]) * 1000
    stats["cut_sum_ms"] = phase["cut_sum_ms"]
    stats["cut_wall_ms"] = phase["cut_wall_ms"]

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
            logger.error(f"[V3] 转写失败: {result}")
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
        update_active("cluster")
        cluster_started = time.perf_counter()
        speaker_labels = cluster_speakers(valid)
        for outcome in valid:
            outcome.role = speaker_labels.get(outcome.index)
        if request.showRoleIdentify:
            _apply_role_identify(valid)
        phase["cluster_ms"] = (time.perf_counter() - cluster_started) * 1000
        stats["cluster_ms"] = phase["cluster_ms"]

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
    update_active("itn")
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
    stats["itn_ms"] = phase["itn_ms"]

    gpu_time_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "[V3 耗时] 段数=%d 总计=%.0fms | VAD=%.0f 裁剪累计=%.0f 裁剪墙钟=%.0f 分段处理=%.0f "
        "(Qwen 等待 %.0fms/调用 %.0fms/%d次, 说话人等待 %.0fms/推理 %.0fms/%d次, "
        "情绪等待 %.0fms/推理 %.0fms/%d次) 聚类=%.0f ITN=%.0f 低音量丢弃=%d task_id=%s",
        len(segments),
        gpu_time_ms,
        phase["vad_ms"],
        phase["cut_sum_ms"],
        phase["cut_wall_ms"],
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
        int(vad_stats.get("low_volume_dropped", 0)),
        getattr(ctx, "task_id", "-"),
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
    "get_phase_stats",
    "run_qwen_asr",
]
