"""生成复读（幻觉循环）的检测与修复。

Qwen3-ASR 在个别音频片段上会进入「同一个短语无限重复」的循环，直到撞上
`max_model_len` 被 vLLM 截断。实测案例：6 秒的 VAD 段产出 1863 个字符的
「最小S零，这个这个这个，……」，语速达到 313 字/秒，远超人类上限。

本模块在文本落库前把这类复读折叠回正常长度：

1. **长度判据**：按语音时长估算合理字数上限，超过即视为异常；
2. **重复判据**：同一子串连续重复达到阈值即判定为循环；
3. **修复**：把循环片段折叠成一次出现，保留首份内容。
"""

from __future__ import annotations
import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# 字符/秒上限。标定依据（68,018 段真实样本）：
#   正常段 p99.9 = 20.3 字符/秒，最高 26.2；幻觉段 282~313。
#   取 40：约为最高正常值的 1.5 倍余量，同时远低于幻觉量级。
_MAX_CHARS_PER_SECOND = 40
_MIN_ABSOLUTE_CHARS = 120

# 同一子串连续重复达到该次数即判定为循环
_MIN_REPEAT_TIMES = 3
# 参与循环判定的子串最小长度（字符）。过短会把正常的叠词误判成循环。
_MIN_LOOP_UNIT = 4
_MAX_LOOP_UNIT = 60

# 循环单元之间允许出现的分隔符（复读常表现为「这个，这个，这个」）
_SEPARATORS = "，。！？；：、,.!?;: \n"


def max_reasonable_chars(duration_s: float) -> int:
    """给定时长下合理文本长度的上限。"""
    if duration_s <= 0:
        return _MIN_ABSOLUTE_CHARS
    return max(_MIN_ABSOLUTE_CHARS, int(duration_s * _MAX_CHARS_PER_SECOND))


def find_repeat_loop(
    text: str,
    min_unit: int = _MIN_LOOP_UNIT,
    min_times: int = _MIN_REPEAT_TIMES,
) -> Optional[Tuple[str, int, int]]:
    """找出文本中最长的连续重复片段。

    返回 (重复单元, 单元长度, 重复次数)；没有重复时返回 None。
    """
    if not text:
        return None
    length = len(text)
    best: Optional[Tuple[str, int, int]] = None
    best_size = 0
    # 单元越长越可能是真正的循环；先按长度降序扫，找到即返回
    upper = min(length // min_times, _MAX_LOOP_UNIT)
    for size in range(upper, min_unit - 1, -1):
        for start in range(0, length - size * min_times + 1):
            unit = text[start:start + size]
            if not unit.strip():
                continue
            times, _ = _count_repeats(text, unit, start, allow_separator=True)
            if times >= min_times:
                if size > best_size:
                    best_size = size
                    best = (unit, size, times)
                break
        if best:
            return best
    return best


def _count_repeats(
    text: str,
    unit: str,
    start: int,
    allow_separator: bool = False,
) -> Tuple[int, int]:
    """统计从 start 开始 unit 连续出现的次数，返回 (次数, 结束位置)。"""
    size = len(unit)
    count = 1
    pos = start + size
    while True:
        probe = pos
        if allow_separator:
            while probe < len(text) and text[probe] in _SEPARATORS:
                probe += 1
        if text[probe:probe + size] == unit:
            count += 1
            pos = probe + size
        else:
            return count, pos


def collapse_repetition(
    text: str,
    min_unit: int = _MIN_LOOP_UNIT,
    min_times: int = _MIN_REPEAT_TIMES,
) -> str:
    """把连续的重复片段折叠成一次出现。

    逐级放宽循环单元的长度：先找长单元（复读特征明显），找不到再放宽到
    短单元，覆盖「这个，这个，这个」这类短词复读。
    """
    if not text:
        return text
    result = text
    for unit_min, times_min in ((min_unit, min_times), (2, 4), (1, 6)):
        for _ in range(12):  # 最多折叠 12 轮，覆盖多层嵌套的循环
            found = find_repeat_loop(
                result, min_unit=unit_min, min_times=times_min
            )
            if not found:
                break
            unit, size, times = found
            sep = r"[，。！？；：、,.!?;:\s]*"
            pattern = f"(?:{re.escape(unit)}{sep}){{{times},}}"
            new_result, count = re.subn(pattern, unit, result)
            if count == 0:
                # 分隔符不完全一致时，退回到「单元 + 可选分隔符」的形式
                pattern = f"(?:{re.escape(unit)}(?:{sep}{re.escape(unit)})*){{1,}}"
                new_result, count = re.subn(pattern, unit, result)
            if count == 0:
                break
            result = new_result
            if len(result) >= len(text):
                break
    return result


def repair_repetition(text: str, duration_s: float = 0.0) -> Tuple[str, bool]:
    """检测并修复复读。

    只对**异常长度**的文本动手：长度正常时原样返回，避免把正常的叠词、
    歌词重复等误判成复读。

    返回 (修复后的文本, 是否发生了修复)。
    """
    stripped = (text or "").strip()
    if not stripped:
        return stripped, False

    limit = max_reasonable_chars(duration_s)
    if len(stripped) <= limit:
        return stripped, False

    collapsed = collapse_repetition(stripped)
    changed = collapsed != stripped

    if not changed:
        # 长度异常但没有检出复读结构：不改写、不截断，只记录告警。
        # 宁可放过，也不可凭空截断有效内容。
        logger.warning(
            "[复读检测] 长度 %d 超过上限 %d（时长 %.1fs），但未检出复读结构，保留原文",
            len(stripped),
            limit,
            duration_s,
        )
        return stripped, False

    if changed:
        logger.warning(
            "[复读修复] 长度 %d -> %d（原上限 %d，时长 %.1fs）: %r",
            len(stripped),
            len(collapsed),
            limit,
            duration_s,
            stripped[:60],
        )
    return collapsed, changed


__all__ = [
    "collapse_repetition",
    "find_repeat_loop",
    "max_reasonable_chars",
    "repair_repetition",
]
