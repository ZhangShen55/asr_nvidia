"""Retain one abnormal tail-loop copy or a conservative non-loop prefix.

Retained model text is not human-verified transcription.
Every path emits a warning; model/device/transport failures must never enter here.
"""
from __future__ import annotations
import re
from utils.incremental_repetition import RepetitionGuardConfig, suffix_repeat

_RECOVERABLE = {"repetition_stop", "length_limit", "invalid_output_prefix"}
_PREFIX = re.compile(r"^\s*language\s+([^<\n]+)<asr_text>", re.I)
_END = re.compile(r"[。！？.!?](?:[”’\"']*)")
_FILLER = re.compile(r"^(?:啊|哎|嗯|哦|呃|呀|对|是|好|天啊|我操|这个|那个|然后|所以|yes|no|ok|okay|uh|um|oh|ah|yeah)+$", re.I)


def _has_content(text):
    content = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if not content or _FILLER.fullmatch(content):
        return False
    return True


def _meaningful(text):
    if not _has_content(text):
        return False
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    return cjk >= 8 or len(words) >= 4


def _collapse_tail(body, loop):
    # 与suffix_repeat使用相同分隔符，仅折叠已确认的异常尾部；
    # 不扫描/折叠前文，避免把正常强调和课堂练习一起改写。
    positions = [i for i, char in enumerate(body)
                 if not re.fullmatch(r"[\s，。！？；：、,.!?;:]", char)]
    content = "".join(body[i] for i in positions)
    start = positions.index(loop["start_char"])
    width = loop["unit_chars"]
    # 停止在半个循环单元时，尾部周期可能旋转（很好…很变成好很）。
    # 沿同一周期回溯起点，再保留第一完整单元，避免遗留末尾残片。
    while start > 0 and content[start - 1] == content[start - 1 + width]:
        previous = start - 1
        gap = body[positions[previous] + 1:positions[start]]
        repeated_gap = body[positions[previous + width] + 1:positions[start + width]]
        # Listen: hello again…hello中的冒号不属于循环，不能跨入Listen的n；
        # 同时保留英文词界，避免忽略分隔符后把正常前文当作旋转周期。
        if re.sub(r"\s+", " ", gap) != re.sub(r"\s+", " ", repeated_gap):
            break
        start -= 1
    # Listen/again:等前文可能与字符周期重叠，首份单元还须匹配下一
    # 周期的内部词界/标点，并从完整英文单词开始，不能吞掉前文词尾。
    unit = body[positions[start]:positions[start + width - 1] + 1]
    latin_words = re.search(r"[A-Za-zÀ-ÿ]", unit) and any(c.isspace() for c in unit)
    for candidate in range(start, start + width):
        offset = positions[candidate]
        if latin_words and offset and re.match(r"[A-Za-zÀ-ÿ0-9_]", body[offset - 1]) and re.match(r"[A-Za-zÀ-ÿ0-9_]", body[offset]):
            continue
        layout_matches = True
        for index in range(candidate, candidate + width - 1):
            gap = body[positions[index] + 1:positions[index + 1]]
            repeated_gap = body[positions[index + width] + 1:positions[index + width + 1]]
            if re.sub(r"\s+", " ", gap) != re.sub(r"\s+", " ", repeated_gap):
                layout_matches = False
                break
        if layout_matches:
            start = candidate
            break
    end = positions[start + width - 1] + 1
    # 原字符切片保留前文、英文词间空格和第一单元；尾标点单独保留。
    return (body[:end] + body[positions[-1] + 1:]).strip()


def recover_segment_output(raw_text, reason, duration_s, *, config=None):
    if reason not in _RECOVERABLE:
        raise ValueError("not a recoverable generation anomaly: " + str(reason))
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be string")
    # Validate duration as the guard does, even when output has no usable prefix.
    import math
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive and finite")
    match = _PREFIX.match(raw_text)
    retained = ""
    loop = None
    if match and match.group(1).strip().lower() != "none" and reason != "invalid_output_prefix":
        body = re.sub(r"<\|[^<>]*\|>", "", raw_text[match.end():]).strip()
        from dataclasses import replace
        policy = config or RepetitionGuardConfig()
        loop = suffix_repeat(body, replace(policy, tail_chars=max(policy.tail_chars, len(body))))
        if loop:
            candidate = _collapse_tail(body, loop)
            # 已确认异常只折叠重复副本；2秒短句不能因不足8字/4词丢失。
            # 空内容和纯填充词仍不可作为恢复文本，恢复也不等于人工真值。
            if _has_content(candidate):
                retained = candidate
        else:
            # 非复读的长度耗尽保持原有完整句恢复策略，不放宽截断内容。
            completed = list(_END.finditer(body))
            candidate = body[:completed[-1].end()].strip() if completed else ""
            if candidate and _meaningful(candidate):
                retained = candidate
    recovered = bool(retained)
    action = "discard_unrecoverable"
    if recovered:
        action = "repetition_collapsed" if loop else "recover_prefix"
    return {"raw_text": (match.group(0).strip() + retained) if recovered else "language None<asr_text>",
            "text": retained, "trusted_prefix": retained, "recovered": recovered,
            "warning": {"reason": reason, "action": action, "recovered": recovered}}
