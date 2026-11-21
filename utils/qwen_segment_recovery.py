"""Conservative partial retention for specified generation anomalies only.

A structurally credible model prefix is not human-verified transcription.
Every path emits a warning; model/device/transport failures must never enter here.
"""
from __future__ import annotations
import re
from utils.incremental_repetition import RepetitionGuardConfig, suffix_repeat

_RECOVERABLE = {"repetition_stop", "length_limit", "invalid_output_prefix"}
_PREFIX = re.compile(r"^\s*language\s+([^<\n]+)<asr_text>", re.I)
_END = re.compile(r"[。！？.!?](?:[”’\"']*)")
_FILLER = re.compile(r"^(?:啊|哎|嗯|哦|呃|呀|对|是|好|天啊|我操|这个|那个|然后|所以|yes|no|ok|okay|uh|um|oh|ah|yeah)+$", re.I)


def _meaningful(text):
    content = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    if not content or _FILLER.fullmatch(content):
        return False
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    words = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    return cjk >= 8 or len(words) >= 4


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
    if match and match.group(1).strip().lower() != "none" and reason != "invalid_output_prefix":
        body = re.sub(r"<\|[^<>]*\|>", "", raw_text[match.end():]).strip()
        from dataclasses import replace
        policy = config or RepetitionGuardConfig()
        loop = suffix_repeat(body, replace(policy, tail_chars=max(policy.tail_chars, len(body))))
        # Never retain one copy of a suspect loop merely because it was folded.
        # Only the non-loop prefix has potential structural credibility.
        prefix = body[:loop["start_char"]].strip() if loop else body
        completed = list(_END.finditer(prefix))
        candidate = prefix[:completed[-1].end()].strip() if completed else ""
        if candidate and _meaningful(candidate):
            retained = candidate
        elif loop and _meaningful(prefix):
            # Longer non-loop prefix with no sentence stop; no generated suffix
            # is invented, and incomplete/model-derived status remains warned.
            retained = prefix
    recovered = bool(retained)
    action = "recover_prefix" if recovered else "discard_unrecoverable"
    return {"raw_text": (match.group(0).strip() + retained) if recovered else "language None<asr_text>",
            "text": retained, "trusted_prefix": retained, "recovered": recovered,
            "warning": {"reason": reason, "action": action, "recovered": recovered}}
