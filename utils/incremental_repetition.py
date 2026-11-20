"""Conservative per-sequence repetition stop signal for incremental decoding.

This module never rewrites output, controls EOS, or accesses a device. Call it
with cumulative decoded output, only for a newly generated token. Keep one guard
per sequence; its memory and scan work are bounded by the configured tail size.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RepetitionGuardConfig:
    enabled: bool = True
    min_generated_tokens: int = 32
    min_chars: int = 120
    max_chars_per_second: int = 40
    min_repeats: int = 8
    min_repeat_chars: int = 64
    max_unit_chars: int = 32
    tail_chars: int = 512
    confirmations: int = 2

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be bool")
        for name in self.__dataclass_fields__:
            if name == "enabled":
                continue
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_repeats < 3 or self.confirmations < 2:
            raise ValueError("repetition stop requires repeats>=3 and confirmations>=2")
        if self.min_repeat_chars > self.tail_chars or self.max_unit_chars > self.tail_chars:
            raise ValueError("tail is too small for repetition thresholds")


@dataclass(frozen=True)
class RepetitionDecision:
    reason: str
    generated_tokens: int
    body_chars: int
    length_limit: int
    unit_chars: int
    repeats: int
    repeated_chars: int


def transcription_body(raw_text: str) -> str:
    """Do not treat language metadata or an incomplete ASR prefix as speech."""
    if "<asr_text>" not in raw_text:
        return ""
    return re.sub(r"<\|[^<>]*\|>", "", raw_text.split("<asr_text>", 1)[1])


class IncrementalRepetitionGuard:
    def __init__(self, duration_s: float, config: RepetitionGuardConfig | None = None):
        if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be positive and finite")
        self.config = config or RepetitionGuardConfig()
        self.length_limit = max(self.config.min_chars, math.ceil(duration_s * self.config.max_chars_per_second))
        self._last_tokens = -1
        self._last_chars = -1
        self._last_tail = ""
        self._confirmations = 0
        self._decision = None

    def update(self, raw_text: str, *, generated_tokens: int) -> RepetitionDecision | None:
        if type(generated_tokens) is not int or generated_tokens < 0:
            raise ValueError("generated_tokens must be a nonnegative integer")
        if not isinstance(raw_text, str):
            raise TypeError("raw_text must be a string")
        if self._decision is not None:
            return self._decision
        c = self.config
        if not c.enabled or generated_tokens <= self._last_tokens:
            return None
        self._last_tokens = generated_tokens
        body = transcription_body(raw_text)
        count = len(body)
        if generated_tokens < c.min_generated_tokens or count <= self.length_limit:
            self._confirmations = 0
            return None
        # Only bounded suffix normalization; do not strip punctuation in the
        # original result. No case-folding, number normalization or ITN.
        tail = re.sub(r"[\s，。！？；：、,.!?;:]+", "", body[-c.tail_chars:])
        if count <= self._last_chars:
            return None
        self._last_chars, self._last_tail = count, tail
        found = None
        for size in range(1, min(c.max_unit_chars, len(tail) // c.min_repeats) + 1):
            unit = tail[-size:]
            repeats = 1
            while (repeats + 1) * size <= len(tail) and tail[-(repeats+1)*size:-repeats*size] == unit:
                repeats += 1
            if repeats >= c.min_repeats and repeats * size >= c.min_repeat_chars:
                found = size, repeats
                break
        if found is None:
            self._confirmations = 0
            return None
        self._confirmations += 1
        if self._confirmations < c.confirmations:
            return None
        size, repeats = found
        self._decision = RepetitionDecision("repetition_stop", generated_tokens, count,
                                            self.length_limit, size, repeats, size * repeats)
        return self._decision
