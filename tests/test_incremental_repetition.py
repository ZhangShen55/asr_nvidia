import math

import pytest

from utils.incremental_repetition import (IncrementalRepetitionGuard, RepetitionGuardConfig,
                                          transcription_body)


def stream(text, duration=2.0, config=None):
    guard = IncrementalRepetitionGuard(duration, config)
    for index in range(1, len(text) + 1):
        decision = guard.update("language Chinese<asr_text>" + text[:index], generated_tokens=index)
        if decision:
            return index, decision
    return None


@pytest.mark.parametrize("text,duration", [
    ("重要的事情说三遍，注意，注意，注意。", 3),
    ("我我我觉得这个这个这个方法可以用。", 3),
    ("一二三四五六七八九十" * 10, 8),
    ("ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 5, 8),
    ("Very, very, very important. Yes, yes, yes!", 4),
    ("Twinkle twinkle little star, how I wonder what you are. " * 4, 12),
    ("学习学习再学习，坚持坚持再坚持。" * 6, 6),
    ("This is a repeated classroom exercise. " * 5, 10),
])
def test_protected_speech(text, duration):
    assert stream(text, duration) is None


@pytest.mark.parametrize("unit", ["这个", "重复的一句话", "hello again ", "a"])
def test_pathological_loop_stops_before_end(unit):
    text = "正常前文。" + unit * 400
    result = stream(text)
    assert result is not None
    index, decision = result
    assert index < len(text)
    assert decision.reason == "repetition_stop"
    assert decision.body_chars > decision.length_limit
    assert decision.repeats >= 8


def test_metadata_is_not_speech():
    guard = IncrementalRepetitionGuard(1)
    assert guard.update("language " + "Chinese" * 100, generated_tokens=100) is None
    assert transcription_body("language None<asr_text>") == ""


def test_confirmation_requires_new_body_not_duplicate_callback():
    guard = IncrementalRepetitionGuard(1)
    raw = "language Chinese<asr_text>" + "a" * 160
    assert guard.update(raw, generated_tokens=100) is None
    assert guard.update(raw, generated_tokens=101) is None
    assert guard.update(raw, generated_tokens=101) is None
    decision = guard.update(raw + "a", generated_tokens=102)
    assert decision.reason == "repetition_stop"
    assert guard.update(raw + "ab", generated_tokens=103) is decision


def test_guards_are_per_sequence_and_disabled_mode():
    raw = "language Chinese<asr_text>" + "a" * 160
    a, b = IncrementalRepetitionGuard(1), IncrementalRepetitionGuard(1)
    assert a.update(raw, generated_tokens=100) is None
    assert b.update(raw, generated_tokens=100) is None
    assert a.update(raw + "a", generated_tokens=101)
    assert b.update(raw + "xyz", generated_tokens=101) is None
    assert stream("a" * 1000, config=RepetitionGuardConfig(enabled=False)) is None


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf, True])
def test_bad_duration(duration):
    with pytest.raises(ValueError):
        IncrementalRepetitionGuard(duration)


@pytest.mark.parametrize("kwargs", [{"confirmations": 1}, {"min_repeats": 1},
                                    {"tail_chars": 32}, {"enabled": "yes"}])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        RepetitionGuardConfig(**kwargs)
