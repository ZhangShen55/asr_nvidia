"""Small, deterministic text post-processing rules for the Qwen v2 response."""

from __future__ import annotations

from typing import Iterable, Mapping

INTERJECTIONS = frozenset("啊哦嗯呃")
ENGLISH_INTERJECTION_MAP = {"啊": "ah", "哦": "oh", "嗯": "um", "呃": "uh"}
_PUNCTUATION = frozenset(
    "，。！？；：、,.?!;:~…—–"
    "“”‘’\"'()（）《》〈〉【】〔〕「」『』"
)


def _trim_outer_punctuation(text: str) -> str:
    value = str(text or "").strip()
    start = 0
    end = len(value)
    while start < end and value[start] in _PUNCTUATION:
        start += 1
    while end > start and value[end - 1] in _PUNCTUATION:
        end -= 1
    return value[start:end].strip()


def is_single_interjection(text: str, interjections: Iterable[str] = INTERJECTIONS) -> bool:
    """Return whether a segment is only one configured interjection."""
    return _trim_outer_punctuation(text) in set(interjections)


def remove_single_interjection_segments(
    segments: Iterable[Mapping],
    interjections: Iterable[str] = INTERJECTIONS,
) -> list[dict]:
    """Drop pure interjection segments while preserving all retained metadata."""
    selected = set(interjections)
    return [
        dict(segment)
        for segment in segments
        if not is_single_interjection(str(segment.get("segment_text", "") or ""), selected)
    ]


def _is_boundary(text: str, index: int) -> bool:
    if index < 0 or index >= len(text):
        return True
    char = text[index]
    return char.isspace() or char in _PUNCTUATION


def replace_english_interjections(text: str) -> str:
    """Replace only boundary-delimited Chinese interjections in a text string."""
    source = str(text or "")
    if not source:
        return source
    output: list[str] = []
    for index, char in enumerate(source):
        replacement = ENGLISH_INTERJECTION_MAP.get(char)
        if replacement and _is_boundary(source, index - 1) and _is_boundary(source, index + 1):
            output.append(replacement)
        else:
            output.append(char)
    return "".join(output)


def replace_english_interjections_in_segments(segments: Iterable[Mapping]) -> list[dict]:
    """Copy segments and replace interjections without changing timing or metadata."""
    output = []
    for segment in segments:
        item = dict(segment)
        item["segment_text"] = replace_english_interjections(item.get("segment_text", ""))
        output.append(item)
    return output


__all__ = [
    "ENGLISH_INTERJECTION_MAP",
    "INTERJECTIONS",
    "is_single_interjection",
    "remove_single_interjection_segments",
    "replace_english_interjections",
    "replace_english_interjections_in_segments",
]
