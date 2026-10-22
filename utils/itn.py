"""WeTextProcessing based inverse text normalization for offline ASR."""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItnConfig:
    enabled: bool = True
    enable_standalone_number: bool = True
    enable_0_to_9: bool = False
    remove_interjections: bool = True
    failure_mode: str = "passthrough"


_NORMALIZER_CACHE: dict[tuple[str, bool, bool, bool], Any] = {}
_CACHE_LOCK = threading.Lock()


def _language(language: str | None) -> str | None:
    normalized = (language or "").strip().lower()
    if normalized == "en":
        return "en"
    if normalized in {"auto", "zh"}:
        return "zh"
    return None


def _normalizer_key(language: str, config: ItnConfig) -> tuple[str, bool, bool, bool]:
    selected = _language(language)
    if selected is None:
        raise ValueError(f"unsupported ITN language: {language}")
    return (
        selected,
        config.enable_standalone_number,
        config.enable_0_to_9,
        config.remove_interjections,
    )


def _build_normalizer(language: str, config: ItnConfig) -> Any:
    lang = _language(language)
    if lang == "en":
        normalizer_cls = importlib.import_module(
            "itn.english.inverse_normalizer"
        ).InverseNormalizer
        # English 1.2.0 has no standalone-number or interjection constructor
        # switches; its graph is still selected explicitly by language.
        return normalizer_cls()

    normalizer_cls = importlib.import_module(
        "itn.chinese.inverse_normalizer"
    ).InverseNormalizer
    return normalizer_cls(
        enable_standalone_number=config.enable_standalone_number,
        enable_0_to_9=config.enable_0_to_9,
        remove_interjections=config.remove_interjections,
    )


def _get_normalizer(language: str, config: ItnConfig) -> Any:
    key = _normalizer_key(language, config)
    normalizer = _NORMALIZER_CACHE.get(key)
    if normalizer is not None:
        return normalizer
    with _CACHE_LOCK:
        normalizer = _NORMALIZER_CACHE.get(key)
        if normalizer is None:
            normalizer = _build_normalizer(language, config)
            _NORMALIZER_CACHE[key] = normalizer
    return normalizer


def itn_dependency_status(config: ItnConfig | None = None) -> dict[str, Any]:
    """Return dependency availability without compiling an FST."""
    config = config or ItnConfig()
    if not config.enabled:
        return {"enabled": False, "available": True, "language_graphs": []}
    try:
        importlib.import_module("pynini")
        importlib.import_module("itn.chinese.inverse_normalizer")
        importlib.import_module("itn.english.inverse_normalizer")
    except Exception as exc:  # pragma: no cover - depends on deployment env
        return {
            "enabled": True,
            "available": False,
            "language_graphs": [],
            "failure": type(exc).__name__,
        }
    return {"enabled": True, "available": True, "language_graphs": ["zh", "en"]}


def normalize_segments(
    segments: Iterable[dict],
    language: str,
    config: ItnConfig,
) -> tuple[list[dict], str, dict[str, Any]]:
    """Normalize segment text while preserving all segment metadata."""
    started = time.perf_counter()
    original_segments = list(segments)
    selected_language = _language(language)
    if selected_language is None:
        text = " ".join(
            str(segment.get("segment_text", "") or "").strip()
            for segment in original_segments
            if str(segment.get("segment_text", "") or "").strip()
        )
        return original_segments, text, {
            "enabled": False,
            "language": (language or "").strip().lower(),
            "segments": len(original_segments),
            "rewritten": 0,
            "failures": 0,
            "skipped": "unsupported_language",
            "duration_ms": (time.perf_counter() - started) * 1000,
        }
    if not config.enabled:
        text = " ".join(
            str(segment.get("segment_text", "") or "").strip()
            for segment in original_segments
            if str(segment.get("segment_text", "") or "").strip()
        )
        return original_segments, text, {
            "enabled": False,
            "language": selected_language,
            "segments": len(original_segments),
            "rewritten": 0,
            "failures": 0,
            "duration_ms": (time.perf_counter() - started) * 1000,
        }

    try:
        normalizer = _get_normalizer(language, config)
    except Exception as exc:
        logger.error(
            "ITN 初始化失败 language=%s stage=initialize error=%s segments=%s",
            selected_language,
            type(exc).__name__,
            len(original_segments),
        )
        if config.failure_mode == "passthrough":
            text = " ".join(
                str(segment.get("segment_text", "") or "").strip()
                for segment in original_segments
                if str(segment.get("segment_text", "") or "").strip()
            )
            return original_segments, text, {
                "enabled": True,
                "language": selected_language,
                "segments": len(original_segments),
                "rewritten": 0,
                "failures": len(original_segments),
                "failure_stage": "initialize",
                "duration_ms": (time.perf_counter() - started) * 1000,
            }
        raise

    output: list[dict] = []
    rewritten = 0
    failures = 0
    for segment in original_segments:
        item = dict(segment)
        raw_text = str(segment.get("segment_text", "") or "")
        stripped = raw_text.strip()
        if not stripped:
            output.append(item)
            continue
        try:
            normalized = str(normalizer.normalize(stripped) or "").strip()
            if normalized != stripped:
                rewritten += 1
            item["segment_text"] = normalized
        except Exception as exc:
            failures += 1
            logger.warning(
                "ITN 分段失败 language=%s stage=segment error=%s",
                selected_language,
                type(exc).__name__,
            )
            if config.failure_mode != "passthrough":
                raise
            item["segment_text"] = raw_text
        output.append(item)

    text = " ".join(
        str(segment.get("segment_text", "") or "").strip()
        for segment in output
        if str(segment.get("segment_text", "") or "").strip()
    )
    stats = {
        "enabled": True,
        "language": selected_language,
        "segments": len(original_segments),
        "rewritten": rewritten,
        "failures": failures,
        "duration_ms": (time.perf_counter() - started) * 1000,
    }
    logger.info(
        "ITN 完成 language=%s segments=%s rewritten=%s failures=%s duration_ms=%.2f",
        stats["language"],
        stats["segments"],
        stats["rewritten"],
        stats["failures"],
        stats["duration_ms"],
    )
    return output, text, stats


def clear_normalizer_cache() -> None:
    with _CACHE_LOCK:
        _NORMALIZER_CACHE.clear()
