"""Qwen3-ASR 返回文本的解析：前缀剥离、语种提取与语种占比统计。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# 模型原生返回形如：language Chinese<asr_text>识别文本
_PREFIX_PATTERN = re.compile(r"language\s+(?P<language>[^<\s]*)\s*<asr_text>", re.IGNORECASE)

# 模型语种名 -> 简写标识。命中则替换键，未命中则原样透传。
LANGUAGE_ABBREVIATIONS: Dict[str, str] = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Spanish": "es",
    "French": "fr",
    "Thai": "th",
    "Malay": "ms",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Hindi": "hi",
    "Arabic": "ar",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Greek": "el",
    "Hebrew": "he",
}


def abbreviate_language(language: str) -> str:
    """把模型返回的语种名换成简写；不在映射表中的原样返回。"""
    return LANGUAGE_ABBREVIATIONS.get(language, language)


@dataclass
class QwenSegmentResult:
    """单段 Qwen 返回的解析结果。"""

    text: str
    language: Optional[str]

    @property
    def has_speech(self) -> bool:
        return bool(self.text.strip()) and bool(self.language)


def parse_qwen_text(raw_text: Optional[str]) -> QwenSegmentResult:
    """解析 Qwen 原始返回文本，剥离 ``language <语种><asr_text>`` 前缀。

    - 多个前缀（服务端按 max_model_len 自切 chunk）时拼接各片段文本；
    - 语种取第一个可识别的值；
    - 前缀不会进入返回的文本。
    """
    if not raw_text:
        return QwenSegmentResult(text="", language=None)

    matches = list(_PREFIX_PATTERN.finditer(raw_text))
    if not matches:
        # 没有前缀：无法判定语种，按无有效语音处理。
        return QwenSegmentResult(text="", language=None)

    pieces: List[str] = []
    language: Optional[str] = None
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        pieces.append(raw_text[start:end])
        if language is None:
            candidate = (match.group("language") or "").strip()
            if candidate and candidate.lower() != "none":
                language = candidate

    text = "".join(pieces).strip()
    return QwenSegmentResult(text=text, language=language)


def count_text_chars(text: Optional[str]) -> int:
    """统计参与语种占比计算的有效字符数（不含空白）。"""
    if not text:
        return 0
    return sum(1 for char in text if not char.isspace())


def build_language_ratio(items: Iterable[Tuple[Optional[str], Optional[str]]]) -> Dict[str, float]:
    """按 (语种, 文本) 统计各语种文本字符占比（百分比，保留一位小数）。

    键为语种简写（如 zh / en）；不在映射表中的语种原样透传。
    各语种占比之和为 100.0；没有有效文本时返回空字典。
    """
    counter: Dict[str, int] = {}
    total = 0
    for language, text in items:
        if not language:
            continue
        chars = count_text_chars(text)
        if chars <= 0:
            continue
        key = abbreviate_language(language)
        counter[key] = counter.get(key, 0) + chars
        total += chars

    if total <= 0:
        return {}

    ratio: Dict[str, float] = {}
    for language, chars in counter.items():
        ratio[language] = round(chars * 100.0 / total, 1)
    return ratio


__all__ = [
    "LANGUAGE_ABBREVIATIONS",
    "QwenSegmentResult",
    "abbreviate_language",
    "build_language_ratio",
    "count_text_chars",
    "parse_qwen_text",
]
