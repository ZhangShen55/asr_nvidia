import math
import re
import unicodedata
from collections import defaultdict


TEACHER_KEYWORDS = [
    # 中文关键词
    "上课", "考试", "上节课", "讲", "我们看", "我们讲", "大家", "同学",
    "请假", "签到", "重点", "考点", "了解一下", "我个人认为",
    "这个非常重要", "要记下来", "这里有", "注意", "这个要考", "下课",
    "休息", "我们", "一起讨论", "回答问题", "下次课",

    # 英文关键词
    "class", "lesson", "homework", "exam", "test", "quiz", "assignment",
    "let's discuss", "let's look at", "let's review", "everyone", "students",
    "important point", "key concept", "remember this", "take notes", "pay attention",
    "I want you to", "please focus", "for next class", "as I mentioned", "in our last class"
]

ENGLISH_QUESTION_WORDS = [
    "what", "how", "why", "when", "where", "who", "which", "whose", "whom",
    "can you", "could you", "would you", "will you", "do you", "did you", "have you"
]


def extract_features(segments: list[dict]):
    """提取各个 SPK ID 的发言特征。"""
    spk_features = defaultdict(lambda: {
        "utterance_count": 0,
        "total_length": 0,
        "avg_length": 0,
        "keyword_count": 0,
        "question_count": 0,
        "speech_time": 0,
        "segments": []
    })

    for segment in segments:
        role = segment["role"]
        content = segment["segment_text"]
        segment_time = float(segment["ed"]) - float(segment["bg"])

        spk_features[role]["utterance_count"] += 1
        spk_features[role]["total_length"] += len(content)
        spk_features[role]["speech_time"] += segment_time

        for keyword in TEACHER_KEYWORDS:
            if keyword in content:
                spk_features[role]["keyword_count"] += 1

        if content.endswith(("？", "?", "吗", "呢")):
            spk_features[role]["question_count"] += 1
        else:
            content_lower = content.lower()
            for question_word in ENGLISH_QUESTION_WORDS:
                if content_lower.startswith(question_word) or f" {question_word} " in content_lower:
                    spk_features[role]["question_count"] += 1
                    break

        spk_features[role]["segments"].append(f"{segment['bg']}-{segment['ed']}")

    for role in spk_features:
        count = spk_features[role]["utterance_count"]
        if count > 0:
            spk_features[role]["avg_length"] = spk_features[role]["total_length"] / count

    return spk_features


def identify_teacher(spk_features):
    """根据发言频率、长度、关键词和问句规则识别教师。"""
    scores = {}
    roles = list(spk_features.keys())
    for role, features in spk_features.items():
        scores[role] = (
            features["utterance_count"] * 1.5
            + features["avg_length"] * 0.5
            + features["keyword_count"] * 2.0
            + features["question_count"] * 1.5
        )

    if scores:
        teacher_role = max(scores, key=scores.get)
        student_roles = [role for role in roles if role != teacher_role]
        return teacher_role, scores, student_roles

    return None, {}, []


def count_content_words(text):
    """
    统计实际内容数量（去除标点、空格等无关内容）：
    中文按字计数，其他语言按 Unicode 字母/标记/数字单词计数。
    英文直引号、弯引号和连字符仅在单词内部保留。
    """
    if not text:
        return 0

    normalized = unicodedata.normalize("NFC", str(text))

    def _is_chinese(char):
        return "\u4e00" <= char <= "\u9fff"

    def _is_word_char(char):
        return unicodedata.category(char)[0] in {"L", "M", "N"}

    count = 0
    in_word = False
    connectors = {"'", "’", "-"}

    for index, char in enumerate(normalized):
        if _is_chinese(char):
            count += 1
            in_word = False
            continue

        if _is_word_char(char):
            if not in_word:
                count += 1
            in_word = True
            continue

        if char in connectors and in_word:
            has_word_after = (
                index + 1 < len(normalized)
                and not _is_chinese(normalized[index + 1])
                and _is_word_char(normalized[index + 1])
            )
            if has_word_after:
                continue

        in_word = False

    return count


def calculate_speech_rate(text, start_time, end_time, rate_factor=1.0):
    """计算单句语速（内容单位/分钟）。"""
    try:
        duration = end_time - start_time
        if duration <= 0:
            return 120

        total_words = count_content_words(text)
        if total_words == 0:
            return 0

        speech_rate = total_words / (duration / 60)
        return int(speech_rate * rate_factor)
    except Exception:
        return 120


def build_speed_info(segments, units=(1, 5, 10), total_duration=None):
    """按时间窗口统计内容语速，跨窗口句子按重叠比例分摊。"""
    parsed = []
    max_end = 0.0
    for seg in segments or []:
        try:
            bg = float(seg.get("bg"))
            ed = float(seg.get("ed"))
        except (TypeError, ValueError):
            continue
        if ed <= bg:
            continue
        words = count_content_words(seg.get("segment_text", ""))
        parsed.append((bg, ed, words))
        max_end = max(max_end, ed)

    axis_end = max_end
    if total_duration is not None:
        try:
            axis_end = max(float(total_duration), max_end)
        except (TypeError, ValueError):
            pass

    result = []
    for unit in units:
        win = unit * 60.0
        if axis_end <= 0 or win <= 0:
            result.append({"unit": unit, "segment_info": {"segment_count": 0, "speed": []}})
            continue

        n = int(math.ceil(axis_end / win))
        win_words = [0.0] * n

        for bg, ed, words in parsed:
            duration = ed - bg
            first = int(bg // win)
            last = int((ed - 1e-9) // win)
            for k in range(first, min(last, n - 1) + 1):
                window_start = k * win
                window_end = window_start + win
                overlap = min(ed, window_end) - max(bg, window_start)
                if overlap > 0:
                    win_words[k] += words * (overlap / duration)

        speeds = []
        for k in range(n):
            start = k * win
            nominal = min(win, axis_end - start)
            speeds.append(int(round(win_words[k] / (nominal / 60.0))) if nominal > 0 else 0)

        result.append({
            "unit": unit,
            "segment_info": {
                "segment_count": n,
                "speed": speeds
            }
        })

    return result
