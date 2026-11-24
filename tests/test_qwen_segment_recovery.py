import pytest
from utils.qwen_segment_recovery import recover_segment_output


def test_preserves_nonloop_prefix_and_one_loop_copy_with_warning():
    raw = "language Chinese<asr_text>今天我们来学习线性代数的基本概念。" + "这个" * 100
    result = recover_segment_output(raw, "repetition_stop", 8.01)
    assert result["raw_text"] == "language Chinese<asr_text>今天我们来学习线性代数的基本概念。这个"
    assert result["warning"] == {"reason": "repetition_stop", "action": "repetition_collapsed", "recovered": True}


@pytest.mark.parametrize("body", ["这个" * 100, "天啊！" + "我操！" * 100,
                                  "哎，对，" + "对，" * 100])
def test_pure_filler_not_promoted_to_transcription(body):
    result = recover_segment_output("language Chinese<asr_text>" + body, "length_limit", 8.01)
    assert result["raw_text"] == "language None<asr_text>"
    assert result["text"] == ""
    assert not result["recovered"]
    assert result["warning"]["action"] == "discard_unrecoverable"


def test_length_limit_retains_only_complete_sentence_not_cut_suffix():
    result = recover_segment_output("language English<asr_text>We study the basic concepts today. Next we will", "length_limit", 8)
    assert result["text"] == "We study the basic concepts today."


def test_missing_prefix_discard_does_not_invent_language():
    result = recover_segment_output("Some words without metadata", "invalid_output_prefix", 8)
    assert result["raw_text"] == "language None<asr_text>"
    assert not result["recovered"]


@pytest.mark.parametrize("reason", ["oom", "cancelled", "timeout", "worker_death", "device_error", "eos"])
def test_other_failures_cannot_enter_recovery(reason):
    with pytest.raises(ValueError):
        recover_segment_output("language Chinese<asr_text>有效内容。", reason, 8)


@pytest.mark.parametrize("duration", [True, False, "8", 0, -1, float("nan"), float("inf")])
def test_recovery_duration_keeps_strict_python_object_validation(duration):
    with pytest.raises(ValueError):
        recover_segment_output("language Chinese<asr_text>有效内容。", "length_limit", duration)


@pytest.mark.parametrize("reason", ["repetition_stop", "length_limit"])
@pytest.mark.parametrize("language,body,expected", [
    ("Chinese", "今天天气" + "很好" * 80, "今天天气很好"),
    ("Chinese", "今天天气很好" * 30, "今天天气很好"),
    ("Chinese", "今天天气" + "很好" * 80 + "很", "今天天气很好"),
    ("Chinese", "先复习，注意，注意，注意。今天天气" + "很好" * 80, "先复习，注意，注意，注意。今天天气很好"),
    ("Chinese", "今天天气" + "很好，" * 80 + "。", "今天天气很好，。"),
    ("English", "hello again " * 80, "hello again"),
    ("English", "again: " + "hello again " * 80 + "hello.", "again: hello again."),
    ("English", "Listen: " + "hello again  " * 80 + "hello.", "Listen: hello again."),
    ("English", "Listen: " + "hello again " * 80 + "hello.", "Listen: hello again."),
    ("Chinese", "12345 " * 40, "12345"),
    ("Chinese", "公式 " + "x=2很好 " * 40, "公式 x=2很好"),
    ("English", "repeat the phrase " * 80, "repeat the phrase"),
    ("English", "Class, very, very, very important. " + "thank you! " * 80, "Class, very, very, very important. thank you!"),
    ("English", "we " + "study energy " * 80 + "study ", "we study energy"),
    ("Chinese", "今天" + "很好" * 40, "今天很好"),
])
def test_confirmed_tail_collapses_once_without_minimum_recovered_length(reason, language, body, expected):
    raw = f"language {language}<asr_text>" + body
    result = recover_segment_output(raw, reason, 2.02)
    assert result["text"] == expected
    assert result["raw_text"] == f"language {language}<asr_text>" + expected
    assert result["warning"] == dict(reason=reason, action="repetition_collapsed", recovered=True)


def test_strong_loop_recovery_does_not_require_general_postprocess_length_gate():
    # 在线强循环可在常规120字符门槛前确认，不能在恢复时再次挡住。
    body = "很好" * 50
    assert len(body) < 120
    assert recover_segment_output("language Chinese<asr_text>" + body, "repetition_stop", 8.01)["text"] == "很好"


@pytest.mark.parametrize("raw", ["language None<asr_text>很好" * 100,
                                "invalid text " * 100,
                                "language Chinese<asr_text>！！！"])
def test_invalid_empty_and_punctuation_outputs_are_not_invented(raw):
    result = recover_segment_output(raw, "repetition_stop", 2)
    assert not result["recovered"]
    assert result["warning"]["action"] == "discard_unrecoverable"


def test_nonloop_length_exhaustion_still_does_not_promote_short_or_unfinished_prefix():
    for body in ["Hi. unfinished", "We study energy without sentence end"]:
        result = recover_segment_output("language English<asr_text>" + body, "length_limit", 2)
        assert result["text"] == ""


@pytest.mark.parametrize("prefix", ["Listen ", "Within ", "plain "])
def test_latin_prefix_suffix_cannot_shift_first_loop_into_middle_of_word(prefix):
    result = recover_segment_output("language English<asr_text>" + prefix + "hello again " * 80 + "hello.",
                                    "repetition_stop", 2)
    assert result["text"] == prefix + "hello again."
