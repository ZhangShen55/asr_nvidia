import pytest
from utils.qwen_segment_recovery import recover_segment_output


def test_preserves_complete_nonloop_model_prefix_with_warning():
    raw = "language Chinese<asr_text>今天我们来学习线性代数的基本概念。" + "这个" * 100
    result = recover_segment_output(raw, "repetition_stop", 8.01)
    assert result["raw_text"] == "language Chinese<asr_text>今天我们来学习线性代数的基本概念。"
    assert result["warning"] == {"reason": "repetition_stop", "action": "recover_prefix", "recovered": True}


@pytest.mark.parametrize("body", ["这个" * 100, "天啊！" + "我操！" * 100,
                                  "哎，对，" + "对，" * 100, "repeat the phrase " * 100])
def test_pure_loop_or_filler_not_promoted_to_credible(body):
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
