import importlib.util
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location("compare_segment_texts", Path(__file__).parents[1]/"scripts/compare_segment_texts.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def test_identity_not_list_order_unicode_and_reason():
    a = [{"segment_id": 1, "source_id": "音频一", "raw_text": "你好", "tokens": [1]},
         {"segment_id": 1, "source_id": "音频二", "raw_text": "你你你"}]
    b = [dict(a[1], finish_reason="repetition_stop"), a[0]]
    report = m.compare(a, b)
    assert report["identical"] == 1
    assert report["differences_count"] == 1
    assert report["differences"][0]["raw_equal"]
    assert not report["differences"][0]["finish_reason_equal"]


def test_missing_not_discarded_and_duplicate_not_overwritten():
    a = [{"index": 0, "text": "A"}]
    report = m.compare(a, [{"index": 1, "text": "B"}])
    assert len(report["differences"]) == 2
    with pytest.raises(ValueError, match="duplicate"):
        m.compare(a + a, [])


def test_asr_normalization_does_not_remove_case_or_punctuation():
    report = m.compare([{"index": 0, "text": "language English<asr_text>A.<|im_end|>"}],
                       [{"index": 0, "text": "A."}])
    assert report["differences"][0]["normalized_equal"]
    assert m.normalized("A.") != m.normalized("a")


def test_html_escapes_untrusted_transcription():
    report = m.compare([{"index": 0, "text": "<img src=x onerror=bad()>"}], [])
    page = m.render(report)
    assert "<img" not in page
    assert "&lt;img" in page
