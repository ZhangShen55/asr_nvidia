import importlib.util
import json
from pathlib import Path

_spec=importlib.util.spec_from_file_location("compare_v3",Path(__file__).parents[1]/"scripts/compare_v3_transcripts.py")
m=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(m)
FIELDS={"showSpk":"false","showEmotion":"false","showRoleIdentify":"false"}


def make(root,samples=None,records=None,responses=None,mode="A"):
    root.mkdir()
    samples=samples or [{"sample_id":"a1","sha256":"abc"}]
    records=records or [{"sample_id":"a1","status":"completed","fields":FIELDS}]
    (root/"run.json").write_text(json.dumps({"samples":samples,"arguments":{"mode":mode}}))
    (root/"summary.json").write_text(json.dumps({"mode":mode,"records":records}))
    for i,response in (responses or {}).items():
        (root/f"response-{i}.json").write_text(json.dumps(response))
    return root


def response(text="你好",warnings=None):
    value={"segments":[{"bg":"1.00","ed":"2.00","segment_text":text,"role":None}],"text":text}
    if warnings is not None:
        value.update(processing_status="completed_with_warnings",segment_warnings=warnings)
    return value


def test_pair_by_boundaries_preserves_unicode_and_all_segment_fields(tmp_path):
    a=make(tmp_path/"a",responses={0:response("旧文字")})
    b=make(tmp_path/"b",responses={0:response("新文字")})
    result=m.compare(a,b);item=result["audios"][0]
    assert item["comparable"] and item["different_segments"]==1
    assert item["changes"][0]["baseline"]["role"] is None
    assert item["changes"][0]["candidate"]["segment_text"]=="新文字"
    assert '<mark>' in m.render(result)


def test_missing_completed_old_is_unknown_not_zero_and_retains_new_warning(tmp_path):
    a=make(tmp_path/"a",records=[{"sample_id":"a1","status":"cancelled","fields":FIELDS}])
    warning={"bg":"1.00","ed":"2.00","reason":"repetition_stop","action":"discard_unrecoverable"}
    b=make(tmp_path/"b",responses={0:response("保留",[warning])})
    item=m.compare(a,b)["audios"][0]
    assert not item["comparable"] and item["different_segments"] is None
    assert item["candidate_segment_warnings"]==[warning]


def test_unrecoverable_missing_segment_is_explicit(tmp_path):
    a=make(tmp_path/"a",responses={0:response()})
    warning={"bg":"1.00","ed":"2.00","action":"discard_unrecoverable"}
    b=make(tmp_path/"b",responses={0:{"segments":[],"processing_status":"completed_with_warnings","segment_warnings":[warning]}})
    change=m.compare(a,b)["audios"][0]["changes"][0]
    assert change["status"]=="candidate_missing_with_warning"
    assert change["candidate"] is None


def test_sha_or_A_B_mismatch_not_compared(tmp_path):
    a=make(tmp_path/"a",responses={0:response()})
    b=make(tmp_path/"b",samples=[{"sample_id":"a1","sha256":"different"}],responses={0:response()})
    c=make(tmp_path/"c",mode="B",responses={0:response()})
    assert m.compare(a,b)["audios"][0]["reason"]=="source_sha_mismatch"
    assert not m.compare(a,c)["audios"][0]["comparable"]


def test_response_alignment_is_sample_id_not_order(tmp_path):
    samples=[{"sample_id":"a1","sha256":"one"},{"sample_id":"a2","sha256":"two"}]
    records=[{"sample_id":x["sample_id"],"status":"completed","fields":FIELDS} for x in samples]
    a=make(tmp_path/"a",samples,records,{0:response("one"),1:response("two")})
    b=make(tmp_path/"b",samples[::-1],records[::-1],{0:response("two"),1:response("one")})
    result=m.compare(a,b)
    assert result["comparable_audio_count"]==2
    assert result["different_segments_in_comparable_pairs"]==0


def test_duplicate_boundaries_unavailable_and_html_escaped(tmp_path):
    value=response("<script>bad()</script>");value["segments"]*=2
    a=make(tmp_path/"a",responses={0:value})
    b=make(tmp_path/"b",responses={0:response()})
    assert not m.compare(a,b)["audios"][0]["comparable"]
    value["segments"]=value["segments"][:1];(a/"response-0.json").write_text(json.dumps(value))
    page=m.render(m.compare(a,b))
    assert '<script>bad()' not in page
    assert '&lt;script&gt;' in page
