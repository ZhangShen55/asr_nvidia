import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("compare_qwen_curve_quality",Path(__file__).resolve().parents[1]/"scripts/compare_qwen_curve_quality.py")
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def member(identity,text,status="completed"):
    return {"id":identity,"status":status,"raw":"language Chinese<asr_text>"+text,
            "parsed":{"text":text,"language":"Chinese"}}


def normalize(text,language):
    return text.replace("四","4"),{"failures":0}


def round_data(members,actual=1):
    return {"status":"completed","warmup":False,"members":members,"elapsed_s":2,
            "audio_seconds_per_wall_second":3,"p50_s":1,"p95_s":2,
            "batches":[{"batch_id":5,"size":actual,"actual_dispatched_size":actual,
                        "runtime_timings":{"batch_size":actual,"processor_shapes":{"input_features":[actual,128,800]},"generate_wall_ms":100},
                        "dispatched_members":[{"request_id":"r","segment_index":0}]}]}


def test_raw_parsed_content_and_itn_are_independent():
    baseline={"x":tool.member_value(member("x","四"),normalize)}
    result=tool.compare_round(round_data([member("x","4")],actual=16),baseline,normalize)
    assert result["counts"]["raw_changed"]==1
    assert result["counts"]["parsed_text_changed"]==1
    assert result["counts"]["content_changed"]==1
    assert not result["counts"].get("itn_changed")
    assert result["actual_batch_distribution"]=={"16":1}
    assert result["batches"][0]["processor_shapes"]["input_features"]==[16,128,800]
    assert result["phase_sum_s_once_per_physical_batch"]["generate_wall_ms"]==.1


def test_punctuation_and_whitespace_not_lexical_change():
    baseline={"x":tool.member_value(member("x","四，"),normalize)}
    result=tool.compare_round(round_data([member("x","四 ")]),baseline,normalize)
    assert result["counts"]["parsed_text_changed"]==1
    assert not result["counts"].get("content_changed")


def test_missing_and_failed_members_are_not_dropped():
    baseline={"a":tool.member_value(member("a","甲"),normalize),"b":tool.member_value(member("b","乙"),normalize)}
    result=tool.compare_round(round_data([member("a","甲",status="failed"),member("c","丙")]),baseline,normalize)
    assert result["N_baseline"]==2 and result["N_candidate"]==2
    assert result["counts"]["uncomparable"]==3 and not result["identity_sets_equal"]
    assert len(result["members"])==3


def test_parser_is_recomputed_not_trusting_recorded_parsed():
    candidate=member("a","甲")
    candidate["parsed"]["text"]="错误的记录"
    baseline={"a":tool.member_value(member("a","甲"),normalize)}
    result=tool.compare_round(round_data([candidate]),baseline,normalize)
    assert not result["counts"].get("parsed_text_changed")
    assert result["counts"]["recorded_parsed_mismatch"]==1


def test_duplicate_batch_and_member_evidence_rejected():
    data=round_data([member("a","甲")])
    data["members"].append(data["members"][0])
    with pytest.raises(ValueError):
        tool.index_members(data)
    data["batches"].append(data["batches"][0])
    with pytest.raises(ValueError):
        tool.batch_evidence(data)


def test_multiple_directories_rounds_and_warmup_exclusion(tmp_path):
    baseline=tmp_path/"scalar.json"
    baseline.write_text(json.dumps(round_data([member("a","甲")])))
    v1=tmp_path/"v1";v1.mkdir()
    v2=tmp_path/"v2";v2.mkdir()
    for directory in (v1,v2):
        warm=round_data([member("a","warm")]);warm["warmup"]=True
        (directory/"batch-128-round-0.json").write_text(json.dumps(warm))
        for index in (1,2):
            (directory/f"batch-128-round-{index}.json").write_text(json.dumps(round_data([member("a","甲" if directory==v2 else "乙")],actual=16)))
    result=tool.analyze(baseline,[v1,v2],normalize)
    assert len(result["runs"])==2
    assert all(len(run["rounds"])==2 and len(run["excluded_warmup"])==1 for run in result["runs"])
    assert result["runs"][0]["cases"][0]["round_differences"][0]["counts"]["parsed_text_changed"]==1
    assert not result["runs"][1]["cases"][0]["round_differences"][0]["counts"].get("parsed_text_changed")


def test_itn_skip_is_explicit_and_does_not_claim_equivalence():
    norm,policy=tool.make_normalizer(skip=True)
    assert norm("四","Chinese")[0] is None
    assert policy["skipped"]
    baseline={"a":tool.member_value(member("a","四"),norm)}
    result=tool.compare_round(round_data([member("a","4")]),baseline,norm)
    assert result["members"][0]["itn_changed"] is None
