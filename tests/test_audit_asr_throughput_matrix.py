import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location("audit_asr_throughput_matrix",Path(__file__).resolve().parents[1]/"scripts/audit_asr_throughput_matrix.py")
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def write(path,data):
    path.write_text(json.dumps(data))


def make_round(root,concurrency,label,text="原始文本",elapsed=10):
    mode=label[0] if label[0] in "AB" else label
    path=root/f"c{concurrency}-{label}";path.mkdir()
    records=[]
    for i in range(8):
        start=100+(i//concurrency)*30+(i%concurrency)*.01
        enhanced=(mode=="B" or mode=="mixed" and i<4 or mode=="mixed-swap" and i>=4)
        row={"sample_id":str(i),"sha256":f"{i:064x}","audio_seconds":60,"status":"completed","elapsed_s":elapsed,
             "started_monotonic":start,"ended_monotonic":start+elapsed,
             "started_utc":"2026-11-18T00:00:01+00:00","ended_utc":"2026-11-18T00:00:10+00:00",
             "fields":{"showSpk":str(enhanced).lower(),"showEmotion":str(enhanced).lower(),"showRoleIdentify":"false"}}
        records.append(row)
        write(path/f"response-{i}.json",{"gpu_time_ms":"100","load_audio_time_ms":"12","language_ratio":{"zh":100},
             "segments":[{"bg":"0.00","ed":"1.00","segment_text":text,"role":None,"emotion":None}]})
    end=max(row["ended_monotonic"] for row in records)
    shutdown={"passed":True,"owned":True,"exit_code":0,"forced":False,"orphan_cleanup":False,"remaining_pids":[],"observe_s":5,"port_observations":[False,False]}
    warm={"status":"completed","complete":True,"ended_monotonic":90}
    summary={"status":"completed","complete":True,"expected":8,"successful":8,"mode":mode,"concurrency":concurrency,
             "batch_started_monotonic":100,"batch_ended_monotonic":end,"batch_elapsed_s":end-100,
             "batch_started_utc":"2026-11-18T00:00:00+00:00","batch_ended_utc":"2026-11-18T00:01:00+00:00",
             "records":records,"success_subset_latency":{"n":8,"p50_s":elapsed,"p95_s":elapsed},
             "source_audio_seconds":480,"source_audio_seconds_per_wall_second":480/(end-100),
             "service_ownership":"owned","temperature":"warmup_completed","shutdown":shutdown,"warmup":warm}
    names=[module+".py" for module in tool.REQUIRED_RUNTIME_MODULES]
    identity={"root":"/fixture/runtime","python_executable":"/fixture/python","python_version":"3.10.fixture","complete":True,"file_count":len(names),
              "git_head":None,"git_head_status":"unknown","files":{name:{"sha256":hashlib.sha256(name.encode()).hexdigest()} for name in names}}
    source_fields={"source_changed_during_round":False,"source_changes":[],"source_identity_complete":True,"source_identity_after":identity}
    summary.update(source_fields)
    write(path/"run.json",dict(source_fields,source_identity=identity))
    write(path/"summary.json",summary)
    write(path/"shutdown.json",shutdown)
    write(path/"warmup.json",warm)
    capacity=8 if concurrency==8 and mode in ("A","B") else 4
    (path/"config.toml").write_text(f"[qwen_asr]\naudio_concurrency=8\naudio_spk_emo_concurrency={capacity}\n")
    b=min(concurrency,4) if mode.startswith("mixed") else concurrency if mode=="B" else 0
    write(path/"status.jsonl",{"status":{"audioConcurrency":{"active_total":concurrency},"spkEmoConcurrency":{"active":b}}})
    return path


def test_complete_round_recomputes_quantiles_and_final_contract(tmp_path):
    path=make_round(tmp_path,4,"A1")
    report=tool.audit_round(path)
    assert report["audit_status"]=="passed" and report["recomputed"]["n"]==8
    assert report["client_peak"]==4 and report["sampled_server_total_peak"]==4
    assert report["sources"]["0"*64]["segments"][0]["pre_itn_or_qwen_raw"] is None


def test_incomplete_n_cannot_create_n8_quantiles(tmp_path):
    path=make_round(tmp_path,1,"A1")
    summary=tool.load(path/"summary.json")
    summary["records"]=summary["records"][:7]
    summary.update(status="diagnostic_aborted",complete=False,diagnostic_trigger={"reason":"420"})
    write(path/"summary.json",summary)
    result=tool.audit_round(path)
    assert result["audit_status"]=="diagnostic"
    assert result["recomputed"]["n"]==7 and result["recomputed"]["p50_s"] is None


@pytest.mark.parametrize("field,value,expected",[("gpu_time_ms",100,"gpu_time_ms_type"),("segments",[{"bg":"2.00","ed":"1.00","segment_text":"x"}],"time_order")])
def test_timing_serialization_and_order_fail(tmp_path,field,value,expected):
    path=make_round(tmp_path,1,"A1")
    payload=tool.load(path/"response-0.json");payload[field]=value;write(path/"response-0.json",payload)
    report=tool.audit_round(path)
    assert report["audit_status"]=="failed" and any(expected in error for error in report["errors"])


def test_shutdown_exit1_cannot_be_washed_by_passed_flag(tmp_path):
    path=make_round(tmp_path,1,"A1")
    data=tool.load(path/"summary.json");data["shutdown"]["exit_code"]=1;write(path/"summary.json",data)
    assert "owned_shutdown_not_normal" in tool.audit_round(path)["errors"]


def test_warmup_is_outside_formal_eight(tmp_path):
    path=make_round(tmp_path,1,"A1")
    data=tool.load(path/"summary.json");data["warmup"]["ended_monotonic"]=101;write(path/"summary.json",data)
    assert "warmup_overlaps_formal_batch" in tool.audit_round(path)["errors"]


def test_same_sha_boundary_alignment_preserves_missing_and_text(tmp_path):
    a=make_round(tmp_path,1,"A1");b=make_round(tmp_path,1,"B1",text="变化文本")
    payload=tool.load(b/"response-0.json");payload["segments"]=[];write(b/"response-0.json",payload)
    result=tool.compare_sources(tool.audit_round(a),tool.audit_round(b))
    assert result["same_eight_source_hashes"]
    assert result["changed_final_text_segments"]==7 and result["missing_final_segments_or_sources"]==1
    assert "unknown" in result["files"][0]["segment_differences"][0]["reason"]


def test_gate_requires_all_warm_pairs_and_applies_30percent_only_single(tmp_path):
    for concurrency in (1,2):
        for label in ("A1","B1","B2","A2"):
            make_round(tmp_path,concurrency,label,elapsed=14 if label.startswith("B") else 10)
    report=tool.audit([tmp_path])
    single=next(g for g in report["gates"] if g["concurrency"]==1)
    double=next(g for g in report["gates"] if g["concurrency"]==2)
    assert single["status"]=="failed" and single["B_p50_overhead_fraction"]==pytest.approx(.4)
    assert double["status"]=="passed" and not double["overhead_gate_enforced"]
    assert not report["complete"]


def test_all_18_rounds_prove_complete_and_no_duplicate_roots(tmp_path):
    for concurrency in (1,2,4,8):
        for label in ("A1","B1","B2","A2"):
            make_round(tmp_path,concurrency,label,elapsed=12 if label.startswith("B") else 10)
    make_round(tmp_path,8,"mixed");make_round(tmp_path,8,"mixed-swap")
    report=tool.audit([tmp_path,tmp_path])
    assert report["complete"] and len(report["rounds"])==18
    assert len(report["coverage"])==18
    assert report["mixed_swap_comparison"]["same_eight_source_hashes"]


def test_cross_concurrency_must_use_same_eight_source_files(tmp_path):
    make_round(tmp_path,1,"A1")
    path=make_round(tmp_path,2,"A1")
    data=tool.load(path/"summary.json")
    for index,row in enumerate(data["records"]):
        row["sha256"]=f"{index+100:064x}"
    write(path/"summary.json",data)
    result=tool.audit([tmp_path])
    assert any("source_set_differs_from_other_matrix_rounds" in row["errors"] for row in result["rounds"])


def test_legacy_round_missing_source_identity_cannot_pass(tmp_path):
    path=make_round(tmp_path,1,"A1")
    write(path/"run.json",{"arguments":{"mode":"A"}})
    summary=tool.load(path/"summary.json")
    for key in ("source_changed_during_round","source_changes","source_identity_complete","source_identity_after"):
        summary.pop(key)
    write(path/"summary.json",summary)
    result=tool.audit_round(path)
    assert result["audit_status"]=="pending" and not result["source_identity_gate"]["passed"]
    assert "source_identity_before_missing_or_incomplete" in result["missing_evidence"]


def test_hot_change_or_lying_false_flag_is_insufficient_evidence(tmp_path):
    path=make_round(tmp_path,1,"A1")
    summary=tool.load(path/"summary.json")
    summary["source_identity_after"]["files"]["main.py"]["sha256"]="f"*64
    write(path/"summary.json",summary)
    result=tool.audit_round(path)
    assert result["audit_status"]=="pending" and result["source_identity_gate"]["changed_runtime_files"]==["main.py"]
    assert "runtime_hashes_changed_during_round" in result["missing_evidence"]


def test_source_trace_rejects_incomplete_hash_count_but_allows_non_git_stage(tmp_path):
    path=make_round(tmp_path,1,"A1")
    summary=tool.load(path/"summary.json");run=tool.load(path/"run.json")
    assert tool.source_identity_gate(run,summary)["passed"]
    run["source_identity"]["file_count"]-=1
    assert not tool.source_identity_gate(run,summary)["passed"]


def test_protected_stage_so_covers_critical_modules_without_fixed_file_count(tmp_path):
    path=make_round(tmp_path,1,"A1")
    summary=tool.load(path/"summary.json");run=tool.load(path/"run.json")
    identity=run["source_identity"]
    files={name if name=="main.py" else name[:-3]+".cpython-310-aarch64-linux-gnu.so":value for name,value in identity["files"].items()}
    files["core/optional_new_runtime.so"]={"sha256":"a"*64}
    identity.update(files=files,file_count=len(files))
    run["source_identity_after"]=identity
    summary["source_identity_after"]=identity
    result=tool.source_identity_gate(run,summary)
    assert result["passed"]
    assert result["before_file_count"]==len(tool.REQUIRED_RUNTIME_MODULES)+1
    assert "minimum_runtime_files" not in result
    removed=files.pop("core/qwen_asr_runtime.cpython-310-aarch64-linux-gnu.so")
    identity["file_count"]=len(files)
    result=tool.source_identity_gate(run,summary)
    assert not result["passed"]
    assert any("critical_module_missing:core/qwen_asr_runtime" in reason for reason in result["missing_evidence"])


def complete_matrix(root):
    for concurrency in (1, 2, 4, 8):
        for label in ("A1", "B1", "B2", "A2"):
            make_round(root, concurrency, label, elapsed=12 if label.startswith("B") else 10)
    make_round(root, 8, "mixed")
    make_round(root, 8, "mixed-swap")


def test_each_round_valid_but_cross_matrix_runtime_change_cannot_pass(tmp_path):
    complete_matrix(tmp_path)
    path = tmp_path / "c4-B2"
    summary = tool.load(path / "summary.json")
    run = tool.load(path / "run.json")
    for identity in (run["source_identity"], run["source_identity_after"], summary["source_identity_after"]):
        identity["files"]["api/routes/asr_qwen_common.py"]["sha256"] = "e" * 64
    write(path / "run.json", run)
    write(path / "summary.json", summary)
    assert tool.audit_round(path)["audit_status"] == "passed"
    report = tool.audit([tmp_path])
    gate = report["matrix_consistency"]["runtime_source_gate"]
    assert gate["status"] == "failed" and len(gate["fingerprint_groups"]) == 2
    assert all(row["status"] == "passed" for row in report["coverage"])
    assert not report["complete"]


def test_b4_b8_only_config_variation_passes_but_other_setting_change_fails(tmp_path):
    complete_matrix(tmp_path)
    report = tool.audit([tmp_path])
    assert report["complete"]
    gate = report["matrix_consistency"]["configuration_gate"]
    assert gate["status"] == "passed" and gate["observed_b_capacities"] == [4, 8]
    assert not gate["differing_other_fields"]
    path = tmp_path / "c8-B2" / "config.toml"
    path.write_text(path.read_text() + "segment_concurrency=8\n")
    assert tool.audit_round(path.parent)["audit_status"] == "passed"
    report = tool.audit([tmp_path])
    gate = report["matrix_consistency"]["configuration_gate"]
    assert gate["status"] == "failed"
    assert "qwen_asr.segment_concurrency" in gate["differing_other_fields"]
    assert not report["complete"]


def test_cold_round_is_not_a_warm_matrix_or_all_shape_steady_state_claim(tmp_path):
    complete_matrix(tmp_path)
    report = tool.audit([tmp_path])
    assert report["matrix_consistency"]["warmup_gate"]["qualified_rounds"] == 18
    assert "unproven" in report["matrix_consistency"]["warmup_gate"]["scope"]
    path = tmp_path / "c1-A1"
    summary = tool.load(path / "summary.json")
    summary["temperature"] = "unknown/cold_possible"
    write(path / "summary.json", summary)
    report = tool.audit([tmp_path])
    assert not report["complete"]
    assert report["matrix_consistency"]["warmup_gate"]["qualified_rounds"] == 17
    assert report["matrix_consistency"]["warmup_gate"]["status"] == "pending"
    assert next(row for row in report["rounds"] if row["path"] == str(path))["audit_status"] == "pending"
