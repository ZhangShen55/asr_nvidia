"""只读审计八完整音频矩阵、计时、资源收尾及同源最终文本波动。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


REQUIRED_RUNTIME_MODULES=(
    "main","model_status_runtime","telemetry_runtime","entity/data",
    "core/config","core/qwen_asr_client","core/qwen_asr_runtime","core/qwen_batch_scheduler",
    "core/qwen_asr_queue","core/models","core/local_inference","core/model_loading",
    "core/vad","core/vad_pool","core/segment_pipeline","core/status","core/request_telemetry",
    "api/routes/asr_v3","api/routes/asr_qwen_common","api/routes/asr_common","api/routes/status",
    "utils/audio_utils","utils/qwen_response","utils/itn",
)


def load(path):
    return json.loads(Path(path).read_text())


def percentile(values,q):
    ordered=sorted(values)
    if not ordered:
        return None
    position=(len(ordered)-1)*q
    lower=int(position);upper=min(lower+1,len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower)


def close(a,b):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-6)


def utc(value):
    parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("UTC字段缺少时区")
    return parsed


def config_caps(path):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    data=tomllib.loads(Path(path).read_text()).get("qwen_asr",{})
    return {"total":data.get("audio_concurrency",8),"b":data.get("audio_spk_emo_concurrency",4)}


def config_identity(path):
    """Compare full parsed configs, excluding only the explicit B-cap profile knob."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    data = tomllib.loads(Path(path).read_text())
    qwen = data.get("qwen_asr", {})
    b_capacity = qwen.pop("audio_spk_emo_concurrency", 4)
    canonical = lambda value: json.dumps(value, sort_keys=True, default=str).encode()
    field_hashes = {}

    def visit(value, prefix):
        if isinstance(value, dict) and value:
            for key, child in value.items():
                visit(child, prefix + [key])
        else:
            field_hashes[".".join(prefix)] = hashlib.sha256(canonical(value)).hexdigest()

    visit(data, [])
    return {
        "normalized_fingerprint": hashlib.sha256(canonical(data)).hexdigest(),
        "excluded_fields": ["qwen_asr.audio_spk_emo_concurrency"],
        "b_capacity": b_capacity,
        "field_hashes": field_hashes,
    }


def matrix_consistency(selected, expected_count=18):
    """Only uniquely selected formal cases qualify; old/retried cases cannot vote."""
    source_groups = defaultdict(list)
    config_groups = defaultdict(list)
    source_missing = []
    config_missing = []
    configs = []
    warm_paths = []
    for row in selected:
        source = row.get("source_identity_gate", {})
        if source.get("passed") and source.get("runtime_source_fingerprint"):
            source_groups[source["runtime_source_fingerprint"]].append(row["path"])
        else:
            source_missing.append(row["path"])
        config = row.get("config_identity")
        if config and config.get("normalized_fingerprint"):
            config_groups[config["normalized_fingerprint"]].append(row["path"])
            configs.append(config)
        else:
            config_missing.append(row["path"])
        if row.get("warmup_qualification", {}).get("passed"):
            warm_paths.append(row["path"])

    def gate(groups, missing):
        status = "failed" if len(groups) > 1 else "pending" if missing or len(selected) != expected_count else "passed"
        return {"status": status, "fingerprint_groups": dict(groups), "missing_identity_paths": missing}

    source_gate = gate(source_groups, source_missing)
    config_gate = gate(config_groups, config_missing)
    keys = set().union(*(set(item["field_hashes"]) for item in configs)) if configs else set()
    changed_fields = sorted(key for key in keys if len({item["field_hashes"].get(key) for item in configs}) > 1)
    config_gate.update(
        allowed_different_fields=["qwen_asr.audio_spk_emo_concurrency"],
        observed_b_capacities=sorted({item["b_capacity"] for item in configs}),
        differing_other_fields=changed_fields,
    )
    warm_gate = {
        "status": "passed" if len(warm_paths) == expected_count and len(selected) == expected_count else "pending",
        "qualified_rounds": len(warm_paths),
        "scope": "independent short-clip model warmup completed; all-instance forward and all-shape steady state are unproven",
    }
    states = [source_gate["status"], config_gate["status"], warm_gate["status"]]
    return {
        "status": "failed" if "failed" in states else "passed" if all(state == "passed" for state in states) else "pending",
        "selected_rounds": len(selected), "required_rounds": expected_count,
        "runtime_source_gate": source_gate, "configuration_gate": config_gate, "warmup_gate": warm_gate,
    }


def client_peak(records):
    events=[]
    for row in records:
        if isinstance(row.get("started_monotonic"),(int,float)) and isinstance(row.get("ended_monotonic"),(int,float)):
            events.extend([(row["started_monotonic"],1),(row["ended_monotonic"],-1)])
    current=peak=0
    for _,change in sorted(events,key=lambda item:(item[0],item[1])):
        current+=change;peak=max(peak,current)
    return peak


def timeline(payload):
    errors=[];segments=[];previous=-1
    if not isinstance(payload.get("segments"),list):
        return [],["segments_type"]
    for index,segment in enumerate(payload.get("segments",[])):
        if not isinstance(segment,dict):
            errors.append(f"segment_{index}_type");continue
        bg,ed=segment.get("bg"),segment.get("ed")
        if not all(isinstance(value,str) and re.fullmatch(r"\d+\.\d{2}",value) for value in (bg,ed)):
            errors.append(f"segment_{index}_time_serialization");continue
        if float(bg)<previous or float(ed)<float(bg):
            errors.append(f"segment_{index}_time_order")
        previous=float(bg)
        if not isinstance(segment.get("segment_text"),str):
            errors.append(f"segment_{index}_text_type")
        segments.append({"bg":bg,"ed":ed,"final_text_after_itn":segment.get("segment_text"),
            "role":segment.get("role"),"emotion":segment.get("emotion"),
            "segment_language_observed":segment.get("language"),
            "pre_itn_or_qwen_raw":None})
    return segments,errors


def source_identity_gate(run,summary):
    """开始/结束实际文件散列是权威证据，不能以HEAD或缺省false替代。"""
    missing=[]
    before=run.get("source_identity") or {}
    after=summary.get("source_identity_after") or {}
    def hashes(identity,when):
        files=identity.get("files")
        if identity.get("complete") is not True or not isinstance(files,dict):
            missing.append(f"source_identity_{when}_missing_or_incomplete")
            return {}
        if identity.get("file_count")!=len(files):
            missing.append(f"source_identity_{when}_runtime_file_count_inconsistent")
        result={}
        for name,value in files.items():
            relative=Path(name)
            allowed=(len(relative.parts)>1 and relative.parts[0] in ("core","api","utils","entity")) or (len(relative.parts)==1 and name.split(".")[0] in ("main","model_status_runtime","telemetry_runtime"))
            digest=value.get("sha256") if isinstance(value,dict) else None
            if relative.is_absolute() or ".." in relative.parts or relative.suffix not in (".py",".so") or not allowed or not isinstance(digest,str) or not re.fullmatch("[0-9a-f]{64}",digest):
                missing.append(f"source_identity_{when}_invalid_runtime_hash_entry")
                continue
            result[name]=digest
        if "main.py" not in result:
            missing.append(f"source_identity_{when}_main_entry_hash_missing")
        modules={str(Path(name).parent/Path(name).name.split(".")[0]) for name in result}
        for module in REQUIRED_RUNTIME_MODULES:
            if module not in modules:
                missing.append(f"source_identity_{when}_critical_module_missing:{module}")
        for key in ("root","python_executable","python_version"):
            if not isinstance(identity.get(key),str) or not identity[key]:
                missing.append(f"source_identity_{when}_{key}_missing")
        return result
    initial=hashes(before,"before")
    final=hashes(after,"after")
    if summary.get("source_changed_during_round") is not False or run.get("source_changed_during_round") is not False:
        missing.append("source_change_state_not_explicitly_false")
    if summary.get("source_identity_complete") is not True or run.get("source_identity_complete") is not True:
        missing.append("source_identity_round_not_complete")
    differences=sorted(name for name in set(initial)|set(final) if initial.get(name)!=final.get(name))
    if differences:
        missing.append("runtime_hashes_changed_during_round")
    if summary.get("source_changes") or run.get("source_changes"):
        missing.append("recorded_source_changes_during_round")
    if any(before.get(key)!=after.get(key) for key in ("root","python_executable","python_version")):
        missing.append("runtime_identity_metadata_changed_during_round")
    # run和summary均保存结束身份；不接受只改summary里false标志掩盖run原始证据。
    run_after=run.get("source_identity_after")
    if not isinstance(run_after,dict) or run_after!=after:
        missing.append("run_and_summary_source_identity_after_disagree")
    fingerprint=hashlib.sha256(json.dumps(initial,sort_keys=True).encode()).hexdigest() if initial else None
    return {"passed":not missing,"required_runtime_modules":list(REQUIRED_RUNTIME_MODULES),
            "before_file_count":len(initial),"after_file_count":len(final),"changed_runtime_files":differences,
            "runtime_source_fingerprint":fingerprint,"git_head_before":before.get("git_head"),"git_head_after":after.get("git_head"),
            "git_head_status_before":before.get("git_head_status","unknown"),"missing_evidence":sorted(set(missing)),
            "comparison_scope":"recorded start/end runtime hashes; no reconstruction for legacy rounds or continuous file-watching claim"}


def audit_round(path):
    path=Path(path);errors=[];missing=[]
    summary=load(path/"summary.json")
    records=summary.get("records",[])
    mode=summary.get("mode")
    concurrency=summary.get("concurrency")
    recomputed={};source_rows={}
    if len(records)!=8 or summary.get("expected")!=8:
        errors.append("expected_exactly_eight_full_records")
    hashes=[row.get("sha256") for row in records]
    if len(set(hashes))!=8 or any(not isinstance(sha,str) or not re.fullmatch("[0-9a-f]{64}",sha) for sha in hashes):
        errors.append("eight_unique_sha256_required")
    if not summary.get("complete") or summary.get("status")!="completed" or summary.get("successful")!=8:
        missing.append("not_all_eight_successfully_completed")
    start,end=summary.get("batch_started_monotonic"),summary.get("batch_ended_monotonic")
    if not close(end-start if isinstance(start,(int,float)) and isinstance(end,(int,float)) else None,summary.get("batch_elapsed_s")):
        errors.append("batch_elapsed_mismatch")
    try:
        if utc(summary["batch_ended_utc"])<utc(summary["batch_started_utc"]):
            errors.append("batch_utc_order")
    except (KeyError,ValueError,TypeError):
        missing.append("batch_utc_missing_or_invalid")
    values=[];total_seconds=0
    for index,row in enumerate(records):
        if row.get("status")!="completed":
            continue
        before,after=row.get("started_monotonic"),row.get("ended_monotonic")
        if not close(after-before if isinstance(before,(int,float)) and isinstance(after,(int,float)) else None,row.get("elapsed_s")):
            errors.append(f"request_{index}_elapsed_mismatch")
        if isinstance(before,(int,float)) and isinstance(after,(int,float)):
            if after<before or (isinstance(start,(int,float)) and before<start) or (isinstance(end,(int,float)) and after>end):
                errors.append(f"request_{index}_monotonic_order")
        try:
            if not utc(summary["batch_started_utc"])<=utc(row["started_utc"])<=utc(row["ended_utc"])<=utc(summary["batch_ended_utc"]):
                errors.append(f"request_{index}_utc_order")
        except (KeyError,ValueError,TypeError):
            missing.append(f"request_{index}_utc_missing_or_invalid")
        elapsed=row.get("elapsed_s")
        if isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and elapsed>=0:
            values.append(elapsed)
            if elapsed>420:
                errors.append(f"request_{index}_over420_without_diagnostic_acceptance")
        duration=row.get("audio_seconds")
        if not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
            errors.append(f"request_{index}_invalid_source_duration")
        else:
            total_seconds+=duration
        fields=row.get("fields",{})
        enhanced=index<4 if mode=="mixed" else index>=4 if mode=="mixed-swap" else mode=="B"
        if any(fields.get(name)!=str(enhanced).lower() for name in ("showSpk","showEmotion")) or fields.get("showRoleIdentify")!="false":
            errors.append(f"request_{index}_capability_flags")
        response_file=path/f"response-{index}.json"
        if not response_file.exists():
            missing.append(f"response_{index}_missing");continue
        payload=load(response_file)
        if payload.get("code") not in (None,0):
            errors.append(f"response_{index}_business_error")
        for name,kind in (("gpu_time_ms",str),("load_audio_time_ms",str),("language_ratio",dict),("segments",list)):
            if not isinstance(payload.get(name),kind):
                errors.append(f"response_{index}_{name}_type")
        segments,issues=timeline(payload)
        errors.extend(f"response_{index}_{issue}" for issue in issues)
        source_rows[row.get("sha256") or f"missing-{index}"]={"sample_id":row.get("sample_id"),"audio_seconds":duration,
            "elapsed_s":elapsed,"capabilities":fields,"segments":segments,"language_ratio_observed":payload.get("language_ratio"),
            "response_path":str(response_file),"response_sha256":hashlib.sha256(response_file.read_bytes()).hexdigest()}
    if len(values)==8 and len(records)==8:
        recomputed.update(n=8,p50_s=percentile(values,.5),p95_s=percentile(values,.95),source_audio_seconds=total_seconds)
        latency=summary.get("success_subset_latency",{})
        if latency.get("n")!=8 or not close(latency.get("p50_s"),recomputed["p50_s"]) or not close(latency.get("p95_s"),recomputed["p95_s"]):
            errors.append("linear_quantile_mismatch")
        if not close(summary.get("source_audio_seconds"),total_seconds):
            errors.append("source_duration_sum_mismatch")
        if isinstance(start,(int,float)) and isinstance(end,(int,float)) and end>start:
            recomputed["throughput_audio_s_per_s"]=total_seconds/(end-start)
            if not close(summary.get("source_audio_seconds_per_wall_second"),recomputed["throughput_audio_s_per_s"]):
                errors.append("throughput_mismatch")
    else:
        recomputed.update(n=len(values),p50_s=None,p95_s=None,reason="N_not_8; no acceptance quantiles synthesized")
    peak=client_peak(records)
    if concurrency not in (1,2,4,8) or peak>concurrency:
        errors.append("client_concurrency_violation")
    elif peak!=concurrency:
        errors.append("configured_client_concurrency_not_reached")
    caps=None;config_source=None;observed=[]
    if (path/"config.toml").exists():
        caps=config_caps(path/"config.toml")
        config_source=config_identity(path/"config.toml")
        expected_b=8 if concurrency==8 and mode in ("A","B") else 4
        if caps!={"total":8,"b":expected_b}:
            errors.append("wrong_total_or_B_profile")
    else:
        missing.append("config_snapshot_missing")
    if (path/"status.jsonl").exists():
        for line in (path/"status.jsonl").read_text().splitlines():
            if not line.strip():continue
            try:
                state=json.loads(line).get("status",{})
                total=state["audioConcurrency"]["active_total"];b=state["spkEmoConcurrency"]["active"]
                observed.append({"total":total,"b":b})
                if total>8 or b>total or b<0 or total<0 or (caps and b>caps["b"]):
                    errors.append("observed_server_capacity_violation")
            except (json.JSONDecodeError,KeyError,TypeError):
                continue
    if not observed:
        missing.append("server_activity_peak_not_observed")
    elif concurrency in (1,2,4,8) and max(row["total"] for row in observed)<concurrency:
        missing.append("sampled_server_peak_below_target; sending_requests_does_not_prove_admission")
    shutdown=summary.get("shutdown")
    if not shutdown and (path/"shutdown.json").exists():
        shutdown=load(path/"shutdown.json")
    if not shutdown:
        missing.append("owned_shutdown_missing")
    elif not (summary.get("service_ownership")=="owned" and shutdown.get("passed") and shutdown.get("exit_code")==0
              and not shutdown.get("forced") and not shutdown.get("orphan_cleanup") and not shutdown.get("remaining_pids")
              and shutdown.get("observe_s",0)>=5 and len(shutdown.get("port_observations",[]))>=2
              and not any(shutdown["port_observations"])):
        errors.append("owned_shutdown_not_normal")
    warm=summary.get("warmup")
    if not warm and (path/"warmup.json").exists():
        warm=load(path/"warmup.json")
    if not warm or not warm.get("complete") or warm.get("status")!="completed" or summary.get("temperature")!="warmup_completed":
        missing.append("completed_warmup_required_for_warm_acceptance")
    elif not isinstance(start,(int,float)) or warm.get("ended_monotonic",float("inf"))>start:
        errors.append("warmup_overlaps_formal_batch")
    warm_qualified = bool(warm and warm.get("complete") and warm.get("status") == "completed"
                          and summary.get("temperature") == "warmup_completed"
                          and isinstance(start, (int, float)) and warm.get("ended_monotonic", float("inf")) <= start)
    run=load(path/"run.json") if (path/"run.json").exists() else {}
    source_gate=source_identity_gate(run,summary)
    missing.extend(source_gate["missing_evidence"])
    if summary.get("diagnostic_trigger") or summary.get("status")=="diagnostic_aborted":
        state="diagnostic"
    elif errors:
        state="failed"
    elif missing:
        state="pending"
    else:
        state="passed"
    return {"path":str(path),"mode":mode,"concurrency":concurrency,"started_utc":summary.get("batch_started_utc"),
        "audit_status":state,"errors":sorted(set(errors)),"missing_evidence":sorted(set(missing)),
        "recomputed":recomputed,"client_peak":peak,"sampled_server_total_peak":max((row["total"] for row in observed),default=None),
        "sampled_server_B_peak":max((row["b"] for row in observed),default=None),"capacity_profile":caps,"sources":source_rows,
        "source_sha256":hashes,
        "source_identity_gate":source_gate,
        "config_identity":config_source,
        "warmup_qualification":{"passed":warm_qualified,"temperature":summary.get("temperature"),
            "evidence_kind":warm.get("kind") if warm else None,
            "scope":"independent warmup request only; no all-instance forward or all-shape steady-state claim"},
        "input_sha256":{name:hashlib.sha256((path/name).read_bytes()).hexdigest()
                        for name in ("summary.json","config.toml","run.json","shutdown.json","warmup.json","status.jsonl") if (path/name).is_file()}}


def segment_map(segments):
    rows=defaultdict(list)
    for segment in segments:
        rows[(segment["bg"],segment["ed"])].append(segment)
    return rows


def compare_sources(left,right):
    files=[];changed=missing=0
    for sha in sorted(set(left["sources"])|set(right["sources"])):
        before=left["sources"].get(sha);after=right["sources"].get(sha)
        if before is None or after is None:
            files.append({"sha256":sha,"missing_left":before is None,"missing_right":after is None,"segments":[]});missing+=1;continue
        a=segment_map(before["segments"]);b=segment_map(after["segments"]);differences=[]
        for boundary in sorted(set(a)|set(b),key=lambda pair:(float(pair[0]),float(pair[1]))):
            for occurrence in range(max(len(a[boundary]),len(b[boundary]))):
                item_a=a[boundary][occurrence] if occurrence<len(a[boundary]) else None
                item_b=b[boundary][occurrence] if occurrence<len(b[boundary]) else None
                if item_a is None or item_b is None:
                    missing+=1
                    differences.append({"bg":boundary[0],"ed":boundary[1],"occurrence":occurrence,
                        "missing_left":item_a is None,"missing_right":item_b is None,"left":item_a,"right":item_b,
                        "reason":"final response missing; raw/pre-ITN unavailable so filter cause unknown"})
                elif item_a["final_text_after_itn"]!=item_b["final_text_after_itn"]:
                    changed+=1
                    differences.append({"bg":boundary[0],"ed":boundary[1],"occurrence":occurrence,
                                        "left":item_a,"right":item_b,"final_text_changed":True})
        files.append({"sha256":sha,"sample_id":before["sample_id"],"same_source_duration":close(before["audio_seconds"],after["audio_seconds"]),
                      "left_segments":len(before["segments"]),"right_segments":len(after["segments"]),"segment_differences":differences,
                      "left_capabilities":before.get("capabilities"),"right_capabilities":after.get("capabilities"),
                      "left_language_ratio":before["language_ratio_observed"],"right_language_ratio":after["language_ratio_observed"]})
    return {"left":left["path"],"right":right["path"],"same_eight_source_hashes":len(left["sources"])==8 and set(left["sources"])==set(right["sources"]),
            "changed_final_text_segments":changed,"missing_final_segments_or_sources":missing,"files":files}


def audit(roots,case_map=None):
    rounds=[];seen=set()
    for root in roots:
        for summary in Path(root).rglob("summary.json"):
            if summary.resolve() in seen:
                continue
            seen.add(summary.resolve())
            data=load(summary)
            if "records" in data and "batch_started_monotonic" in data:
                rounds.append(audit_round(summary.parent))
    # 显式case-map可绑定重试/分拆driver输出；否则按每档每mode的UTC顺序配对。
    overrides={str(Path(item["path"]).resolve()):item["label"] for item in (case_map or {}).get("cases",[])}
    grouped=defaultdict(list)
    for row in rounds:
        grouped[(row["concurrency"],row["mode"])].append(row)
    for (_,mode),rows in grouped.items():
        for index,row in enumerate(sorted(rows,key=lambda r:(r["started_utc"] or "",r["path"]))):
            match=re.search(r"(?:^|[-_/])(A[12]|B[12])(?:$|[-_/])",row["path"])
            row["label"]=overrides.get(str(Path(row["path"]).resolve())) or (match[1] if match else mode+str(index+1) if mode in ("A","B") else mode)
    reference=next((set(row["source_sha256"]) for row in rounds if len(set(row["source_sha256"]))==8),None)
    for row in rounds:
        row["same_matrix_eight_sources"]=reference is not None and set(row["source_sha256"])==reference
        if reference is not None and not row["same_matrix_eight_sources"]:
            row["errors"].append("source_set_differs_from_other_matrix_rounds")
            if row["audit_status"]!="diagnostic":
                row["audit_status"]="failed"
    expected=[(c,label) for c in (1,2,4,8) for label in ("A1","B1","B2","A2")]+[(8,"mixed"),(8,"mixed-swap")]
    lookup=defaultdict(list)
    for row in rounds:
        lookup[(row["concurrency"],row["label"])].append(row)
    coverage=[{"concurrency":c,"label":label,"status":"pending" if not lookup[(c,label)] else "ambiguous" if len(lookup[(c,label)])>1 else lookup[(c,label)][0]["audit_status"],
               "paths":[r["path"] for r in lookup[(c,label)]]} for c,label in expected]
    selected_rounds = [lookup[(concurrency, label)][0] for concurrency, label in expected
                       if len(lookup[(concurrency, label)]) == 1]
    consistency = matrix_consistency(selected_rounds, len(expected))
    comparisons=[];gates=[]
    for concurrency in (1,2,4,8):
        selected={label:lookup[(concurrency,label)][0] for label in ("A1","A2","B1","B2") if len(lookup[(concurrency,label)])==1}
        if len(selected)!=4 or any(row["audit_status"]!="passed" for row in selected.values()):
            gates.append({"concurrency":concurrency,"status":"pending","reason":"requires four complete audited warm rounds"});continue
        a_values=[source["elapsed_s"] for label in ("A1","A2") for source in selected[label]["sources"].values()]
        b_values=[source["elapsed_s"] for label in ("B1","B2") for source in selected[label]["sources"].values()]
        overhead=percentile(b_values,.5)/percentile(a_values,.5)-1
        self_variation=compare_sources(selected["A1"],selected["A2"])
        cross=[compare_sources(selected[a],selected[b]) for a in ("A1","A2") for b in ("B1","B2")]
        comparisons.append({"concurrency":concurrency,"A_self":self_variation,"A_B_pairs":cross})
        sources_equal=all(row["same_eight_source_hashes"] for row in [self_variation,*cross])
        text_gate=sources_equal and all(row["changed_final_text_segments"]<=self_variation["changed_final_text_segments"]
                                       and row["missing_final_segments_or_sources"]<=self_variation["missing_final_segments_or_sources"] for row in cross)
        gates.append({"concurrency":concurrency,"status":"passed" if text_gate and (concurrency!=1 or overhead<=.3+1e-12) else "failed",
            "same_eight_sources":sources_equal,"A_N":len(a_values),"B_N":len(b_values),"A_p50_s":percentile(a_values,.5),"B_p50_s":percentile(b_values,.5),
            "B_p50_overhead_fraction":overhead,"overhead_gate_enforced":concurrency==1,"overhead_limit":.3 if concurrency==1 else None,
            "overhead_within_30_percent":overhead<=.3+1e-12,"text_variation_gate":text_gate,
            "A_self_changed_text_segments":self_variation["changed_final_text_segments"],"max_A_B_changed_text_segments":max(row["changed_final_text_segments"] for row in cross)})
    mixed=None
    if all(len(lookup[(8,label)])==1 and lookup[(8,label)][0]["audit_status"]=="passed" for label in ("mixed","mixed-swap")):
        mixed=compare_sources(lookup[(8,"mixed")][0],lookup[(8,"mixed-swap")][0])
    passed=all(row["status"]=="passed" for row in coverage) and all(row["status"]=="passed" for row in gates) and consistency["status"]=="passed"
    return {"complete":passed,"status":"passed" if passed else "incomplete", "coverage":coverage,"gates":gates,"rounds":rounds,"comparisons":comparisons,
            "matrix_reference_sha256":sorted(reference) if reference else None,"mixed_swap_comparison":mixed,"matrix_consistency":consistency,
            "limits":["Only final post-ITN response text is exposed; raw Qwen/pre-ITN and per-segment language are unavailable and not reconstructed",
                      "Language ratio is whole-response observation, not per-segment language or accuracy",
                      "30% overhead is mandatory for concurrency1; higher concurrency overhead is measured without inventing a new gate",
                      "A-self vs A/B final-text variation preserves missing/filtered segments; no human labels so no accuracy claim",
                      "Server peaks are sampled observations; sending eight requests alone does not prove eight admitted",
                      "All selected matrix rounds require identical runtime hashes and configuration except the B4/B8 admission cap",
                      "warmup_completed qualifies a short independent model-warmup request, not all-instance forward or all-shape steady state"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",action="append",required=True)
    parser.add_argument("--case-map")
    parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():
        parser.error("输出已存在，使用新路径保留原始证据")
    result=audit(args.input_dir,load(args.case_map) if args.case_map else None)
    result["auditor_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=True,indent=2)+"\n")
    print(json.dumps({"output":str(output),"complete":result["complete"],"coverage":result["coverage"],"gates":result["gates"]}))


if __name__=="__main__":
    main()
