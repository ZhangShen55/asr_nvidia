"""只读聚合真实批、阶段墙钟、增强池计数差值和HBM，不调用模型。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


def pool_delta(current,previous):
    output={}
    for name in ("speech_campplus","emotion2vec"):
        end=current.get(name,{})
        before=previous.get(name) if previous else None
        fields=("calls","wait_total_ms","inference_total_ms","handoff_count","handoff_total_ms")
        delta={key:end.get(key)-before.get(key) for key in fields
               if before and isinstance(end.get(key),(int,float)) and isinstance(before.get(key),(int,float))}
        reset=any(value<0 for value in delta.values())
        if reset:
            delta={}
        calls=delta.get("calls")
        handoffs=delta.get("handoff_count")
        output[name]={"configured":end.get("configured_instances"),"healthy":end.get("instances"),
            "failed_instances":end.get("failed_instances"),"prefetch_depth":end.get("prefetch_queue_depth"),
            "actual_batch_size":end.get("actual_batch_size"),"round_delta":delta,
            "previous_snapshot_present":before is not None,"counter_reset_detected":reset,
            "mean_inference_envelope_ms":delta.get("inference_total_ms",0)/calls if calls else None,
            "mean_pool_wait_ms":delta.get("wait_total_ms",0)/calls if calls else None,
            "mean_handoff_ms":delta.get("handoff_total_ms",0)/handoffs if handoffs else None}
    return output


def envelope(rows):
    starts=[row["start_monotonic_s"] for row in rows if isinstance(row.get("start_monotonic_s"),(int,float))]
    ends=[row["ended_monotonic_s"] for row in rows if isinstance(row.get("ended_monotonic_s"),(int,float))]
    return max(ends)-min(starts) if starts and ends else None


def one_round(data,previous):
    batches=data.get("batches",[])
    assert len({batch.get("batch_id") for batch in batches})==len(batches),"duplicate batch_id"
    distribution=Counter()
    phases=Counter()
    shapes=[]
    for batch in batches:
        runtime=batch.get("runtime_timings",{})
        distribution[str(runtime.get("batch_size",batch.get("actual_dispatched_size",batch.get("size"))))]+=1
        for key,value in runtime.items():
            if key.endswith("_ms") and isinstance(value,(int,float)):
                phases[key]+=value/1000
        shapes.append(runtime.get("processor_shapes"))
    enhanced=data.get("enhanced",{}).get("members",[])
    stage_wall={"qwen_callers_s":envelope(data.get("members",[]))}
    stage_counts={}
    for name in ("speech_campplus","emotion2vec"):
        rows=[member.get(name,{}) for member in enhanced]
        stage_wall[name+"_callers_s"]=envelope(rows)
        stage_counts[name]=dict(Counter(row.get("status","missing") for row in rows))
    resources=data.get("resources",[])
    peak=lambda key:max([row[key] for row in resources if isinstance(row.get(key),(int,float))],default=None)
    proc_hbm=[]
    usage=[]
    for row in resources:
        raw=row.get("npu7_proc-mem",{}).get("stdout","")
        match=re.search(r"Process id:"+str(row.get("pid"))+r"\b[^\n]*Process memory\(MB\):(\d+)",raw)
        if match:
            proc_hbm.append(int(match.group(1)))
        match=re.search(r"HBM Usage Rate\(%\)\s*:\s*(\d+)",row.get("npu7_usages",{}).get("stdout",""))
        if match:
            usage.append(int(match.group(1)))
    return {"status":data.get("status"),"elapsed_s":data.get("elapsed_s"),
        "audio_seconds":data.get("audio_seconds"),"throughput_audio_s_per_s":data.get("audio_seconds_per_wall_second"),
        "N":data.get("total_count"),"success_count":data.get("success_count"),
        "actual_physical_batch_distribution":dict(distribution),"processor_shapes":shapes,
        "qwen_phase_sum_s_once_per_physical_batch":dict(phases),
        "qwen_batch_execution_wall_sum_s":sum(batch.get("execution_wall_ms",0) for batch in batches)/1000,
        "parallel_stage_caller_envelopes":stage_wall,"enhanced_status_counts":stage_counts,
        "pool_round_deltas":pool_delta(data.get("enhanced_pool_status",{}),previous),
        "peak_resources":{"allocator_allocated_bytes":peak("peak_allocator_allocated_bytes"),
            "allocator_reserved_bytes":peak("peak_allocator_reserved_bytes"),"rss_bytes":peak("rss_bytes"),
            "sampled_own_process_hbm_mb":max(proc_hbm) if proc_hbm else None,
            "sampled_device_total_hbm_usage_percent":max(usage) if usage else None}}


def analyze(directory):
    directory=Path(directory)
    grouped=defaultdict(list)
    for path in directory.glob("batch-*-round-*.json"):
        match=re.fullmatch(r"batch-(\d+)-round-(\d+)\.json",path.name)
        if match:
            grouped[int(match[1])].append((int(match[2]),path))
    cases=[]
    for size,files in sorted(grouped.items()):
        previous=None;rounds=[];pending=[]
        for index,path in sorted(files):
            try:
                data=json.loads(path.read_text())
            except json.JSONDecodeError:
                pending.append({"path":str(path),"status":"partially_written"});continue
            if data.get("status")!="completed":
                pending.append({"path":str(path),"status":data.get("status")});continue
            if data.get("warmup") or index==0:
                previous=data.get("enhanced_pool_status",{})
                continue
            row=one_round(data,previous)
            row.update(round=index,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rounds.append(row)
            previous=data.get("enhanced_pool_status",{})
        phase_names={name for row in rounds for name in row["qwen_phase_sum_s_once_per_physical_batch"]}
        cases.append({"configured_batch_size":size,"completed_warm_rounds":len(rounds),"pending":pending,
                      "median_elapsed_s":statistics.median(row["elapsed_s"] for row in rounds) if rounds else None,
                      "median_throughput_audio_s_per_s":statistics.median(row["throughput_audio_s_per_s"] for row in rounds) if rounds else None,
                      "median_qwen_phases_s":{name:statistics.median(row["qwen_phase_sum_s_once_per_physical_batch"].get(name,0) for row in rounds) for name in phase_names},
                      "rounds":rounds})
    return {"directory":str(directory),"cases":cases}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",action="append",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        parser.error("输出已存在，请选新文件名保留历史证据")
    report={"runs":[analyze(directory) for directory in args.run_dir],
            "scope":"CPU JSON recomputation only; independent pure/changed-frontend/changed-scheduler runs share card with uncontrolled external load",
            "limits":["queue waits are cumulative across concurrent callers, not critical-path time",
                "caller envelopes include queue/execution/read/sync; parallel stages must not be added",
                "runtime generate and synchronize are wall envelopes, not kernel time or proof of stream contention",
                "pool deltas subtract preceding completed round including warmup; missing/reset snapshots yield unknown",
                "allocator allocated is included in reserved; do not add; device usage includes other processes"]}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=True,indent=2)+"\n")
    print(json.dumps({"output":str(output),"cases":[{"run":run["directory"],"cases":[{key:value for key,value in case.items() if key!="rounds"} for case in run["cases"]]} for run in report["runs"]]}))


if __name__=="__main__":
    main()
