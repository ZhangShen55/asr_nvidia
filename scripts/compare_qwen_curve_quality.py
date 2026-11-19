"""从既有Qwen批曲线JSON复算质量差异与实际shape；只读，不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_only(text):
    # 该层只区分标点/空白差异；数字、字母和其它符号不被强制归一。
    return "".join(ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P"))


def index_members(data):
    result={}
    for member in data.get("members",[]):
        identity=member.get("id") or (str(member["request_id"])+":"+str(member["segment_index"]))
        if identity in result:
            raise ValueError("重复成员ID: "+identity)
        result[identity]=member
    return result


def make_normalizer(config_path=None, *, language_policy="parsed", skip=False):
    from utils.itn import ItnConfig, normalize_segments
    options={}
    if config_path:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        options=tomllib.loads(Path(config_path).read_text()).get("itn",{})
    config=ItnConfig(**{key:options[key] for key in asdict(ItnConfig()) if key in options})
    cache={}
    def normalize(text, language):
        selected=("en" if (language or "").lower() in ("english","en") else "zh"
                  if (language or "").lower() in ("chinese","zh","cantonese","yue") else "unsupported") if language_policy=="parsed" else language_policy
        key=(text,selected)
        if key not in cache:
            if skip:
                cache[key]=(None,{"skipped":"explicit_skip_itn","language":selected})
            else:
                segments,_,stats=normalize_segments([{"segment_text":text}],selected,config)
                # 不把初始化失败后的passthrough伪装成ITN成功，保留逐项failure证据。
                cache[key]=(segments[0]["segment_text"],{k:v for k,v in stats.items() if k!="duration_ms"})
        return cache[key]
    return normalize,{"config":asdict(config),"source":str(config_path) if config_path else "ItnConfig defaults",
                      "source_sha256":sha256(config_path) if config_path else None,
                      "language_policy":language_policy,"skipped":skip,
                      "scope":"parsed segment text -> ITN only; whole-audio dominant-language selection, repetition repair and final sanitization are not simulated"}


def member_value(member, normalize):
    from utils.qwen_response import parse_qwen_text
    if member.get("status")!="completed" or not isinstance(member.get("raw"),str):
        return {"status":member.get("status"),"available":False,"raw":member.get("raw"),"error":member.get("error")}
    parsed=parse_qwen_text(member["raw"])
    normalized,stats=normalize(parsed.text,parsed.language)
    return {"status":"completed","available":True,"raw":member["raw"],
            "parsed":{"text":parsed.text,"language":parsed.language},
            "recorded_parsed":member.get("parsed"),
            "recorded_parsed_matches":member.get("parsed")=={"text":parsed.text,"language":parsed.language},
            "content":content_only(parsed.text),"itn_text":normalized,"itn_stats":stats}


def batch_evidence(data):
    batches=data.get("batches",[])
    identifiers=[batch.get("batch_id") for batch in batches]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError("一个轮次重复batch_id，不能重复累计物理批包络")
    evidence=[]
    for batch in batches:
        timing=batch.get("runtime_timings",{})
        singles=timing.get("single_processor_shapes",[])
        histogram=Counter(json.dumps(shape,sort_keys=True) for shape in singles)
        evidence.append({"batch_id":batch.get("batch_id"),"selected_size":batch.get("selected_size",batch.get("size")),
            "actual_dispatched_size":batch.get("actual_dispatched_size",batch.get("size")),
            "effective_runtime_size":timing.get("batch_size"),"frontend_mode":timing.get("frontend_mode"),
            "processor_shapes":timing.get("processor_shapes"),
            "single_processor_shape_histogram":[{"shapes":json.loads(shapes),"count":count} for shapes,count in histogram.items()],
            "generated_padded_steps":timing.get("generated_padded_tokens"),
            "reason":batch.get("reason"),"execution_wall_ms":batch.get("execution_wall_ms"),
            "member_identity":[[row.get("request_id"),row.get("segment_index")] for row in batch.get("dispatched_members",batch.get("members",[]))],
            "phases_ms":{key:value for key,value in timing.items() if key.endswith("_ms") and isinstance(value,(int,float))}})
    return evidence


def compare_round(data, baseline, normalize):
    members=index_members(data)
    details=[]
    counts=Counter()
    ids=sorted(set(baseline)|set(members))
    for identity in ids:
        before=baseline.get(identity)
        after=member_value(members[identity],normalize) if identity in members else None
        row={"id":identity,"missing_baseline":before is None,"missing_candidate":after is None}
        if before is None or after is None or not before["available"] or not after["available"]:
            row.update(comparable=False,baseline=before,candidate=after)
            counts["uncomparable"]+=1
        else:
            flags={"raw_changed":before["raw"]!=after["raw"],
                   "parsed_text_changed":before["parsed"]["text"]!=after["parsed"]["text"],
                   "language_changed":before["parsed"]["language"]!=after["parsed"]["language"],
                   "content_changed":before["content"]!=after["content"],
                   "itn_changed":None if before["itn_text"] is None or after["itn_text"] is None else before["itn_text"]!=after["itn_text"]}
            row.update(comparable=True,**flags)
            counts["comparable"]+=1
            for key,value in flags.items():
                if value is True:
                    counts[key]+=1
            counts["recorded_parsed_mismatch"]+=not after["recorded_parsed_matches"]
            counts["itn_failures"]+=int(after["itn_stats"].get("failures",0))
            if any(value is True for value in flags.values()) or not after["recorded_parsed_matches"] or after["itn_stats"].get("failures",0):
                row.update(baseline=before,candidate=after)
        details.append(row)
    batches=batch_evidence(data)
    distribution=Counter(str(row["actual_dispatched_size"]) for row in batches)
    phase_sums=Counter()
    for batch in batches:
        for name,value in batch["phases_ms"].items():
            phase_sums[name]+=value/1000
    return {"source_status":data.get("status"),"N_candidate":len(members),"N_baseline":len(baseline),
            "identity_sets_equal":set(members)==set(baseline),"counts":dict(counts),"members":details,
            "elapsed_s":data.get("elapsed_s"),"audio_seconds_per_wall_second":data.get("audio_seconds_per_wall_second"),
            "member_p50_s":data.get("p50_s"),"member_p95_s":data.get("p95_s"),
            "actual_batch_distribution":dict(distribution),"batches":batches,
            "phase_sum_s_once_per_physical_batch":dict(phase_sums),
            "empty_text_ids":[mid for mid,row in members.items() if row.get("status")=="completed" and not member_value(row,normalize).get("parsed",{}).get("text","").strip()]}


def analyze(baseline_path, directories, normalize):
    baseline_path=Path(baseline_path)
    baseline_data=json.loads(baseline_path.read_text())
    if baseline_data.get("status")!="completed" or baseline_data.get("warmup"):
        raise ValueError("标量基线必须为完成的非预热轮次")
    if any(batch.get("actual_dispatched_size",batch.get("size"))!=1 for batch in baseline_data.get("batches",[])):
        raise ValueError("标量基线存在实际非单段批，不能冒充scalar")
    baseline={identity:member_value(member,normalize) for identity,member in index_members(baseline_data).items()}
    if not baseline or not all(row["available"] for row in baseline.values()):
        raise ValueError("基线存在缺失/失败raw，不能作完整质量对照")
    runs=[]
    for directory in directories:
        directory=Path(directory)
        rounds=[]; skipped=[]; stability=defaultdict(lambda:defaultdict(set))
        for path in sorted(directory.glob("batch-*-round-*.json")):
            match=re.fullmatch(r"batch-(\d+)-round-(\d+)\.json",path.name)
            if not match:
                continue
            size,index=map(int,match.groups())
            data=json.loads(path.read_text())
            if data.get("warmup") or index==0:
                skipped.append({"path":str(path),"reason":"warmup","sha256":sha256(path)})
                continue
            comparison=compare_round(data,baseline,normalize)
            comparison.update(path=str(path),sha256=sha256(path),configured_batch_size=size,round=index)
            rounds.append(comparison)
            for identity,member in index_members(data).items():
                row=member_value(member,normalize)
                if row["available"]:
                    stability[size][identity].add(json.dumps(row["parsed"],sort_keys=True,ensure_ascii=True))
        cases=[]
        if not rounds:
            raise ValueError("目录中没有非预热曲线证据: "+str(directory))
        for size in sorted({row["configured_batch_size"] for row in rounds}):
            selected=[row for row in rounds if row["configured_batch_size"]==size]
            successful=[row for row in selected if row["source_status"]=="completed" and row["identity_sets_equal"] and not row["counts"].get("uncomparable")]
            cases.append({"configured_batch_size":size,"rounds":len(selected),"complete_rounds":len(successful),
                          "parsed_varying_ids":[identity for identity,values in stability[size].items() if len(values)>1],
                          "median_elapsed_s":statistics.median(row["elapsed_s"] for row in successful) if successful else None,
                          "median_throughput":statistics.median(row["audio_seconds_per_wall_second"] for row in successful) if successful else None,
                          "round_differences":[{"round":row["round"],"counts":row["counts"],"actual_batch_distribution":row["actual_batch_distribution"]} for row in selected]})
        runs.append({"directory":str(directory),"cases":cases,"rounds":rounds,"excluded_warmup":skipped})
    return {"scope":"read-only raw/parsed/ITN differences on existing curve JSON; no model/NPU execution, no accuracy or WER claim",
            "baseline":{"path":str(baseline_path),"sha256":sha256(baseline_path),"N":len(baseline),"members":baseline},
            "runs":runs,"limitations":["shape evidence is captured runtime metadata, not inferred from configured batch size",
            "wall phase envelopes overlap/shared synchronization; do not sum as kernel occupancy",
            "same outputs do not prove correctness; text differences have no winner without human labels",
            "ITN is a separate per-segment transform, not complete v3 output simulation"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--candidate-dir",action="append",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--config")
    parser.add_argument("--itn-language",choices=("parsed","auto","zh","en"),default="parsed")
    parser.add_argument("--skip-itn",action="store_true")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        parser.error("输出已存在，保留历史证据；请选新文件名")
    normalize,policy=make_normalizer(args.config,language_policy=args.itn_language,skip=args.skip_itn)
    result=analyze(args.baseline,args.candidate_dir,normalize)
    result["itn_policy"]=policy
    result["analyzer_sha256"]=sha256(__file__)
    result["parser_sha256"]=sha256(ROOT/"utils/qwen_response.py")
    result["itn_source_sha256"]=sha256(ROOT/"utils/itn.py")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=True,indent=2)+"\n")
    print(json.dumps({"output":str(output),"runs":[{"directory":run["directory"],"cases":run["cases"]} for run in result["runs"]]},ensure_ascii=False))


if __name__=="__main__":
    main()
