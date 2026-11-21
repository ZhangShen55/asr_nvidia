#!/usr/bin/env python3
"""Compare completed real-v3 responses by sample identity and exact VAD boundaries.

Only model-relative differences, never human-reference accuracy. Missing baseline
responses, censored requests and mismatched source/request settings are explicit
unavailable comparisons rather than zero differences.
"""
from __future__ import annotations
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.compare_segment_texts import marked


def load(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def boundary(row):
    try:
        a, b = Decimal(str(row["bg"])), Decimal(str(row["ed"]))
    except (KeyError, InvalidOperation):
        raise ValueError("missing or invalid bg/ed")
    if not a.is_finite() or not b.is_finite() or a < 0 or b <= a:
        raise ValueError("invalid time interval")
    return str(a.normalize()), str(b.normalize())


def index_segments(rows):
    result = {}
    for row in rows:
        key = boundary(row)
        if key in result:
            raise ValueError("duplicate bg/ed: " + str(key))
        result[key] = row
    return result


def normalized_fields(row):
    # JSON numeric/string casing is not a change in these boolean form fields.
    fields = row.get("fields")
    if fields is None:
        return None
    return {k:str(fields.get(k, False)).lower() for k in ("showSpk", "showEmotion", "showRoleIdentify")}


def read_run(directory):
    directory = Path(directory)
    run = load(directory/"run.json", {})
    summary = load(directory/"summary.json", {})
    samples = {}
    for i, sample in enumerate(run.get("samples", [])):
        sid = sample["sample_id"]
        if sid in samples:
            raise ValueError("duplicate sample_id in run manifest")
        samples[sid] = dict(sample, response_path=str(directory/f"response-{i}.json"))
    records = {x["sample_id"]:x for x in summary.get("records", [])}
    mode = summary.get("mode", run.get("arguments", {}).get("mode"))
    return {"directory":str(directory),"run":run,"summary":summary,"samples":samples,"records":records,"mode":mode}


def completed_response(run, sid):
    record = run["records"].get(sid)
    if not record or record.get("status") != "completed":
        return None, "no_completed_response_in_summary"
    sample = run["samples"][sid]
    payload = load(Path(sample["response_path"]))
    if not isinstance(payload, dict) or payload.get("code") not in (None, 0) or not isinstance(payload.get("segments"), list):
        return None, "missing_or_invalid_response_file"
    if payload.get("processing_status", "complete") not in ("complete", "completed_with_warnings"):
        return None, "response_not_processing_complete"
    return payload, None


def compare(baseline, candidate):
    old, new = read_run(baseline), read_run(candidate)
    output = {"baseline":str(baseline),"candidate":str(candidate),"baseline_mode":old["mode"],"candidate_mode":new["mode"],
        "scope":"Only completed responses in summaries, matched declared source SHA and A/B/form settings, exact sample_id+bg+ed. Model outputs are not human truth; recall>=98% remains unknown.","audios":[]}
    for sid in sorted(old["samples"].keys() | new["samples"].keys()):
        a, b = old["samples"].get(sid), new["samples"].get(sid)
        item = {"sample_id":sid,"comparable":False,"reason":None,"different_segments":None,"changes":[],
                "baseline_record":old["records"].get(sid),"candidate_record":new["records"].get(sid)}
        output["audios"].append(item)
        if a is None or b is None:
            item["reason"] = "source_absent_from_one_manifest"
            continue
        item["baseline_sha256"],item["candidate_sha256"] = a.get("sha256"),b.get("sha256")
        if not a.get("sha256") or a.get("sha256") != b.get("sha256"):
            item["reason"] = "source_sha_mismatch"
            continue
        if old["mode"] is None or old["mode"] != new["mode"]:
            item["reason"] = "A_B_request_mode_mismatch_or_unknown"
            continue
        af, bf = normalized_fields(old["records"].get(sid, {})), normalized_fields(new["records"].get(sid, {}))
        if af is None or bf is None or af != bf:
            item["reason"] = "request_fields_mismatch_or_unknown"
            continue
        x, ex = completed_response(old, sid)
        y, ey = completed_response(new, sid)
        item["baseline_unavailable_reason"],item["candidate_unavailable_reason"] = ex, ey
        # Even if old is unavailable, candidate warnings must not disappear.
        yp = load(Path(b["response_path"]), {})
        item["candidate_segment_warnings"] = yp.get("segment_warnings", []) if isinstance(yp,dict) else []
        if ex or ey:
            item["reason"] = "completed_pair_unavailable"
            continue
        try:
            xs, ys = index_segments(x["segments"]), index_segments(y["segments"])
        except ValueError as exc:
            item["reason"] = str(exc)
            continue
        warnings = {}
        for warning in y.get("segment_warnings", []):
            try:
                warnings.setdefault(boundary(warning), []).append(warning)
            except ValueError:
                pass
        item.update(comparable=True, baseline_segment_count=len(xs), candidate_segment_count=len(ys), identical_segments=0,
                    baseline_response_fields={k:v for k,v in x.items() if k not in ("segments", "text")},
                    candidate_response_fields={k:v for k,v in y.items() if k not in ("segments", "text")})
        for key in sorted(xs.keys() | ys.keys() | warnings.keys(),key=lambda k:(Decimal(k[0]),Decimal(k[1]))):
            left, right = xs.get(key), ys.get(key)
            attached = warnings.get(key, [])
            if left == right and not attached:
                item["identical_segments"] += 1
                continue
            status = ("candidate_missing_with_warning" if right is None and attached else
                      "candidate_missing" if right is None else "baseline_missing" if left is None else
                      "warning_on_matched_segment" if left == right and attached else "changed")
            item["changes"].append({"sample_id":sid,"bg":key[0],"ed":key[1],"status":status,
                "baseline":left,"candidate":right,"candidate_warnings":attached,
                "text_equal":bool(left is not None and right is not None and left.get("segment_text")==right.get("segment_text"))})
        item["different_segments"] = len(item["changes"])
        item["matched_text_changed_segments"] = sum(x["baseline"] is not None and x["candidate"] is not None and not x["text_equal"] for x in item["changes"])
        item["missing_candidate_segments"] = sum(x["candidate"] is None for x in item["changes"])
        item["missing_baseline_segments"] = sum(x["baseline"] is None for x in item["changes"])
    output["comparable_audio_count"] = sum(x["comparable"] for x in output["audios"])
    output["unavailable_audio_count"] = sum(not x["comparable"] for x in output["audios"])
    output["different_segments_in_comparable_pairs"] = sum(x["different_segments"] or 0 for x in output["audios"])
    return output


def render(report):
    sections=[]
    for item in report["audios"]:
        meta={k:v for k,v in item.items() if k != "changes"}
        parts=[f'<section><h2>{html.escape(item["sample_id"])}</h2><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre>']
        for change in item["changes"]:
            a,b=change["baseline"],change["candidate"]
            left,right=marked(a.get("segment_text","") if a else "[无对应旧分段]",b.get("segment_text","") if b else "[无对应新分段]")
            parts.append(f'<article><h3>{html.escape(change["bg"])}–{html.escape(change["ed"])} {html.escape(change["status"])}</h3><div class="pair"><pre>{left}</pre><pre>{right}</pre></div><details><summary>完整old/new字段与warning</summary><pre>{html.escape(json.dumps(change,ensure_ascii=False,indent=2))}</pre></details></article>')
        sections.append(''.join(parts)+'</section>')
    return ('<!doctype html><meta charset="utf-8"><title>完整音频v3逐段对照</title><style>body{font:16px sans-serif;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}mark{background:#ffdf96}article,section{border-top:1px solid #aaa;padding:12px}</style>'
            '<h1>完整音频v3逐段对照</h1><p>'+html.escape(report['scope'])+'</p><input id="q" placeholder="搜索段文本/时间/warning">'+''.join(sections)+
            '<script>document.getElementById("q").addEventListener("input",function(){for(const x of document.querySelectorAll("article"))x.hidden=!x.textContent.toLowerCase().includes(this.value.toLowerCase())})</script>')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline",type=Path);p.add_argument("candidate",type=Path);p.add_argument("--output",required=True,type=Path)
    a=p.parse_args();result=compare(a.baseline,a.candidate);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    a.output.with_suffix('.html').write_text(render(result))
    print(json.dumps({k:result[k] for k in ['comparable_audio_count','unavailable_audio_count','different_segments_in_comparable_pairs']},ensure_ascii=False))


if __name__=='__main__':main()
