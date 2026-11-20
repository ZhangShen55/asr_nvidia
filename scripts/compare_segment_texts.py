#!/usr/bin/env python3
"""Compare authorized test transcripts by stable segment ID, never by list order.

Inputs: JSON list or {"outputs": [...]} records containing segment_id (preferred)
or index, raw_text/text, optional tokens/finish_reason/source/boundary fields.
Missing and duplicate segments are explicit, not silently aligned or discarded.
"""
from __future__ import annotations
import argparse
import difflib
import html
import json
from pathlib import Path
import re


def records(value):
    rows = value if isinstance(value, list) else value["outputs"]
    result = {}
    for row in rows:
        identity = row.get("segment_id", row.get("index"))
        if identity is None:
            raise ValueError("record requires segment_id or index")
        # request_id is part of identity: index alone is not unique across audio.
        key = json.dumps([row.get("request_id", row.get("source_id")), identity], ensure_ascii=False)
        if key in result:
            raise ValueError(f"duplicate segment identity: {key}")
        result[key] = row
    return result


def raw(row):
    value = row.get("raw_text", row.get("text", ""))
    if not isinstance(value, str):
        raise ValueError("raw_text/text must be string")
    return value


def normalized(value):
    value = value.split("<asr_text>", 1)[-1]
    value = re.sub(r"<\|[^<>]*\|>", "", value)
    return re.sub(r"\s+", "", value)


def compare(left, right):
    a, b = records(left), records(right)
    differences = []
    identical = 0
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            differences.append({"identity": key, "status": "missing_baseline" if x is None else "missing_candidate",
                                "baseline": x, "candidate": y})
            continue
        same = raw(x) == raw(y)
        tokens_same = x.get("tokens") == y.get("tokens")
        reason_same = x.get("finish_reason") == y.get("finish_reason")
        if same and tokens_same and reason_same:
            identical += 1
        else:
            differences.append({"identity": key, "status": "changed", "baseline": x, "candidate": y,
                                "raw_equal": same, "normalized_equal": normalized(raw(x)) == normalized(raw(y)),
                                "tokens_equal": tokens_same, "finish_reason_equal": reason_same})
    return {"baseline_count": len(a), "candidate_count": len(b), "identical": identical,
            "differences_count": len(differences), "differences": differences,
            "note": "相对输出差异，不代表人工真值或准确率；规范化仅剥离ASR前缀/特殊标记/空白，保留大小写、标点及数字。"}


def marked(a, b):
    left, right = [], []
    for op, i, j, u, v in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        x, y = html.escape(a[i:j]), html.escape(b[u:v])
        left.append(x if op == "equal" else "<mark>" + x + "</mark>")
        right.append(y if op == "equal" else "<mark>" + y + "</mark>")
    return "".join(left), "".join(right)


def render(report):
    articles = []
    for item in report["differences"]:
        x, y = item["baseline"], item["candidate"]
        a, b = marked(raw(x) if x else "[缺失]", raw(y) if y else "[缺失]")
        meta = html.escape(json.dumps({"status": item["status"], "baseline": x, "candidate": y}, ensure_ascii=False))
        articles.append(f'<article><h3>{html.escape(item["identity"])}</h3><div class="pair"><pre>{a}</pre><pre>{b}</pre></div><details><summary>来源、边界、原始记录</summary><pre>{meta}</pre></details></article>')
    summary = html.escape(json.dumps({k:v for k,v in report.items() if k != "differences"}, ensure_ascii=False))
    return ('<!doctype html><meta charset="utf-8"><title>逐段文本差异</title><style>body{font:16px sans-serif;margin:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}mark{background:#ffdf96}article{border-top:1px solid #bbb;padding:12px}input{width:90%;padding:8px}</style>'
            '<h1>逐段文本差异</h1><pre>' + summary + '</pre><input id="q" placeholder="搜索来源、文本或段ID">' + ''.join(articles) +
            '<script>document.getElementById("q").addEventListener("input",function(){for(const x of document.querySelectorAll("article"))x.hidden=!x.textContent.toLowerCase().includes(this.value.toLowerCase())})</script>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output basename (.json and .html)")
    args = parser.parse_args()
    report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.with_suffix(".html").write_text(render(report))
    print(json.dumps({k:v for k,v in report.items() if k != "differences"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
