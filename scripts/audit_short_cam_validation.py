#!/usr/bin/env python3
"""审计短CAM验证JSONL覆盖与数值证据；不导入torch、不加载模型、不访问NPU。

实机报告由validate_npu_models.py生成。本工具只检查报告内部证据的一致性，
不能代替硬件执行所有权日志或人工说话人准确率验收。
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

EXPECTED_DURATIONS_S = (0.3, 0.999, 1.001)


def audit(records, durations=EXPECTED_DURATIONS_S):
    issues = []
    def check(condition, message):
        if not condition:
            issues.append(message)
    def vector(output):
        values = output.get('values') if isinstance(output, dict) else None
        return (isinstance(values, list) and len(values) == 192 and
                all(type(value) in (int, float) and math.isfinite(value) for value in values) and
                sum(value * value for value in values) > 0)

    plans = [row for row in records if row.get('type') == 'plan']
    check(len(plans) == 1 and records[0].get('type') == 'plan', 'one leading plan record required')
    tolerance = plans[0].get('tolerances', {}).get('fp32', {}) if plans else {}
    check(tolerance.get('cam_cosine_min') == 0.9999 and tolerance.get('cam_max_abs') == 0.001,
          'predeclared FP32 tolerance must be cosine>=0.9999 and max_abs<=0.001')
    check(bool(plans and plans[0].get('device') == 'npu:7'), 'plan must target physical npu:7')
    check(any(row.get('type') == 'environment' and row.get('device') == 'npu:7' and row.get('torch_npu')
              for row in records), 'actual run environment record missing')
    clips = [row for row in records if row.get('type') == 'clip']
    ids = [row.get('clip_id') for row in clips]
    check(bool(clips) and len(ids) == len(set(ids)) and all(ids), 'clip IDs must be nonempty and unique')
    expected_samples = {round(duration * 16000) for duration in durations}
    by_source = {}
    for clip in clips:
        source = clip.get('sample_id')
        by_source.setdefault(source, []).append(clip)
        samples = round(float(clip.get('duration_s', -1)) * 16000)
        check(clip.get('sample_rate') == 16000 and clip.get('frames') == samples,
              f'{clip.get("clip_id")}: invalid effective sample count/rate')
        check(isinstance(clip.get('pcm_sha256'), str) and len(clip['pcm_sha256']) == 64,
              f'{clip.get("clip_id")}: PCM SHA256 missing')
    for source, source_clips in by_source.items():
        samples = {round(float(clip.get('duration_s', -1)) * 16000) for clip in source_clips}
        check(expected_samples <= samples, f'{source}: required short-duration coverage missing')
        check(any(clip.get('source_sha256_declared') and
                  clip.get('source_sha256_verified') == clip.get('source_sha256_declared')
                  for clip in source_clips), f'{source}: verified source SHA256 missing')
    references = [row for row in records if row.get('type') == 'reference' and row.get('model') == 'speech_campplus']
    ref_by_id = {row.get('clip_id'): row for row in references}
    check(set(ids) == set(ref_by_id) and len(references) == len(ids), 'CPU FP32 reference coverage mismatch')
    for key, row in ref_by_id.items():
        check(vector(row.get('output')), f'{key}: reference must be finite nonzero 192D')
    candidates = [row for row in records if row.get('type') == 'candidate' and
                  row.get('model') == 'speech_campplus' and row.get('precision') == 'fp32' and
                  row.get('compiler') == 'eager']
    check({row.get('clip_id') for row in candidates} == set(ids), 'NPU eager FP32 coverage mismatch')
    for key in ids:
        check(any(row.get('clip_id') == key and row.get('phase') == 'steady' for row in candidates),
              f'{key}: no steady NPU result')
    for row in candidates:
        key = row.get('clip_id')
        comparison = row.get('comparison') or {}
        check(vector(row.get('output')), f'{key}: candidate must be finite nonzero 192D')
        cosine, maximum = comparison.get('cosine'), comparison.get('max_abs')
        check(comparison.get('passed') is True and comparison.get('valid') is True and
              type(cosine) in (int, float) and math.isfinite(cosine) and cosine >= 0.9999 and
              type(maximum) in (int, float) and math.isfinite(maximum) and maximum <= 0.001,
              f'{key}: FP32 numerical gate failed or missing')
    clusters = [row for row in records if row.get('type') == 'cluster' and row.get('precision') == 'fp32'
                and row.get('compiler') == 'eager']
    check(len(clusters) == 1, 'one eager FP32 clustering comparison required')
    expected_indices = {str(index) for index in range(len(clips))}
    for row in clusters:
        a, b = row.get('reference_labels') or {}, row.get('candidate_labels') or {}
        check({str(key) for key in a} == expected_indices and {str(key) for key in b} == expected_indices,
              'clustering must include all clips, including 0.3~<1s embeddings')
        def partition(labels):
            return sorted(sorted(str(index) for index, label in labels.items() if label == value)
                          for value in set(labels.values()))
        check(row.get('same_partition') is True and partition(a) == partition(b), 'CPU/NPU clustering partition mismatch')
    check(any(row.get('type') == 'candidate_summary' and row.get('model') == 'speech_campplus' and
              row.get('precision') == 'fp32' and row.get('compiler') == 'eager' and row.get('passed') is True
              for row in records), 'successful CAM candidate summary missing')
    check(bool(records and records[-1].get('type') == 'complete' and records[-1].get('baseline_passed') is True),
          'successful complete marker missing')
    return {'passed': not issues, 'issues': issues, 'clips': len(clips), 'references': len(references),
            'candidate_results': len(candidates), 'required_durations_s': list(durations),
            'scope': 'CPU/NPU FP32 report consistency and short-embedding clustering coverage; no human accuracy claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        records = [json.loads(line) for line in args.report.read_text().splitlines() if line.strip()]
        result = audit(records)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result = {'passed': False, 'issues': [f'{type(exc).__name__}: {exc}']}
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(text)
    print(text, end='')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
