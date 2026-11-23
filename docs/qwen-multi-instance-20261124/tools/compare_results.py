#!/usr/bin/env python3
"""Compare complete real ASR artifacts without importing application/model code.

Only declared matching source SHA and summary status=completed are comparable.
Unavailable/censored pairs remain unknown, never zero differences or accuracy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
import hashlib
import html
import json
from pathlib import Path
import re


BASELINE = Path('/home/zhangs/asr_nvidia/logs/qwen-200s-feasibility-20261123/m6n6-nohbm-formal')
SCOPE = ('仅比较summary中status=completed的完整响应与声明源SHA。模型间相对一致/不一致不是人工准确率、'
         '召回率或WER/CER；censored、缺失和无法精确对齐者为unknown。性能是否通过由原summary独立判断。')


def read_json(path):
    try:
        # Keep JSON fractional tokens exact before Decimal boundary matching.
        return json.loads(path.read_text(encoding='utf-8'), parse_float=Decimal), None
    except FileNotFoundError:
        return None, 'file_missing'
    except (OSError, ValueError) as exc:
        return None, type(exc).__name__ + ': ' + str(exc)


def artifact(path):
    try:
        data = path.read_bytes()
        return {'path': str(path), 'exists': True, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    except OSError as exc:
        return {'path': str(path), 'exists': False, 'error_type': type(exc).__name__}


def input_index(value):
    if isinstance(value, dict):
        value = value.get('samples', value.get('inputs', []))
    if value is None:
        return {}
    if not isinstance(value, list):
        raise ValueError('input manifest must be a list or contain samples/inputs list')
    result = {}
    for row in value:
        if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str):
            raise ValueError('input row missing sample_id')
        if row['sample_id'] in result:
            raise ValueError('duplicate sample_id in input manifest: '+row['sample_id'])
        result[row['sample_id']] = row
    return result


def records_index(summary):
    result = {}
    for row in summary.get('records', []) if isinstance(summary, dict) else []:
        if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str):
            raise ValueError('summary record missing sample_id')
        if row['sample_id'] in result:
            raise ValueError('duplicate sample_id in summary: '+row['sample_id'])
        result[row['sample_id']] = row
    return result


def boundary(row):
    try:
        start, end = Decimal(str(row['bg'])), Decimal(str(row['ed']))
    except (KeyError, TypeError, InvalidOperation):
        raise ValueError('missing/invalid bg or ed') from None
    if not start.is_finite() or not end.is_finite() or start < 0 or end <= start:
        raise ValueError('nonfinite or invalid time interval')
    return start, end


def display_boundary(key):
    return [format(key[0], 'f'), format(key[1], 'f')]


def segment_index(rows):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('segment must be object')
        key = boundary(row)
        if key in result:
            raise ValueError('duplicate exact segment boundary: '+str(display_boundary(key)))
        result[key] = row
    return result


def warning_index(payload):
    rows = payload.get('segment_warnings')
    if rows is None:
        return {}, [], None
    if not isinstance(rows, list):
        raise ValueError('segment_warnings is not list')
    index, unaligned = {}, []
    for row in rows:
        try:
            key = boundary(row)
            index.setdefault(key, []).append(row)
        except (ValueError, TypeError):
            unaligned.append(row)
    return index, unaligned, rows


def side(directory, inputs, response_paths, mode, provenance):
    summary_path = directory/'summary.json'
    summary, error = read_json(summary_path)
    return {'directory': str(directory), 'inputs': inputs, 'records': records_index(summary),
            'summary': summary if isinstance(summary, dict) else {}, 'summary_error': error,
            'response_paths': response_paths, 'mode': mode, 'mode_source': provenance,
            'summary_artifact': artifact(summary_path)}


def completed(side_data, sample_id):
    record = side_data['records'].get(sample_id)
    path = side_data['response_paths'].get(sample_id)
    evidence = {'record': record, 'response_artifact': artifact(path) if path is not None else None}
    if side_data['summary_error']:
        return None, 'summary_unavailable: '+side_data['summary_error'], evidence
    if not record or record.get('status') != 'completed':
        return None, 'summary_record_not_completed: '+str((record or {}).get('status', 'missing')), evidence
    if path is None:
        return None, 'response_path_unknown', evidence
    payload, error = read_json(path)
    if error:
        return None, 'response_unavailable: '+error, evidence
    if (not isinstance(payload, dict) or payload.get('code') not in (None, 0)
            or not isinstance(payload.get('segments'), list)
            or not all(isinstance(row, dict) for row in payload['segments'])):
        return None, 'invalid_complete_response_schema', evidence
    if payload.get('processing_status', 'complete') not in ('complete', 'completed_with_warnings'):
        return None, 'response_processing_status_not_complete', evidence
    return payload, None, evidence


def source_sha(sample, record):
    value = sample.get('sha256') if sample else None
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', value):
        return None, 'manifest_source_sha_unknown'
    value = value.lower()
    recorded = (record or {}).get('sha256')
    if recorded is not None and (not isinstance(recorded, str) or recorded.lower() != value):
        return None, 'summary_sha_disagrees_with_manifest'
    return value, None


def response_metadata(payload):
    segments = payload['segments']
    return {'fields': {key:value for key,value in payload.items() if key not in ('segments', 'text')},
            'segment_count': len(segments),
            'role_nonnull': sum(row.get('role') is not None for row in segments),
            'emotion_nonnull': sum(row.get('emotion') is not None for row in segments),
            'top_level_text': payload.get('text'), 'top_level_text_present': 'text' in payload}


def compare(name, left, right, sample_ids, *, cross_mode=False):
    report = {'name': name, 'scope': SCOPE, 'left_mode': left['mode'], 'right_mode': right['mode'],
              'mode_evidence': {'left': left['mode_source'], 'right': right['mode_source']},
              'comparison_kind': 'cross_mode_A_vs_B' if cross_mode else 'declared_A_vs_A',
              'expected_enhancement_field_changes': ['role', 'emotion'] if cross_mode else [],
              'mode_note': ('A4关闭增强，B4请求增强，role/emotion出现差异符合开关目的；这不等于增强正确，'
                            '仍完整保留实际字段、空值和转写差异。' if cross_mode else
                            '旧实验A与正式A6的同源输出对照；不把隔离实验视为本次正式性能验收。'),
              'request_field_note': '逐请求fields有记录则保留；正式controller未记录的表单字段为unknown，不伪造同表单证明。',
              'left_summary_artifact': left['summary_artifact'], 'right_summary_artifact': right['summary_artifact'],
              'left_phase_trigger': left['summary'].get('trigger'), 'right_phase_trigger': right['summary'].get('trigger'),
              'audios': []}
    for sample_id in sample_ids:
        old, old_error, old_evidence = completed(left, sample_id)
        new, new_error, new_evidence = completed(right, sample_id)
        old_sample, new_sample = left['inputs'].get(sample_id), right['inputs'].get(sample_id)
        old_sha, old_sha_error = source_sha(old_sample, old_evidence['record'])
        new_sha, new_sha_error = source_sha(new_sample, new_evidence['record'])
        item = {'sample_id': sample_id, 'status': 'unknown', 'unknown_reasons': [],
                'left_source': old_sample, 'right_source': new_sample,
                'left_sha256': old_sha, 'right_sha256': new_sha,
                'left_evidence': old_evidence, 'right_evidence': new_evidence,
                'left_request_fields': (old_evidence['record'] or {}).get('fields'),
                'right_request_fields': (new_evidence['record'] or {}).get('fields'),
                'different_boundaries': None, 'matched_text_changes': None, 'changes': []}
        report['audios'].append(item)
        for label, reason in [('left', old_error), ('right', new_error), ('left', old_sha_error), ('right', new_sha_error)]:
            if reason:
                item['unknown_reasons'].append(label+': '+reason)
        if old_sha and new_sha and old_sha != new_sha:
            item['unknown_reasons'].append('source_sha_mismatch')
        if old is not None:
            item['left_complete_response_metadata'] = response_metadata(old)
        if new is not None:
            item['right_complete_response_metadata'] = response_metadata(new)
        if item['unknown_reasons']:
            continue
        try:
            old_segments, new_segments = segment_index(old['segments']), segment_index(new['segments'])
            old_warnings, old_unaligned, old_warning_rows = warning_index(old)
            new_warnings, new_unaligned, new_warning_rows = warning_index(new)
        except ValueError as exc:
            item['unknown_reasons'].append('exact_alignment_invalid: '+str(exc))
            item['unalignable_complete_payloads'] = {'left': old, 'right': new}
            continue
        item.update(status='comparable', source_sha_matches=True,
            left_segment_count=len(old_segments), right_segment_count=len(new_segments),
            left_warnings=old_warning_rows, right_warnings=new_warning_rows,
            unaligned_warnings_left=old_unaligned, unaligned_warnings_right=new_unaligned,
            warning_comparison='known' if old_warning_rows is not None and new_warning_rows is not None else 'unknown_missing_warning_field',
            top_level_text_changed=old.get('text') != new.get('text'))
        for key in sorted(old_segments.keys() | new_segments.keys() | old_warnings.keys() | new_warnings.keys()):
            a, b = old_segments.get(key), new_segments.get(key)
            aw, bw = old_warnings.get(key, []), new_warnings.get(key, [])
            field_changes = ({field: {'left': a.get(field), 'right': b.get(field),
                'left_present': field in a, 'right_present': field in b,
                'expected_cross_mode_field': cross_mode and field in {'role','emotion'}}
                for field in sorted(a.keys() | b.keys()) if field not in ('bg','ed')
                and (a.get(field) != b.get(field) or (field in a) != (field in b))} if a is not None and b is not None else {})
            if a is not None and b is not None and not field_changes and aw == bw:
                continue
            if a is None and b is None and aw == bw:
                continue
            item['changes'].append({'bg': display_boundary(key)[0], 'ed': display_boundary(key)[1],
                'status': 'right_only_boundary' if a is None and b is not None else
                          'left_only_boundary' if b is None and a is not None else
                          'warning_only_boundary' if a is None and b is None else 'matched_changed',
                'left': a, 'right': b, 'field_changes': field_changes,
                'left_warnings': aw, 'right_warnings': bw,
                'warning_evidence': item['warning_comparison'],
                'text_changed': a is not None and b is not None and a.get('segment_text') != b.get('segment_text')})
        item['different_boundaries'] = len(item['changes'])
        item['matched_text_changes'] = sum(change['text_changed'] for change in item['changes'])
        item['left_only_boundaries'] = sum(change['status'] == 'left_only_boundary' for change in item['changes'])
        item['right_only_boundaries'] = sum(change['status'] == 'right_only_boundary' for change in item['changes'])
        item['field_change_counts'] = dict(Counter(field for change in item['changes'] for field in change['field_changes']))
        # Keep transcript changes independently accessible even when every
        # enhanced role/emotion field changes in the cross-mode comparison.
        item['text_change_details'] = [change for change in item['changes'] if change['text_changed']]
    usable = [item for item in report['audios'] if item['status'] == 'comparable']
    report['summary'] = {'expected_audio_pairs': len(sample_ids), 'comparable_audio_pairs': len(usable),
        'unknown_audio_pairs': len(sample_ids)-len(usable),
        'different_boundaries_in_comparable_pairs': sum(item['different_boundaries'] for item in usable) if usable else None,
        'matched_text_changes_in_comparable_pairs': sum(item['matched_text_changes'] for item in usable) if usable else None,
        'matched_field_changes_in_comparable_pairs': dict(Counter(
            field for item in usable for change in item['changes'] for field in change['field_changes'])),
        'left_only_boundaries_in_comparable_pairs': sum(item['left_only_boundaries'] for item in usable) if usable else None,
        'right_only_boundaries_in_comparable_pairs': sum(item['right_only_boundaries'] for item in usable) if usable else None,
        'all_audio_pairs_comparable': len(usable) == len(sample_ids),
        'aggregation_scope': '仅可比较完整配对；unknown不参与求和且不解释为0变化'}
    return report


def highlighted(a, b):
    a, b = str(a), str(b)
    left, right = [], []
    for kind, i, j, k, l in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        x, y = html.escape(a[i:j]), html.escape(b[k:l])
        left.append(x if kind == 'equal' else '<mark>'+x+'</mark>')
        right.append(y if kind == 'equal' else '<mark>'+y+'</mark>')
    return ''.join(left), ''.join(right)


def as_pre(value):
    return '<pre>'+html.escape(json.dumps(value, ensure_ascii=False, indent=2, default=json_decimal))+'</pre>'


def json_decimal(value):
    if isinstance(value, Decimal):
        return format(value, 'f')
    raise TypeError('not JSON serializable: '+type(value).__name__)


def render_html(report):
    body = ['<h1>'+html.escape(report['name'])+'</h1><p>'+html.escape(SCOPE)+'</p>',
            '<p>'+html.escape(report['mode_note'])+'</p>', as_pre(report['summary']),
            '<label>搜索文本／时间／字段／警告 <input id="query" type="search"></label>',
            '<label><input id="textOnly" type="checkbox"> 仅匹配边界的文本变化（独立于增强字段）</label>',
            '<p id="visible"></p>']
    for item in report['audios']:
        body.append('<section><h2>'+html.escape(item['sample_id'])+' — '+html.escape(item['status'])+'</h2>')
        body.append('<details><summary>SHA、完成状态、源文件、原始警告及响应元数据</summary>'+as_pre({
            key:value for key,value in item.items() if key not in ('changes', 'text_change_details')})+'</details>')
        if item['status'] == 'unknown':
            body.append('<p>UNKNOWN：'+html.escape('; '.join(item['unknown_reasons']))+'</p>')
        for direction in ('left', 'right'):
            warning_rows = item.get(direction+'_warnings',
                item.get(direction+'_complete_response_metadata', {}).get('fields', {}).get('segment_warnings'))
            if warning_rows:
                body.append('<article data-kind="warning"><h3>'+html.escape(item['sample_id']+' '+direction.upper()+
                    ' 完整警告审计（未必是变化）')+'</h3>'+as_pre(warning_rows)+'</article>')
        for change in item['changes']:
            a, b = change['left'], change['right']
            left, right = highlighted(a.get('segment_text', '') if a else '[无对应分段]',
                                      b.get('segment_text', '') if b else '[无对应分段]')
            body.append('<article data-kind="'+('text' if change['text_changed'] else 'fields')+'"><h3>'+html.escape(item['sample_id']+' '+change['bg']+'–'+change['ed']+' '+change['status'])+
                '</h3><div class="pair"><div><strong>LEFT</strong><pre>'+left+'</pre></div><div><strong>RIGHT</strong><pre>'+right+
                '</pre></div></div><details><summary>完整字段与警告实际内容</summary>'+as_pre(change)+'</details></article>')
        body.append('</section>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>'+html.escape(report['name'])+'</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:24px;max-width:1400px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f8;padding:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}section,article{border-top:1px solid #ccd;padding:14px 0}mark{background:#ffe09a}input{font:inherit;min-width:300px}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>'+
        ''.join(body)+'<script>const input=document.getElementById("query"),textOnly=document.getElementById("textOnly");function filter(){let n=0;for(const a of document.querySelectorAll("article")){a.hidden=!a.textContent.toLowerCase().includes(input.value.toLowerCase())||(textOnly.checked&&a.dataset.kind!=="text");if(!a.hidden)n++;}document.getElementById("visible").textContent="显示差异／警告审计条目："+n;}input.addEventListener("input",filter);textOnly.addEventListener("change",filter);filter();</script></html>')


def render_markdown(report):
    rows = ['# '+report['name'], '', SCOPE, '', report['mode_note'], '',
            '匹配边界文本变化单列在JSON的 `text_change_details`；HTML勾选“仅匹配边界的文本变化”可独立查阅。', '',
            '| 音频 | 状态 | 文本变化 | 差异边界 | role / emotion变化 | 左 / 右独有边界 | 说明 |', '|---|---|---:|---:|---:|---:|---|']
    for item in report['audios']:
        text_count = str(item['matched_text_changes']) if item['matched_text_changes'] is not None else 'unknown'
        boundary_count = str(item['different_boundaries']) if item['different_boundaries'] is not None else 'unknown'
        rows.append('| '+item['sample_id']+' | '+item['status']+' | '+text_count+
                    ' | '+boundary_count+' | '+(' / '.join(str(item.get('field_change_counts', {}).get(field, 0)) for field in ('role','emotion')) if item['status']=='comparable' else 'unknown')+
                    ' | '+(' / '.join(str(item.get(field)) for field in ('left_only_boundaries','right_only_boundaries')) if item['status']=='comparable' else 'unknown')+
                    ' | '+('; '.join(item['unknown_reasons']) or '完整同源配对，SHA一致')+' |')
    rows.extend(['', '完整实际内容见 ['+report['name']+'.html]('+report['name']+'.html) 与 JSON；HTML可搜索并展开完整字段。',
                 '', 'A/B开关导致的角色、情绪字段差异与文本差异分别保留；具体质量需人工听音/真值，不能据此称准确率提高。'])
    return '\n'.join(rows)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not args.candidate_run.is_dir():
        parser.error('candidate run directory does not exist')
    if args.output.exists():
        parser.error('output must be a new directory; existing reports are not overwritten')
    old_run, old_run_error = read_json(BASELINE/'run.json')
    inputs, input_error = read_json(args.candidate_run/'inputs.json')
    old_inputs, new_inputs = input_index((old_run or {}).get('samples')), input_index(inputs)
    old_paths = {row['sample_id']: BASELINE/f'response-{index}.json'
                 for index,row in enumerate((old_run or {}).get('samples', []))}
    baseline = side(BASELINE, old_inputs, old_paths, (old_run or {}).get('arguments', {}).get('mode'), 'baseline run.json.arguments.mode')
    candidates = {phase: side(args.candidate_run/phase, new_inputs,
        {sample_id: args.candidate_run/phase/f'response-{sample_id}.json' for sample_id in new_inputs},
        'B' if phase == 'B4' else 'A', 'formal controller stage '+phase) for phase in ('A6','A4','B4')}
    reports = [compare('baseline_vs_formal_A6', baseline, candidates['A6'], [f'a{i}' for i in range(1,7)]),
               compare('A4_vs_B4', candidates['A4'], candidates['B4'], [f'a{i}' for i in range(1,5)], cross_mode=True)]
    args.output.mkdir(parents=True, exist_ok=False)
    for report in reports:
        report.update(generated_utc=datetime.now(timezone.utc).isoformat(),
            candidate_run=str(args.candidate_run), baseline_run=str(BASELINE),
            manifest_read_errors={'baseline': old_run_error, 'candidate': input_error},
            source_identity_scope='比较manifest与各summary声明的源SHA；不重新读完整WAV。controller应已独立核验真实源SHA。',
            numeric_representation='输入小数以Decimal精确保留；报告JSON中的小数写为十进制字符串，分段按精确bg/ed对齐。',
            input_artifacts=[artifact(BASELINE/'run.json'),artifact(args.candidate_run/'inputs.json')])
        name = report['name']
        (args.output/(name+'.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=json_decimal), encoding='utf-8')
        (args.output/(name+'.html')).write_text(render_html(report), encoding='utf-8')
        (args.output/(name+'.md')).write_text(render_markdown(report), encoding='utf-8')
    print(json.dumps({report['name']:report['summary'] for report in reports}, ensure_ascii=False))


if __name__ == '__main__':
    main()
