#!/usr/bin/env python3
"""Build final N4 transcript comparisons only after owned shutdown passed.

This report-only tool imports the adjacent standard-library comparison helpers.
It never imports application/model code, sends HTTP, or queries a device.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Importing the adjacent helper must not create an extra write outside output.
sys.dont_write_bytecode = True
from compare_results import (artifact, boundary, compare, completed, input_index,
                             json_decimal, read_json, render_html,
                             render_markdown, side)


PROJECT = Path('/home/zhangs/asr_nvidia')
FINAL = PROJECT/'logs/qwen-multi-instance-20261124/final-m4-cam4-emo1-v1'
OLD_N5 = PROJECT/'logs/qwen-multi-instance-20261124/followup-m5-s16-v1'
DEFAULT_OUTPUT = PROJECT/'docs/qwen-multi-instance-20261124/final4-differences'
SAMPLES = ['a1', 'a2', 'a3', 'a4']


def load_inputs(directory):
    payload, error = read_json(directory/'inputs.json')
    if error:
        raise ValueError('input manifest unavailable: '+str(directory)+': '+error)
    return input_index(payload)


def build_side(directory, phase, inputs, mode):
    return side(directory/phase, inputs,
                {sample: directory/phase/f'response-{sample}.json' for sample in SAMPLES},
                mode, str(directory)+' controller phase '+phase)


def separate_warnings(report):
    warning_known_pairs = 0
    warning_change_count = 0
    for audio in report['audios']:
        if audio['status'] != 'comparable':
            audio['warning_changes'] = None
            continue
        if audio['warning_comparison'] != 'known':
            audio['warning_changes'] = None
            continue
        warning_known_pairs += 1
        changes = [{'bg': change['bg'], 'ed': change['ed'],
                    'left': change['left_warnings'], 'right': change['right_warnings']}
                   for change in audio['changes']
                   if change['left_warnings'] != change['right_warnings']]
        unaligned_changed = audio['unaligned_warnings_left'] != audio['unaligned_warnings_right']
        audio['warning_changes'] = changes
        audio['unaligned_warnings_changed'] = unaligned_changed
        warning_change_count += len(changes)
        # These are relative label changes, never accuracy/recall scores.
        audio['enhancement_field_changes'] = {
            field: [{'bg': change['bg'], 'ed': change['ed'], **change['field_changes'][field]}
                    for change in audio['changes'] if field in change['field_changes']]
            for field in ('role', 'emotion')}
    report['summary']['warning_known_audio_pairs'] = warning_known_pairs
    report['summary']['warning_unknown_audio_pairs'] = len(report['audios'])-warning_known_pairs
    report['summary']['changed_warning_boundaries_in_known_pairs'] = warning_change_count if warning_known_pairs else None


def normalize(value):
    # Reports deliberately serialize exact Decimal values as decimal strings.
    return json.loads(json.dumps(value, ensure_ascii=False, default=json_decimal))


def verify_example(report, left, right, preferred_sample):
    """Independently read source response rows for one concrete report example."""
    # Prefer a different audio in each comparison so the overview does not
    # repeat one coincidentally shared unstable boundary three times.
    for audio in sorted(report['audios'], key=lambda row: row['sample_id'] != preferred_sample):
        if audio['status'] != 'comparable':
            continue
        choices = audio.get('text_change_details') or audio['changes']
        for change in choices:
            if change['left'] is None or change['right'] is None:
                continue
            old, old_error, _ = completed(left, audio['sample_id'])
            new, new_error, _ = completed(right, audio['sample_id'])
            if old_error or new_error:
                raise ValueError('response changed during example verification')
            key = boundary(change)
            a = [row for row in old['segments'] if boundary(row) == key]
            b = [row for row in new['segments'] if boundary(row) == key]
            if len(a) != 1 or len(b) != 1 or normalize(a[0]) != normalize(change['left']) or normalize(b[0]) != normalize(change['right']):
                raise ValueError('raw response differs from report example: '+report['name'])
            return {'report': report['name'], 'sample_id': audio['sample_id'],
                    'bg': change['bg'], 'ed': change['ed'],
                    'status': 'raw_response_rows_verified',
                    'left_response': artifact(left['response_paths'][audio['sample_id']]),
                    'right_response': artifact(right['response_paths'][audio['sample_id']]),
                    'left': a[0], 'right': b[0],
                    'field_changes': change['field_changes'],
                    'left_warnings': change['left_warnings'], 'right_warnings': change['right_warnings']}
    return {'report': report['name'], 'status': 'no_comparable_matched_change_available',
            'note': '未补造具体差异；详见独有边界、警告或unknown状态。'}


def markdown(report):
    text = render_markdown(report)
    # The shared renderer's generic A/B footer does not describe same-A or
    # same-B comparisons. Keep the local phase-specific note authoritative.
    text = text.replace('A/B开关导致的角色、情绪字段差异与文本差异分别保留；具体质量需人工听音/真值，不能据此称准确率提高。',
                        '文本、角色/情绪字段与警告变化分别保存；同A/同B相对差异同样不等于准确率变化，无人工GT不认证准确率或召回。')
    return text+'\n警告明细见JSON的 `warning_changes`，增强字段明细见 `enhancement_field_changes`；原始警告数组始终保留。\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', type=Path, default=FINAL)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_path = args.candidate_run/'run.json'
    run, run_error = read_json(run_path)
    if run_error or not isinstance(run, dict):
        parser.error('final run.json must exist and be valid; report generation is not allowed before shutdown')
    shutdown = run.get('phases', {}).get('shutdown', {})
    if not isinstance(shutdown, dict) or shutdown.get('passed') is not True:
        parser.error('final run.json phases.shutdown.passed must be true before any report is written')
    if run.get('instance_count') != 4:
        parser.error('candidate must be the final four-owner run')
    if args.output.exists():
        parser.error('output must be a new directory; existing reports are not overwritten')

    new_inputs = load_inputs(args.candidate_run)
    old_inputs = load_inputs(OLD_N5)
    a4 = build_side(args.candidate_run, 'A4', new_inputs, 'A')
    b4 = build_side(args.candidate_run, 'B4', new_inputs, 'B')
    resident = build_side(args.candidate_run, 'A4-resident', new_inputs, 'A')
    old_b4 = build_side(OLD_N5, 'B4', old_inputs, 'B')
    definitions = [
        ('N4_A4_vs_N4_B4', a4, b4, True, 'cross_mode_A_vs_B',
         '最终N4/CAM4/emo1：同a1～a4纯A4对双增强B4。A关闭增强，B开启showSpk/showEmotion且showRoleIdentify=false；角色/情绪由null变为标签属开关预期，文本变化独立列出，不等于人工准确率。'),
        ('N4_A4_vs_N4_A4_resident', a4, resident, False, 'same_mode_A_different_residency',
         '最终N4/CAM4/emo1：同a1～a4纯A4对B4完成后全部权重继续驻留的纯A4。两轮均不请求增强；对照驻留状态下的具体输出波动，不套用旧A6实验说明，也不把相对一致当质量认证。'),
        ('N5_B4_vs_N4_B4', old_b4, b4, False, 'same_mode_B_different_owner_and_CAM_count',
         '原N5/CAM1/emo1补测B4对最终N4/CAM4/emo1 B4，均为同a1～a4双增强且showRoleIdentify=false。两轮均为B，Qwen与CAM实例数/运行状态不同；角色ID为各请求局部聚类标签，标签差异不能直接解释成人物识别错误或准确率变化。'),
    ]
    reports, examples = [], []
    generated = datetime.now(timezone.utc).isoformat()
    for index, (name, left, right, cross_mode, kind, note) in enumerate(definitions):
        report = compare(name, left, right, SAMPLES, cross_mode=cross_mode)
        report.update(comparison_kind=kind, mode_note=note,
                      generated_utc=generated, candidate_run=str(args.candidate_run),
                      old_n5_run=str(OLD_N5), final_run_artifact=artifact(run_path),
                      shutdown_gate={'path': 'phases.shutdown.passed', 'value': True},
                      report_tool_artifact=artifact(Path(__file__)),
                      shared_helper_artifact=artifact(Path(__file__).with_name('compare_results.py')),
                      manifest_artifacts=[artifact(args.candidate_run/'inputs.json'), artifact(OLD_N5/'inputs.json')],
                      source_identity_scope='精确核对各manifest和summary声明源SHA，不重读完整WAV；真实源哈希由controller独立核验。',
                      numeric_representation='输入小数以Decimal精确保留；报告JSON中的小数为十进制字符串，按bg/ed精确配对。')
        separate_warnings(report)
        examples.append(verify_example(report, left, right, SAMPLES[index]))
        reports.append(report)

    # Prepare everything before the one allowed output directory is created.
    rendered = {report['name']: {'json': json.dumps(report, ensure_ascii=False, indent=2, default=json_decimal),
                                'html': render_html(report), 'md': markdown(report)} for report in reports}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, formats in rendered.items():
        for suffix, content in formats.items():
            (args.output/(name+'.'+suffix)).write_text(content, encoding='utf-8')
    verification = {'generated_utc': generated, 'shutdown_gate': True,
                    'scope': '纯stdlib回读实际响应校验具体变化；无人工GT，不是准确率/召回验收。',
                    'examples': examples,
                    'summaries': {report['name']: report['summary'] for report in reports}}
    (args.output/'verified-examples.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2, default=json_decimal), encoding='utf-8')
    overview = ['# 最终四路具体输出差异', '', '三组均为原a1～a4；仅完整、SHA匹配响应可比，unknown不当零变化。无人工GT，不认证准确率或召回。', '',
                '| 对照 | 可比 / 4 | unknown | 匹配文本变化 | role / emotion变化 | 左 / 右独有边界 |', '|---|---:|---:|---:|---:|---:|']
    for report in reports:
        summary = report['summary']
        fields = summary['matched_field_changes_in_comparable_pairs']
        known = summary['comparable_audio_pairs'] > 0
        value = lambda key: summary[key] if summary[key] is not None else 'unknown'
        roles = fields.get('role',0) if known else 'unknown'
        emotions = fields.get('emotion',0) if known else 'unknown'
        overview.append(f"| [{report['name']}]({report['name']}.html) | {summary['comparable_audio_pairs']} | {summary['unknown_audio_pairs']} | {value('matched_text_changes_in_comparable_pairs')} | {roles} / {emotions} | {value('left_only_boundaries_in_comparable_pairs')} / {value('right_only_boundaries_in_comparable_pairs')} |")
    overview.extend(['', '完整差异见各HTML/JSON/MD，HTML可搜索并筛选仅文本；警告和增强字段另有独立明细。', '', '## 原始响应回读样例', ''])
    for example in examples:
        if example['status'] != 'raw_response_rows_verified':
            overview.append('- '+example['report']+'：'+example['status'])
            continue
        overview.extend(['- '+example['report']+' / '+example['sample_id']+' / '+example['bg']+'～'+example['ed']+'：',
                         '  - LEFT：'+json.dumps(example['left'].get('segment_text'),ensure_ascii=False),
                         '  - RIGHT：'+json.dumps(example['right'].get('segment_text'),ensure_ascii=False)])
    (args.output/'README.md').write_text('\n'.join(overview)+'\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, default=json_decimal))


if __name__ == '__main__':
    main()
