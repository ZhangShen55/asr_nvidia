import json

import numpy as np
import soundfile as sf

from scripts.vad_quality_validation import align_file, parse_logs, scan_quality


def _payload(segments):
    return {"segments": segments}


def test_align_file_keeps_overlap_and_baseline_only(tmp_path):
    wav = tmp_path / "fixture.wav"
    sf.write(wav, np.zeros(16000 * 4, dtype=np.float32), 16000)
    baseline = _payload([
        {"bg": "0", "ed": "1", "segment_text": "保留"},
        {"bg": "1", "ed": "2", "segment_text": "被过滤候选"},
    ])
    response = _payload([
        {"bg": "0.2", "ed": "0.8", "segment_text": "保留"},
    ])
    result = align_file(baseline, response, wav)
    assert result["overlap_count"] == 1
    assert result["baseline_only_count"] == 1
    assert result["baseline_only"][0]["baseline"]["text"] == "被过滤候选"
    assert result["baseline_only"][0]["rms_dbfs"] < -100


def test_quality_scanner_distinguishes_repetition_from_normal_text():
    repeated = "这个，" * 80
    payload = _payload([
        {"bg": "0", "ed": "1", "segment_text": repeated},
        {"bg": "1", "ed": "3", "segment_text": "同学们请看这里。"},
    ])
    result = scan_quality(payload)
    assert result["loop_segments"]
    assert result["long_text_segments"]
    assert not scan_quality(_payload([
        {"bg": "0", "ed": "1", "segment_text": "重复这个词一次"},
    ]))["loop_segments"]


def test_parse_logs_supports_current_and_legacy_v2_lines(tmp_path):
    log = tmp_path / "service.log"
    log.write_text(
        "2026 INFO [V2 耗时] 段数=4 总计=120ms | VAD=10 裁剪=2 分段处理=90 "
        "(Qwen 等待 0ms/调用 20ms/4次, 说话人等待 0ms/推理 0ms/0次, "
        "情绪等待 0ms/推理 0ms/0次) 聚类=0 ITN=1 低音量丢弃=2 task_id=t1\n"
        "2026 INFO [V2 耗时] 段数=3 总计=80ms | VAD=8 裁剪=1 分段处理=60 "
        "(Qwen 等待 0ms/调用 20ms/3次, 说话人等待 0ms/推理 0ms/0次, "
        "情绪等待 0ms/推理 0ms/0次) 聚类=0 ITN=1 task_id=t2\n",
        encoding="utf-8",
    )
    parsed = parse_logs(log)
    assert len(parsed["phases"]) == 2
    assert parsed["phases"][0]["low_volume_dropped"] == 2
    assert parsed["phases"][1]["low_volume_dropped"] == 0
