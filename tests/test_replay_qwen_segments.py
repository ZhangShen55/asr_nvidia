import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("replay_qwen_segments", Path(__file__).parents[1]/"scripts/replay_qwen_segments.py")
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)


def archive(tmp_path):
    audio = np.arange(160, dtype=np.float32) / 160
    samples = tmp_path/"samples.npz"
    np.savez(samples, audio_000=audio)
    data = {"sample_rate": 16000, "samples": [{"index": 0, "sample_id": "vad0", "audio_key": "audio_000",
           "source_id": "a1", "start_sample": 20, "end_sample": 180, "duration_s": .01,
           "pcm_sha256": hashlib.sha256(audio.tobytes()).hexdigest()}]}
    manifest = tmp_path/"manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, samples, data


def test_verified_pcm_and_pinned_settings(tmp_path):
    a, b, _ = archive(tmp_path)
    result = m.build(a, b, language="English")
    assert result["sample_count"] == 1
    assert result["samples"][0]["segment_id"] == "vad0"
    assert result["samples"][0]["language"] == "English"
    assert result["generation"]["max_new_tokens"] == 256
    assert not result["historical_replay"]["exact_old_service_batch_restored"]


def test_corruption_and_missing_selection_rejected(tmp_path):
    a, b, data = archive(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        m.build(a, b, indices=[99])
    data["samples"][0]["pcm_sha256"] = "wrong"
    a.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="SHA"):
        m.build(a, b)
