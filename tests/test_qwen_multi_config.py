"""正式continuous配置只测试loader，不导入或启动模型。"""
import logging
from pathlib import Path
import pytest
from core.config import Settings


def config(tmp_path, values=""):
    path = tmp_path / "multi.toml"
    path.write_text("[qwen_asr]\n" + values, encoding="utf-8")
    return Settings(config_path=str(path))


def test_formal_defaults(tmp_path):
    cfg = config(tmp_path)
    cfg.validate_model_backends()
    assert (cfg.qwen_generation_mode, cfg.qwen_execution_mode) == ("continuous", "process")
    assert cfg.qwen_preload_on_startup is True
    assert (cfg.qwen_local_instance_count, cfg.qwen_segment_concurrency) == (4, 16)
    assert (cfg.qwen_audio_concurrency, cfg.qwen_request_concurrency) == (4, 64)
    assert (cfg.qwen_max_inference_batch_size, cfg.qwen_continuous_prefill_batch_size) == (16, 16)
    assert (cfg.qwen_max_batch_audio_frames, cfg.qwen_max_batch_tokens) == (14400, 10000)
    assert (cfg.qwen_pending_segment_limit, cfg.qwen_pending_audio_bytes_limit) == (256, 268435456)
    assert cfg.qwen_owner_threads == 1
    assert cfg.qwen_online_repetition_enabled is True
    assert cfg.qwen_continuous_admission_drain_ms == 3


@pytest.mark.parametrize("n", range(1, 9))
@pytest.mark.parametrize("s", [1, 8, 16, 32, 128])
def test_derived_capacity_and_owner_budgets(tmp_path, n, s):
    cfg = config(tmp_path, f"local_instance_count={n}\nsegment_concurrency={s}\n")
    cfg.validate_model_backends()
    assert (cfg.qwen_audio_concurrency, cfg.qwen_request_concurrency) == (n, n*s)
    assert cfg.qwen_max_inference_batch_size == s
    assert cfg.qwen_continuous_prefill_batch_size == min(16, s)
    assert (cfg.qwen_max_batch_audio_frames, cfg.qwen_max_batch_tokens) == (14400, 10000)
    assert cfg.qwen_pending_segment_limit == 256


def test_equal_legacy_capacity_warns_once(tmp_path, caplog):
    cfg = config(tmp_path, "local_instance_count=6\naudio_concurrency=6\nrequest_concurrency=96\n")
    with caplog.at_level(logging.WARNING):
        cfg.validate_model_backends()
        cfg.validate_model_backends()
    assert sum("派生" in row.message for row in caplog.records) == 2


@pytest.mark.parametrize("field,value", [
    ("audio_concurrency", "8"), ("request_concurrency", "128"),
    ("audio_concurrency", "true"), ("request_concurrency", '"96"'),
    ("local_instance_count", "9"), ("local_instance_count", "false"),
    ("segment_concurrency", "0"), ("max_inference_batch_size", "17"),
    ("continuous_prefill_batch_size", "17"), ("owner_threads", "0"),
    ("owner_threads", "true"), ("max_batch_audio_frames", "99"),
    ("max_batch_tokens", "524"), ("max_new_tokens", "10000"),
])
def test_bad_configs_rejected_before_loading(tmp_path, field, value):
    with pytest.raises(ValueError):
        config(tmp_path, f"{field}={value}\n").validate_model_backends()


def test_explicit_legacy_keeps_independent_capacities(tmp_path):
    cfg = config(tmp_path, 'generation_mode="legacy"\nlocal_instance_count=2\nsegment_concurrency=16\nrequest_concurrency=128\naudio_concurrency=8\nmax_inference_batch_size=64\n')
    cfg.validate_model_backends()
    assert (cfg.qwen_audio_concurrency, cfg.qwen_request_concurrency) == (8, 128)
    assert cfg.qwen_max_inference_batch_size == 64
    assert cfg.qwen_execution_mode == "thread"
    assert cfg.qwen_preload_on_startup is False
    assert cfg.qwen_online_repetition_enabled is False
    assert cfg.qwen_continuous_admission_drain_ms == 0


def test_shipped_templates_derive_capacities(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name, device in (("config.toml", "npu:7"), ("docker/config.toml", "npu:0")):
        cfg = Settings(config_path=str(root/name))
        cfg.validate_model_backends()
        assert cfg.instance_count == 1
        assert cfg.qwen_local_device == device
        assert cfg.qwen_local_instance_count == 4
        assert (cfg.qwen_audio_concurrency, cfg.qwen_request_concurrency) == (4, 64)
        assert cfg.qwen_segment_concurrency == 16
        assert cfg.qwen_audio_spk_emo_concurrency == 4
        assert cfg.model_device_speech_campplus_devices == [device] * 4
        assert cfg.model_device_speech_campplus_execution_mode == "process"
        assert cfg.model_device_emotion2vec_devices == [device]
        assert not {"audio_concurrency", "request_concurrency", "max_inference_batch_size", "continuous_prefill_batch_size"}.intersection(cfg._cfg["qwen_asr"])
