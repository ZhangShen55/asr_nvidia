"""数组驱动增强能力及旧键迁移契约；不加载模型或设备。"""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from core.config import Settings


def settings_file(tmp_path, text):
    path = tmp_path / "enhancement.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load(tmp_path, text=""):
    return Settings(config_path=str(settings_file(tmp_path, text)))


@pytest.mark.parametrize("cam", [[], ["npu:7", "npu:7"]])
@pytest.mark.parametrize("emo", [[], ["cpu"]])
def test_arrays_are_the_only_enabled_source_without_health_or_request_state(tmp_path, cam, emo):
    import json
    cfg = load(tmp_path, "[model_device_pool]\n"
               f"speech_campplus_devices={json.dumps(cam)}\n"
               f"emotion2vec_devices={json.dumps(emo)}\n")
    assert cfg.speech_campplus_enabled is bool(cam)
    assert cfg.emotion2vec_enabled is bool(emo)
    assert cfg.model_device_speech_campplus_devices == cam
    assert cfg.model_device_emotion2vec_devices == emo
    assert cfg.local_model_pool_speech_campplus_instances == len(cam)
    assert cfg.local_model_pool_emotion2vec_instances == len(emo)
    assert not hasattr(cfg, "open_v2_role")
    assert not hasattr(cfg, "open_v2_emotion")


def test_missing_arrays_default_to_disabled_and_results_are_not_mutable_configuration(tmp_path):
    cfg = load(tmp_path)
    assert cfg.speech_campplus_enabled is False
    assert cfg.emotion2vec_enabled is False
    with pytest.raises(AttributeError):
        cfg.speech_campplus_enabled = True
    with pytest.raises(AttributeError):
        cfg.emotion2vec_enabled = True


@pytest.mark.parametrize("key,array", [("open_v2_role", "speech_campplus_devices"),
                                      ("open_v2_emotion", "emotion2vec_devices")])
@pytest.mark.parametrize("literal", ["true", "false", '"false"', "1"])
def test_every_removed_key_value_is_rejected_with_field_and_migration(tmp_path, key, array, literal):
    text = f"[features]\n{key}={literal}\n[model_device_pool]\n{array}=[\"npu:7\"]\n"
    with pytest.raises(ValueError) as caught:
        load(tmp_path, text)
    message = str(caught.value)
    assert f"features.{key}" in message
    assert array in message
    assert "true" in message and "false" in message and "[]" in message


def test_combined_removed_keys_report_both_and_preflight_rechecks_in_memory_input(tmp_path):
    cfg = load(tmp_path)
    cfg._cfg["features"] = {"open_v2_role": False, "open_v2_emotion": True}
    with pytest.raises(ValueError) as caught:
        cfg.validate_model_backends()
    assert "features.open_v2_role" in str(caught.value)
    assert "features.open_v2_emotion" in str(caught.value)


@pytest.mark.parametrize("model,prop", [("speech_campplus", "speech_campplus_enabled"),
                                       ("emotion2vec", "emotion2vec_enabled")])
@pytest.mark.parametrize("literal", ['"npu:7"', "true", "1", '["cuda:0"]', '["npu:x"]', "[false]", '[""]'])
def test_enabled_getter_strictly_validates_arrays_instead_of_truthiness(tmp_path, model, prop, literal):
    cfg = load(tmp_path, f"[model_device_pool]\n{model}_devices={literal}\n")
    with pytest.raises(ValueError, match=model + "_devices"):
        getattr(cfg, prop)
    with pytest.raises(ValueError, match=model + "_devices"):
        cfg.validate_model_backends()


@pytest.mark.parametrize("legacy_features", ["features=true", "features=[]"])
def test_features_section_type_is_not_silently_coerced(tmp_path, legacy_features):
    with pytest.raises(ValueError, match="features"):
        load(tmp_path, legacy_features)


@pytest.mark.parametrize("key", ["open_v2_role", "open_v2_emotion"])
@pytest.mark.parametrize("literal", ["true", "false"])
def test_removed_keys_fail_main_import_before_worker_capable_modules(tmp_path, key, literal):
    # 新进程证明main入口在配置阶段失败，而非已启动部分模型后才清理。
    path = settings_file(tmp_path, f"[features]\n{key}={literal}\n")
    root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name in {"core.vad_pool", "core.models", "core.qwen_asr_client"}:
        raise AssertionError("worker-capable module imported before rejecting old configuration")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
try:
    import main
except ValueError as error:
    assert "features." in str(error) and "已删除" in str(error)
    print("REMOVED_FEATURE_REJECTED_BEFORE_MODEL_IMPORT")
else:
    raise AssertionError("old configuration accepted")
'''
    env = dict(os.environ, CONFIG_PATH=str(path), PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "REMOVED_FEATURE_REJECTED_BEFORE_MODEL_IMPORT" in result.stdout


def test_template_cleanup_keeps_effective_defaults_and_retained_guard_controls(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ("config.toml", "docker/config.toml"):
        cfg = Settings(config_path=str(root / name))
        qwen = cfg._cfg["qwen_asr"]
        pool = cfg._cfg["model_device_pool"]
        assert "features" not in cfg._cfg
        assert not {"batch_wait_timeout_ms", "batch_max_duration_ratio"}.intersection(qwen)
        assert not {"speech_campplus_batch_size", "emotion2vec_batch_size"}.intersection(pool)
        assert cfg.speech_campplus_enabled and cfg.emotion2vec_enabled
        assert cfg.speech_campplus_batch_size == cfg.emotion2vec_batch_size == 1
        assert cfg.speech_campplus_prefetch_queue_depth == cfg.emotion2vec_prefetch_queue_depth == 0
        assert cfg.qwen_local_instance_count == 4 and cfg.qwen_segment_concurrency == 16
        assert (cfg.qwen_max_batch_audio_frames, cfg.qwen_max_batch_tokens) == (14400, 10000)
        assert cfg.qwen_online_repetition_enabled and cfg.qwen_repetition_guard_enabled
        assert cfg.qwen_queue_wait_timeout_s == 0


def test_explicit_legacy_batch_and_fixed_one_validation_remain_available(tmp_path):
    cfg = load(tmp_path, '[qwen_asr]\ngeneration_mode="legacy"\nbatch_wait_timeout_ms=100\n'
               'batch_max_duration_ratio=32\n[model_device_pool]\nspeech_campplus_batch_size=1\n'
               'emotion2vec_batch_size=1\n')
    cfg.validate_model_backends()
    assert cfg.qwen_batch_wait_timeout_ms == 100
    assert cfg.qwen_batch_max_duration_ratio == 32
    for key in ("speech_campplus_batch_size", "emotion2vec_batch_size"):
        cfg._cfg["model_device_pool"][key] = 2
        with pytest.raises(ValueError, match=key):
            cfg.validate_model_backends()
        cfg._cfg["model_device_pool"][key] = 1
