import pytest
from core.config import Settings


def test_strong_threshold_settings_are_typed(tmp_path):
    p=tmp_path/"config.toml"
    p.write_text('[qwen_asr]\nonline_repetition_strong_tail_min_chars=96\nonline_repetition_strong_tail_min_ratio=95\nonline_repetition_strong_tail_repeats_per_second=4\n')
    c=Settings(config_path=str(p)).qwen_online_repetition_config
    assert (c.strong_tail_min_chars,c.strong_tail_min_ratio,c.strong_tail_repeats_per_second)==(96,95,4)
    for value in ['0','101','true','"95"','95.0']:
        p.write_text('[qwen_asr]\nonline_repetition_strong_tail_min_ratio='+value+'\n')
        with pytest.raises(ValueError):Settings(config_path=str(p)).validate_model_backends()


def test_transport_budget_is_bounded_and_typed(tmp_path):
    p=tmp_path/"transport.toml"
    p.write_text("")
    assert Settings(config_path=str(p)).qwen_continuous_admission_drain_ms == 0
    for v in [0,3,5]:
        p.write_text('[qwen_asr]\ncontinuous_admission_drain_ms='+str(v)+'\n')
        assert Settings(config_path=str(p)).qwen_continuous_admission_drain_ms == v
    for v in ['-1','6','true','"3"','3.0']:
        p.write_text('[qwen_asr]\ncontinuous_admission_drain_ms='+v+'\n')
        with pytest.raises(ValueError):Settings(config_path=str(p)).validate_model_backends()
