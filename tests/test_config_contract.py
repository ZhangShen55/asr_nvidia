import logging
import tempfile
import unittest
from pathlib import Path

from core.config import Settings


class ConfigContractTests(unittest.TestCase):
    def _settings(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Settings(config_path=handle.name)

    def _legacy_settings(self, text):
        # 历史能力合约显式选择legacy，避免新默认将旧fixture改成另一执行语义。
        if "generation_mode" not in text:
            if "[qwen_asr]" in text:
                text = text.replace("[qwen_asr]", '[qwen_asr]\ngeneration_mode="legacy"', 1)
            else:
                text += '\n[qwen_asr]\ngeneration_mode="legacy"\n'
        return self._settings(text)

    def test_continuous_generation_contract(self):
        self.assertEqual(self._settings("").qwen_generation_mode, "continuous")
        self.assertFalse(self._legacy_settings("").qwen_online_repetition_enabled)
        self.assertEqual(self._legacy_settings("").qwen_continuous_prefill_batch_size, 16)
        for value in ("0", "129", "true", '"16"'):
            with self.assertRaises(ValueError):
                self._legacy_settings("[qwen_asr]\ncontinuous_prefill_batch_size="+value+"\n").validate_model_backends()
        valid = self._legacy_settings('[qwen_asr]\ngeneration_mode="continuous"\nexecution_mode="process"\nlocal_instance_count=1\nsegment_concurrency=128\nrequest_concurrency=128\nmax_inference_batch_size=128\nonline_repetition_enabled=true\n')
        valid.validate_model_backends()
        for text in ('generation_mode=true', 'generation_mode="bad"', 'online_repetition_enabled="false"', 'generation_mode="continuous"\nexecution_mode="thread"', 'generation_mode="continuous"\nexecution_mode="process"\nrequest_concurrency=128\nmax_inference_batch_size=64'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self._legacy_settings('[qwen_asr]\n'+text+'\n').validate_model_backends()

    def test_online_repetition_threshold_defaults_and_overrides(self):
        from dataclasses import asdict
        from utils.incremental_repetition import RepetitionGuardConfig
        default = self._legacy_settings("").qwen_online_repetition_config
        self.assertIsInstance(default, RepetitionGuardConfig)
        self.assertEqual(asdict(default), dict(enabled=False, min_generated_tokens=32,
            min_chars=120, max_chars_per_second=40, min_repeats=8,
            min_repeat_chars=64, max_unit_chars=32, tail_chars=512, confirmations=2, strong_tail_min_chars=96,
            strong_tail_min_ratio=95, strong_tail_repeats_per_second=4))
        cfg = self._legacy_settings('[qwen_asr]\nonline_repetition_enabled=true\n'
            'online_repetition_min_generated_tokens=48\nonline_repetition_min_chars=160\n'
            'online_repetition_max_chars_per_second=45\nonline_repetition_min_repeats=10\n'
            'online_repetition_min_repeat_chars=80\nonline_repetition_max_unit_chars=40\n'
            'online_repetition_tail_chars=640\nonline_repetition_confirmations=3\n')
        cfg.validate_model_backends()
        self.assertEqual(asdict(cfg.qwen_online_repetition_config), dict(enabled=True,
            min_generated_tokens=48, min_chars=160, max_chars_per_second=45, min_repeats=10,
            min_repeat_chars=80, max_unit_chars=40, tail_chars=640, confirmations=3, strong_tail_min_chars=96,
            strong_tail_min_ratio=95, strong_tail_repeats_per_second=4))

    def test_online_repetition_thresholds_reject_wrong_types_and_combinations(self):
        from dataclasses import fields
        from utils.incremental_repetition import RepetitionGuardConfig
        for field in fields(RepetitionGuardConfig):
            if field.name == "enabled":
                continue
            for literal in ('true', 'false', '0', '-1', '1.5', '"32"', 'nan', 'inf'):
                with self.subTest(field=field.name, literal=literal), self.assertRaises(ValueError):
                    self._legacy_settings('[qwen_asr]\nonline_repetition_enabled=false\nonline_repetition_'
                        + field.name + '=' + literal + '\n').validate_model_backends()
        for text in ('online_repetition_min_repeats=2', 'online_repetition_confirmations=1',
                     'online_repetition_tail_chars=32',
                     'online_repetition_max_unit_chars=513',
                     'online_repetition_min_repeat_chars=513'):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, 'online_repetition'):
                self._legacy_settings('[qwen_asr]\n' + text + '\n').validate_model_backends()

    def test_nested_log_path_has_priority(self):
        settings = self._legacy_settings("[log]\nlog_path = \"nested.log\"\n")
        self.assertEqual(settings.log_path, "nested.log")

    def test_resource_sampling_interval_is_independent_status_configuration(self):
        self.assertEqual(self._legacy_settings("").resource_sample_interval_s, 1.0)
        cfg = self._legacy_settings('[status]\nresource_sample_interval_s=1.5\n[qwen_asr]\nrequest_timeout_s=120\n')
        self.assertEqual(cfg.resource_sample_interval_s, 1.5)
        self.assertIsInstance(cfg.resource_sample_interval_s, float)
        self.assertEqual(cfg.qwen_request_timeout_s, 120)

    def test_execution_mode_and_independent_process_shutdown_grace(self):
        self.assertEqual(self._legacy_settings("").qwen_execution_mode,"thread")
        self.assertEqual(self._legacy_settings("").qwen_process_shutdown_grace_s,30)
        for device in ("npu:0","npu:7","npu:13"):
            cfg=self._legacy_settings(f'[qwen_asr]\nexecution_mode="process"\nlocal_instance_count=1\nlocal_device="{device}"\nprocess_shutdown_grace_s=0.05\n')
            cfg.validate_model_backends()
            self.assertEqual(cfg.qwen_execution_mode,"process")
            self.assertEqual(cfg.qwen_local_device,device)
            self.assertEqual(cfg.qwen_request_timeout_s,120)
        for literal in ('true','1','"PROCESS"','"invalid"'):
            with self.assertRaisesRegex(ValueError,'execution_mode'):
                self._legacy_settings(f'[qwen_asr]\nexecution_mode={literal}\n').validate_model_backends()
        for count in range(1, 9):
            cfg = self._legacy_settings(f'[qwen_asr]\nexecution_mode="process"\nlocal_instance_count={count}\n')
            cfg.validate_model_backends()
            self.assertEqual(cfg.qwen_local_instance_count, count)
        for literal in ('true', 'false', '0', '-1', '2.5', '"2"'):
            with self.assertRaisesRegex(ValueError, 'local_instance_count'):
                self._legacy_settings(f'[qwen_asr]\nexecution_mode="process"\nlocal_instance_count={literal}\n').validate_model_backends()
        for value in ('true','0','-1','nan','inf','301','"30"'):
            with self.assertRaisesRegex(ValueError,'process_shutdown_grace_s'):
                self._legacy_settings(f'[qwen_asr]\nprocess_shutdown_grace_s={value}\n').validate_model_backends()

    def test_cam_execution_defaults_and_device_mapping(self):
        defaults = self._legacy_settings("")
        self.assertEqual(defaults.model_device_speech_campplus_execution_mode, "thread")
        self.assertEqual(defaults.model_device_speech_campplus_process_startup_timeout_s, 60.0)
        self.assertEqual(defaults.model_device_speech_campplus_process_shutdown_grace_s, 30.0)
        for devices in ('["npu:0"]', '["npu:7", "npu:13"]', '[]'):
            with self.subTest(devices=devices):
                cfg = self._legacy_settings('[model_device_pool]\nspeech_campplus_execution_mode="process"\n'
                                     f'speech_campplus_devices={devices}\n')
                cfg.validate_model_backends()
                self.assertEqual(cfg.model_device_speech_campplus_execution_mode, "process")
                self.assertEqual(cfg.qwen_request_timeout_s, 120)
                self.assertEqual(cfg.qwen_execution_mode, "thread")
        cfg = self._legacy_settings('[model_device_pool]\nspeech_campplus_devices=["cpu", "npu:0"]\n')
        cfg.validate_model_backends()
        self.assertEqual(cfg.model_device_speech_campplus_execution_mode, "thread")

    def test_cam_mode_and_unsupported_process_devices_reject_at_preflight(self):
        for literal in ('true', '1', '"PROCESS"', '""', '"invalid"'):
            with self.subTest(mode=literal):
                cfg = self._legacy_settings(f'[model_device_pool]\nspeech_campplus_execution_mode={literal}\n')
                with self.assertRaisesRegex(ValueError, 'speech_campplus_execution_mode'):
                    cfg.validate_model_backends()
        for devices in ('["cpu"]', '["npu:7", "cpu"]'):
            with self.subTest(devices=devices):
                # Disabled capabilities do not make an unsupported process combination valid.
                cfg = self._legacy_settings('[features]\nopen_v2_role=false\n[qwen_asr]\nenabled=false\n'
                                     '[model_device_pool]\nspeech_campplus_execution_mode="process"\n'
                                     f'speech_campplus_devices={devices}\n')
                with self.assertRaisesRegex(ValueError, 'speech_campplus_execution_mode.*NPU'):
                    cfg.validate_model_backends()

    def test_cam_process_deadlines_are_strict_finite_and_independent(self):
        keys = ('speech_campplus_process_startup_timeout_s', 'speech_campplus_process_shutdown_grace_s')
        for key in keys:
            for literal in ('true', 'false', '"30"', '0', '-1', '300.001', '301', 'nan', 'inf', '-inf'):
                with self.subTest(key=key, invalid=literal):
                    cfg = self._legacy_settings(f'[model_device_pool]\n{key}={literal}\n')
                    with self.assertRaisesRegex(ValueError, key):
                        cfg.validate_model_backends()
            for literal, expected in (('0.05', .05), ('60', 60.), ('300', 300.)):
                with self.subTest(key=key, valid=literal):
                    cfg = self._legacy_settings(f'[model_device_pool]\n{key}={literal}\n')
                    cfg.validate_model_backends()
                    self.assertEqual(getattr(cfg, 'model_device_' + key), expected)
                    self.assertIsInstance(getattr(cfg, 'model_device_' + key), float)
                    self.assertEqual(cfg.qwen_request_timeout_s, 120)

    def test_formal_templates_keep_independent_cam_modes(self):
        root = Path(__file__).resolve().parents[1]
        for name, device in (('config.toml', 'npu:7'), ('docker/config.toml', 'npu:0')):
            with self.subTest(template=name):
                cfg = Settings(config_path=str(root / name))
                cfg.validate_model_backends()
                self.assertEqual(cfg.model_device_speech_campplus_execution_mode, 'process' if name == 'config.toml' else 'thread')
                self.assertEqual(cfg.model_device_speech_campplus_process_startup_timeout_s, 60.)
                self.assertEqual(cfg.model_device_speech_campplus_process_shutdown_grace_s, 30.)
                self.assertEqual(cfg.model_device_speech_campplus_devices, [device])

    def test_calibrated_qwen_limits_keep_one_model_copy(self):
        settings = self._legacy_settings("")
        self.assertEqual(settings.qwen_segment_concurrency, 1)
        self.assertEqual(settings.qwen_request_concurrency, 1)
        self.assertEqual(settings.qwen_local_instance_count, 1)

    def test_batch_and_prefetch_are_independent_configuration(self):
        settings = self._legacy_settings('[qwen_asr]\nsegment_concurrency=16\nrequest_concurrency=128\n'
            'max_inference_batch_size=64\nbatch_wait_timeout_ms=10\nmax_batch_audio_frames=102400\n'
            'max_batch_tokens=78336\npending_segment_limit=256\npending_audio_bytes_limit=268435456\n'
            '[model_device_pool]\nspeech_campplus_devices=["npu:7","npu:7"]\n'
            'speech_campplus_prefetch_queue_depth=5\nspeech_campplus_batch_size=1\n')
        settings.validate_model_backends()
        self.assertEqual(settings.qwen_segment_concurrency, 16)
        self.assertEqual(settings.qwen_request_concurrency, 128)
        self.assertEqual(settings.qwen_max_inference_batch_size, 64)
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 2)
        self.assertEqual(settings.speech_campplus_prefetch_queue_depth, 5)
        self.assertEqual(settings.speech_campplus_batch_size, 1)
        self.assertEqual(settings.qwen_audio_spk_emo_concurrency, 4)

    def test_batch_duration_ratio_default_and_valid_boundaries(self):
        self.assertEqual(self._legacy_settings("").qwen_batch_max_duration_ratio, 4.0)
        for literal,expected in [("1",1.0),("1.5",1.5),("4.0",4.0),("32",32.0),("64",64.0)]:
            with self.subTest(value=literal):
                settings=self._legacy_settings(f'[qwen_asr]\nbatch_max_duration_ratio={literal}\n'
                    'segment_concurrency=16\nrequest_concurrency=128\nlocal_instance_count=1\n')
                settings.validate_model_backends()
                self.assertEqual(settings.qwen_batch_max_duration_ratio,expected)
                self.assertIsInstance(settings.qwen_batch_max_duration_ratio,float)
                self.assertEqual(settings.qwen_segment_concurrency,16)
                self.assertEqual(settings.qwen_request_concurrency,128)
                self.assertEqual(settings.qwen_local_instance_count,1)
                self.assertEqual(settings.qwen_audio_spk_emo_concurrency,4)

    def test_batch_duration_ratio_rejects_types_nonfinite_and_out_of_bounds_at_preflight(self):
        for literal in ['true','false','"32"','0','-1','0.999','64.001','65','nan','inf','-inf']:
            with self.subTest(value=literal):
                settings=self._legacy_settings(f'[qwen_asr]\nbatch_max_duration_ratio={literal}\n')
                with self.assertRaisesRegex(ValueError,'batch_max_duration_ratio'):
                    settings.validate_model_backends()

    def test_invalid_batch_budgets_fail_before_loading(self):
        fields = {'max_inference_batch_size':[0,385,True,1.5,'128'],
                  'batch_wait_timeout_ms':[-1,1001,True,'10'],
                  'max_batch_audio_frames':[0,True,'100'], 'max_batch_tokens':[0,True],
                  'pending_segment_limit':[0,4097,False], 'pending_audio_bytes_limit':[0,True],
                  'segment_concurrency':[0,129,True], 'request_concurrency':[0,4097,True]}
        import json
        for field, values in fields.items():
            for value in values:
                with self.subTest(field=field,value=value):
                    settings = self._legacy_settings(f'[qwen_asr]\n{field}={json.dumps(value)}\n')
                    with self.assertRaisesRegex(ValueError, field):
                        settings.validate_model_backends()

    def test_unsupported_model_batch_and_bad_prefetch_are_rejected(self):
        for field,value in [('speech_campplus_batch_size',2),('emotion2vec_batch_size',2),
                            ('speech_campplus_prefetch_queue_depth',-1),('emotion2vec_prefetch_queue_depth',65)]:
            with self.subTest(field=field):
                settings = self._legacy_settings(f'[model_device_pool]\n{field}={value}\n')
                with self.assertRaisesRegex(ValueError,field):
                    settings.validate_model_backends()
        settings = self._legacy_settings('[model_device_pool]\nspeech_campplus_prefetch_queue_depth=0\n')
        self.assertEqual(settings.speech_campplus_prefetch_queue_depth,0)

    def test_audio_spk_emo_concurrency_defaults_and_limits(self):
        self.assertEqual(self._legacy_settings("").qwen_audio_spk_emo_concurrency, 4)
        settings = self._legacy_settings("[qwen_asr]\naudio_concurrency = 2\naudio_spk_emo_concurrency = 7\n")
        self.assertEqual(settings.qwen_audio_concurrency, 2)
        self.assertEqual(settings.qwen_audio_spk_emo_concurrency, 7)

    def test_invalid_audio_spk_emo_concurrency(self):
        for value in ["0", "-1", "true", "false", "4.0", '"4"']:
            with self.subTest(value=value):
                settings = self._legacy_settings(f"[qwen_asr]\naudio_spk_emo_concurrency = {value}\n")
                with self.assertRaisesRegex(ValueError, "audio_spk_emo_concurrency.*正整数"):
                    _ = settings.qwen_audio_spk_emo_concurrency

    def test_legacy_model_device_is_ignored_with_warning(self):
        settings = self._legacy_settings('[model_device]\nemotion2vec = "npu:1"\n')
        with self.assertLogs("core.config", level=logging.WARNING) as logs:
            self.assertEqual(settings.model_device_emotion2vec_devices, [])
        self.assertTrue(any("model_device_pool" in message for message in logs.output))

    def test_device_arrays_define_instances_and_order(self):
        settings = self._legacy_settings(
            '[vad]\nvad_device = ["cpu", "npu:0"]\n'
            '[model_device_pool]\n'
            'speech_campplus_devices = ["npu:1", "npu:1", "npu:2"]\n'
            'emotion2vec_devices = ["npu:0"]\n'
        )
        self.assertEqual(settings.vad_device, ["cpu", "npu:0"])
        self.assertEqual(settings.model_device_speech_campplus_devices, ["npu:1", "npu:1", "npu:2"])
        self.assertEqual(settings.model_device_emotion2vec_devices, ["npu:0"])
        self.assertEqual(settings.local_model_pool_speech_campplus_instances, 3)
        self.assertEqual(settings.local_model_pool_emotion2vec_instances, 1)

    def test_empty_arrays_disable_models(self):
        settings = self._legacy_settings(
            '[vad]\nvad_device = []\n'
            '[model_device_pool]\nspeech_campplus_devices = []\nemotion2vec_devices = []\n'
        )
        self.assertEqual(settings.vad_device, [])
        self.assertEqual(settings.model_device_speech_campplus_devices, [])
        self.assertEqual(settings.model_device_emotion2vec_devices, [])

    def test_invalid_device_arrays_are_rejected(self):
        settings = self._legacy_settings('[model_device_pool]\nspeech_campplus_devices = ["gpu:0"]\n')
        with self.assertRaises(ValueError):
            _ = settings.model_device_speech_campplus_devices
        settings = self._legacy_settings('[vad]\nvad_device = ["npu:x"]\n')
        with self.assertRaises(ValueError):
            _ = settings.vad_device

    def test_low_volume_defaults_and_boundaries(self):
        settings = self._legacy_settings("[vad]\n")
        self.assertEqual(settings.qwen_vad_snr_thres, 18.0)
        self.assertEqual(settings.qwen_vad_low_volume_rms_dbfs_thres, -40.0)
        self.assertTrue(settings.qwen_vad_low_volume_filter_enabled)
        settings = self._legacy_settings(
            "[vad]\n"
            "vad_snr_thres = 5\n"
            "vad_low_volume_rms_dbfs_thres = -42.5\n"
            "vad_low_volume_filter_enabled = true\n"
        )
        self.assertEqual(settings.vad_snr_thres, 5.0)
        self.assertEqual(settings.vad_low_volume_rms_dbfs_thres, -42.5)
        self.assertTrue(settings.vad_low_volume_filter_enabled)

    def test_legacy_qwen_vad_keys_do_not_override_vad_section(self):
        settings = self._legacy_settings(
            "[qwen_asr]\nvad_device=[\"npu:9\"]\nvad_snr_thres=99\n"
            "[vad]\nvad_device=[\"cpu\"]\nvad_snr_thres=-18\n"
        )
        self.assertEqual(settings.vad_device, ["cpu"])
        self.assertEqual(settings.vad_snr_thres, -18.0)


if __name__ == "__main__":
    unittest.main()
