import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core import vad as vad_module
from core.config import settings


class _FakeVadPool:
    def __init__(self, spans=None):
        self.spans = spans or [[10000, 11000], [300000, 301000], [590000, 591000]]

    async def detect(self, _audio_path):
        return [{"value": self.spans}]


class VadSegmentFilteringTests(unittest.TestCase):
    def test_default_scope_filters_only_beginning_and_end(self):
        audio = np.zeros(600 * 16000, dtype=np.float32)
        old = settings._cfg
        settings._cfg = {
            "vad": {
                "min_segment_duration_s": 0.3,
                "vad_decibel_thres_filter_enabled": True,
                "vad_segment_decibel_thres": -40.0,
                "vad_decibel_frame_sample_ratio": 0.5,
                "vad_decibel_low_frame_ratio_thres": 0.8,
                "vad_snr_thres_filter_enabled": False,
                "vad_low_volume_filter_enabled": False,
                "after_start": 3,
                "before_end": 3,
            }
        }
        try:
            async def scenario(path):
                result = await vad_module.detect_segments(
                    path, audio_tensor=audio, sample_rate=16000,
                )
                return result, vad_module.get_vad_stats()
            with tempfile.NamedTemporaryFile(suffix=".wav") as handle, \
                    patch.object(vad_module, "get_vad_pool", return_value=_FakeVadPool()):
                result, stats = asyncio.run(scenario(handle.name))
            self.assertEqual([(s.start_s, s.end_s) for s in result], [(300.0, 301.0)])
            self.assertEqual(stats["range_candidates"], 2)
            self.assertEqual(stats["decibel_dropped"], 2)
            self.assertEqual(stats["kept_segments"], 1)
        finally:
            settings._cfg = old

    def test_zero_scope_filters_every_segment(self):
        audio = np.zeros(600 * 16000, dtype=np.float32)
        old = settings._cfg
        settings._cfg = {
            "vad": {
                "min_segment_duration_s": 0.3,
                "vad_decibel_thres_filter_enabled": True,
                "vad_segment_decibel_thres": -40.0,
                "vad_decibel_frame_sample_ratio": 1.0,
                "vad_decibel_low_frame_ratio_thres": 0.8,
                "vad_snr_thres_filter_enabled": False,
                "vad_low_volume_filter_enabled": False,
                "after_start": 0,
                "before_end": 0,
            }
        }
        try:
            async def scenario(path):
                result = await vad_module.detect_segments(
                    path, audio_tensor=audio, sample_rate=16000,
                )
                return result, vad_module.get_vad_stats()
            with tempfile.NamedTemporaryFile(suffix=".wav") as handle, \
                    patch.object(vad_module, "get_vad_pool", return_value=_FakeVadPool()):
                result, stats = asyncio.run(scenario(handle.name))
            self.assertEqual(result, [])
            self.assertEqual(stats["range_candidates"], 3)
        finally:
            settings._cfg = old

    def test_missing_audio_tensor_is_loaded_before_filtering(self):
        old = settings._cfg
        settings._cfg = {
            "vad": {
                "min_segment_duration_s": 0.3,
                "vad_decibel_thres_filter_enabled": True,
                "vad_segment_decibel_thres": -40.0,
                "vad_decibel_frame_sample_ratio": 1.0,
                "vad_decibel_low_frame_ratio_thres": 0.8,
                "vad_snr_thres_filter_enabled": False,
                "vad_low_volume_filter_enabled": False,
                "after_start": 0,
                "before_end": 0,
            }
        }
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
                sf.write(handle.name, np.zeros(1600, dtype=np.float32), 16000)
                async def scenario():
                    return await vad_module.detect_segments(handle.name)
                with patch.object(vad_module, "get_vad_pool", return_value=_FakeVadPool([[0, 1000]])):
                    result = asyncio.run(scenario())
            self.assertEqual(result, [])
        finally:
            settings._cfg = old


if __name__ == "__main__":
    unittest.main()
