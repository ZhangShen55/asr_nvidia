import unittest

import numpy as np

from core.vad import VadSegment
from core.vad_volume import compute_volume_stats, filter_low_volume_segments


class VadVolumeTests(unittest.TestCase):
    def test_stats_use_dbfs_and_ten_ms_frames(self):
        audio = np.concatenate([np.zeros(160), np.ones(1600, dtype=np.float32) * 0.1])
        stats = compute_volume_stats(audio, 16000, 0, 0.11)
        self.assertEqual(stats.frame_count, 11)
        self.assertLess(stats.rms_dbfs, -18)
        self.assertAlmostEqual(stats.peak_dbfs, -20, places=2)
        self.assertEqual(stats.valid_speech_frame_count, 10)
        self.assertEqual(stats.sampled_frame_count, 11)

    def test_frame_sample_ratio_is_deterministic_and_uses_sampled_denominator(self):
        audio = np.concatenate([
            np.zeros(50 * 160, dtype=np.float32),
            np.ones(50 * 160, dtype=np.float32) * 0.1,
        ])
        stats = compute_volume_stats(
            audio, 16000, 0, 1.0, frame_sample_ratio=0.5,
            low_energy_dbfs_thres=-40,
        )
        self.assertEqual(stats.frame_count, 100)
        self.assertEqual(stats.sampled_frame_count, 50)
        self.assertAlmostEqual(stats.low_energy_frame_ratio, 0.5, places=2)

    def test_shorter_than_one_frame_still_has_one_sample(self):
        audio = np.ones(80, dtype=np.float32) * 0.1
        stats = compute_volume_stats(audio, 16000, 0, 0.005, frame_sample_ratio=0.5)
        self.assertEqual(stats.frame_count, 1)
        self.assertEqual(stats.sampled_frame_count, 1)

    def test_filter_disabled_and_enabled(self):
        audio = np.concatenate([
            np.ones(1600, dtype=np.float32) * 0.001,
            np.ones(1600, dtype=np.float32) * 0.2,
        ])
        segments = [VadSegment(0, 0.1), VadSegment(0.1, 0.2)]
        kept, dropped = filter_low_volume_segments(
            segments, audio, 16000, enabled=False, rms_dbfs_thres=-40
        )
        self.assertEqual(kept, segments)
        self.assertEqual(dropped, [])
        kept, dropped = filter_low_volume_segments(
            segments, audio, 16000, enabled=True, rms_dbfs_thres=-40
        )
        self.assertEqual(kept, [segments[1]])
        self.assertEqual(dropped[0][2], "low_volume:rms_dbfs")

    def test_dropped_segment_has_no_runtime_side_effect_contract(self):
        audio = np.ones(1600, dtype=np.float32) * 0.001
        segment = VadSegment(1.0, 1.1)
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True, rms_dbfs_thres=-40
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0][0], segment)
        self.assertTrue(dropped[0][2].startswith("low_volume:"))

    def test_decibel_filter_uses_low_energy_frame_ratio(self):
        audio = np.concatenate([
            np.zeros(1600, dtype=np.float32),
            np.ones(1600, dtype=np.float32) * 0.1,
        ])
        segments = [VadSegment(0, 0.1), VadSegment(0.1, 0.2)]
        kept, dropped = filter_low_volume_segments(
            segments,
            audio,
            16000,
            enabled=True,
            low_energy_dbfs_thres=-40,
            frame_sample_ratio=0.5,
            low_energy_frame_ratio_thres=0.8,
            decibel_enabled=True,
            rms_enabled=False,
            snr_enabled=False,
        )
        self.assertEqual(kept, [segments[1]])
        self.assertEqual(dropped[0][2], "low_volume:decibel")

    def test_decibel_filter_and_other_rules_are_independent(self):
        audio = np.ones(1600, dtype=np.float32) * 0.1
        segment = VadSegment(0, 0.1)
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True,
            low_energy_dbfs_thres=-40,
            decibel_enabled=True, rms_enabled=False, snr_enabled=False,
        )
        self.assertEqual(kept, [segment])
        self.assertEqual(dropped, [])

    def test_numeric_threshold_minus_hundred_disables_rule(self):
        audio = np.zeros(1600, dtype=np.float32)
        segment = VadSegment(0, 0.1)
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True,
            low_energy_dbfs_thres=-100,
            decibel_enabled=True, rms_enabled=False, snr_enabled=False,
        )
        self.assertEqual(kept, [segment])
        self.assertEqual(dropped, [])

    def test_snr_and_rms_rules_can_be_enabled_independently(self):
        audio = np.ones(1600, dtype=np.float32) * 0.1
        segment = VadSegment(0, 0.1)
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True,
            snr_thres=100, rms_dbfs_thres=-100,
            decibel_enabled=False, rms_enabled=False, snr_enabled=True,
        )
        self.assertEqual(kept, [])
        self.assertIn("snr", dropped[0][2])
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True,
            snr_thres=-100, rms_dbfs_thres=-10,
            decibel_enabled=False, rms_enabled=True, snr_enabled=False,
        )
        self.assertEqual(kept, [])
        self.assertIn("rms_dbfs", dropped[0][2])
        kept, dropped = filter_low_volume_segments(
            [segment], audio, 16000, enabled=True,
            rms_dbfs_thres=-100, snr_thres=-100,
            decibel_enabled=True, rms_enabled=False, snr_enabled=False,
        )
        self.assertEqual(kept, [segment])
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
