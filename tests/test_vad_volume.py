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


if __name__ == "__main__":
    unittest.main()
