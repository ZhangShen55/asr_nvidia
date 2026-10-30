"""音频裁剪的形状与输入标准化回归测试。"""

import tempfile
import unittest
from pathlib import Path

import torch

from utils.audio_utils import extract_audio_clip_array, extract_audio_clip_from_tensor


class AudioClipShapeTests(unittest.TestCase):
    def test_stereo_tensor_is_folded_to_mono(self):
        tensor = torch.stack((torch.ones(160), torch.zeros(160)))
        array, sample_rate = extract_audio_clip_array(tensor, 16000, 0.0, 0.01)
        self.assertEqual(array.shape, (160,))
        self.assertEqual(sample_rate, 16000)
        self.assertTrue((array == 0.5).all())

    def test_saved_clip_is_two_dimensional_audio(self):
        tensor = torch.stack((torch.ones(160), torch.zeros(160)))
        path = extract_audio_clip_from_tensor(tensor, 16000, 0.0, 0.01)
        try:
            try:
                import torchaudio

                audio, sample_rate = torchaudio.load(path)
                self.assertEqual(tuple(audio.shape), (1, 160))
                self.assertEqual(sample_rate, 16000)
            except ModuleNotFoundError:
                import soundfile as sf

                audio, sample_rate = sf.read(path)
                self.assertEqual(audio.shape, (160,))
                self.assertEqual(sample_rate, 16000)
        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
