import asyncio
import io
import unittest
from unittest.mock import patch
import wave

import numpy as np

from core.qwen_asr_client import QwenAsrClient
from utils.audio_utils import encode_audio_clip_wav


class MemoryClipTests(unittest.TestCase):
    def test_wav_bytes_are_pcm16_mono_and_decodable(self):
        samples = np.linspace(-0.25, 0.25, 1600, dtype=np.float32)
        payload = encode_audio_clip_wav(samples, 16000)
        with wave.open(io.BytesIO(payload), "rb") as reader:
            self.assertEqual(reader.getframerate(), 16000)
            self.assertEqual(reader.getnchannels(), 1)
            self.assertEqual(reader.getsampwidth(), 2)
            self.assertEqual(reader.getnframes(), len(samples))

    def test_qwen_uploads_bytes_without_a_path(self):
        seen = {}

        class Runtime:
            failure = None

            def __init__(self, *args):
                pass

            def transcribe(self, payload, language):
                seen["body"] = payload
                seen["language"] = language
                return "language Chinese<asr_text>内存上传"

            def close(self):
                seen["closed"] = True

        async def scenario():
            payload = encode_audio_clip_wav(np.zeros(1600, dtype=np.float32), 16000)
            client = QwenAsrClient("http://legacy-unused", model="model", retry_count=0,
                                   generation_mode="legacy", local_instance_count=1,
                                   execution_mode="thread", runtime_factory=Runtime)
            try:
                # filename remains a compatible keyword; the local runner receives
                # the same in-memory WAV bytes and never requires a segment path.
                with patch("builtins.open", side_effect=AssertionError("段级推理不得落盘")), \
                     patch("tempfile.NamedTemporaryFile", side_effect=AssertionError("段级推理不得创建临时文件")):
                    text = await client.transcribe_bytes(payload, filename="segment.wav")
                self.assertEqual(text, "language Chinese<asr_text>内存上传")
                self.assertIs(seen["body"], payload)
                self.assertTrue(seen["body"].startswith(b"RIFF"))
                self.assertIsNone(seen["language"])
            finally:
                await client.aclose()
            self.assertTrue(seen["closed"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
