import asyncio
import io
import unittest
import wave

import httpx
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

        async def scenario():
            def handler(request):
                body = request.content
                seen["body"] = body
                self.assertIn(b'filename="segment.wav"', body)
                self.assertIn(b"audio/wav", body)
                return httpx.Response(
                    200,
                    json={"text": "[Chinese]\n内存上传"},
                    headers={"content-type": "application/json"},
                )

            client = QwenAsrClient("http://qwen", model="model", retry_count=0)
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                text = await client.transcribe_bytes(b"RIFFmemory", filename="segment.wav")
                self.assertIn("内存上传", text)
                self.assertIn(b"RIFFmemory", seen["body"])
            finally:
                await client.aclose()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
