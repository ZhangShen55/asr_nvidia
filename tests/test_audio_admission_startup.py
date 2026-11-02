import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from core.config import Settings


class AudioAdmissionStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_limit_fails_before_vad_or_model_loading(self):
        import main

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.toml"
            config.write_text("[qwen_asr]\naudio_spk_emo_concurrency = 0\n")
            with patch.object(main, "settings", Settings(config_path=str(config))), \
                    patch.object(main, "start_vad_pool", new_callable=AsyncMock) as vad, \
                    patch("core.models.ModelPool.ensure_loaded", new_callable=AsyncMock) as models:
                with self.assertRaisesRegex(ValueError, "audio_spk_emo_concurrency"):
                    async with main.lifespan(main.create_app()):
                        self.fail("invalid configuration must not become ready")
                vad.assert_not_awaited()
                models.assert_not_awaited()
