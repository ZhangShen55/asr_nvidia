"""真实PyAV内存重采样，不依赖外部ffmpeg命令。"""
import io
import wave
from unittest.mock import patch
import numpy as np
import pytest
from utils.audio_utils import standardize_audio


def wav_bytes(rate, channels, frames):
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setparams((channels, 2, rate, 0, 'NONE', ''))
        audio.writeframes(frames)
    return output.getvalue()


def test_compliant_pcm_remains_byte_identical():
    data = wav_bytes(16000, 1, np.arange(1600, dtype='<i2').tobytes())
    assert standardize_audio(data, 'wav') is data


def test_resampling_uses_memory_and_preserves_duration_without_cli():
    source = (np.sin(np.arange(8000) * (2 * np.pi * 220 / 8000)) * 6000).astype('<i2')
    stereo = np.stack((source, source), axis=1).tobytes()
    with patch('subprocess.run', side_effect=AssertionError('must not invoke ffmpeg CLI')):
        output = standardize_audio(wav_bytes(8000, 2, stereo), 'wav')
    with wave.open(io.BytesIO(output), 'rb') as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (16000, 1, 2)
        assert audio.getnframes() == 16000
        samples = np.frombuffer(audio.readframes(16000), dtype='<i2')
        assert np.sqrt(np.square(samples.astype('float64')).mean()) > 3000


def test_corrupt_stream_is_not_disguised_as_success():
    with pytest.raises(Exception):
        standardize_audio(b'not-an-audio-stream', 'mp3')
