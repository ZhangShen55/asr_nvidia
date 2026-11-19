"""浮点秒端点统一采样格；实际数组/内存WAV/张量保存路径不得差一个样本。"""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np
import soundfile as sf
import torch

from utils.audio_utils import (audio_sample_bounds, crop_audio, extract_audio_clip_array,
                              extract_audio_clip_from_tensor, encode_audio_clip_wav)


def frames(payload):
    with wave.open(BytesIO(payload),'rb') as reader:
        assert (reader.getframerate(),reader.getnchannels(),reader.getsampwidth())==(16000,1,2)
        return reader.getnframes()


class RoundedAudioCropTests(unittest.TestCase):
    def setUp(self):
        self.array=np.linspace(-0.25,0.25,80000,dtype=np.float32)
        self.tensor=torch.from_numpy(self.array).reshape(1,-1)

    def assert_channels(self,start,end,expected):
        original=(start,end)
        duration=end-start
        left,right=audio_sample_bounds(start,end,16000,len(self.array))
        self.assertEqual(right-left,expected)
        arrays=[]
        for source in [self.array,self.tensor,np.stack([self.array,self.array]),torch.cat([self.tensor,self.tensor])]:
            clipped,rate=extract_audio_clip_array(source,16000,start,duration)
            self.assertEqual(len(clipped),expected)
            self.assertEqual(clipped.dtype,np.float32)
            np.testing.assert_array_equal(clipped,self.array[left:right])
            self.assertEqual(frames(encode_audio_clip_wav(clipped,rate)),expected)
            arrays.append(clipped)
        raw=crop_audio(self.tensor,start*1000,end*1000,16000)
        self.assertEqual(raw.shape[1],expected)
        np.testing.assert_array_equal(raw.numpy().reshape(-1),arrays[0])
        path=extract_audio_clip_from_tensor(self.tensor,16000,start,duration)
        try:
            # 既有保存路径可用FLOAT WAV；这里只核对采样格，不借边界修复更改编码。
            with sf.SoundFile(path) as stored:
                self.assertEqual((stored.samplerate,stored.channels),(16000,1))
                self.assertEqual(len(stored),expected)
        finally:Path(path).unlink(missing_ok=True)
        self.assertEqual((start,end),original)  # 不调整对外VAD bg/ed。

    def test_nonzero_start_299_300_301_and_999_1000_1001_ms(self):
        for start in [.001,.03,.1,.317]:
            for millis in [299,300,301,999,1000,1001]:
                with self.subTest(start=start,millis=millis):
                    self.assert_channels(start,start+millis/1000,millis*16)

    def test_floating_endpoints_regressions(self):
        self.assert_channels(.001,1.001,16000)
        self.assert_channels(.03,.32999999999999996,4800)

    def test_eof_and_negative_start_clamp_without_padding(self):
        self.assert_channels(4.7,5.3,4800)
        self.assert_channels(-.03,.3,4800)
        self.assert_channels(4.999,5.3,16)
        for start,end in [(5,5.3),(6,7),(.3,.2),(-1,-.1)]:
            with self.subTest(start=start,end=end):
                with self.assertRaisesRegex(ValueError,'裁剪区间为空'):
                    extract_audio_clip_array(self.tensor,16000,start,end-start)

    def test_nonfinite_endpoint_fails_explicitly(self):
        for start,end in [(float('nan'),1),(0,float('inf'))]:
            with self.assertRaisesRegex(ValueError,'有限秒值'):audio_sample_bounds(start,end,16000,len(self.array))

    def test_curve_manifest_uses_identical_sample_bounds_and_preserves_overrun_guard(self):
        from scripts.benchmark_qwen_batch_curve import read_manifest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.wav';manifest=root/'manifest.json'
            sf.write(source,self.array,16000,format='WAV',subtype='PCM_16')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            encoded=base64.b64encode(bytes(source)).decode()
            rows=[{'id':f'{request}-{segment}','request_id':str(request),'segment_index':segment,
                   'path_bytes_b64':encoded,'source_sha256':digest,'start_s':.03,'end_s':.32999999999999996}
                  for request in range(8) for segment in range(16)]
            manifest.write_text(json.dumps({'segments':rows}))
            prepared=read_manifest(manifest)
            self.assertEqual(len(prepared),128)
            self.assertTrue(all(len(row['pcm'])==4800 and frames(row['wav'])==4800 for row in prepared))
            self.assertTrue(all(row['start_s']==.03 and row['end_s']==.32999999999999996 for row in prepared))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            rows[0]['end_s']=5.1;manifest.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'超出完整源音频'):read_manifest(manifest)
