"""Run unchanged against source or the compiled utils.audio_utils extension."""
from io import BytesIO
import struct
from unittest.mock import patch
import numpy as np
import pytest
import soundfile as sf
import utils.audio_utils as audio_module
from utils.audio_utils import encode_audio_clip_wav

COMPILED = str(audio_module.__file__).endswith(".so")

def reference(array, rate):
    if getattr(array,'ndim',1)!=1:array=array.reshape(-1)
    buffer=BytesIO();sf.write(buffer,array,int(rate),format='WAV',subtype='PCM_16')
    return buffer.getvalue()

def compare(array,rate,fast):
    original=array.copy()if hasattr(array,'copy')else None
    expected=reference(array,rate)
    with patch.object(sf,'write',wraps=sf.write)as writer:
        actual=encode_audio_clip_wav(array,rate)
        assert writer.call_count==(0 if fast else 1)
    assert actual==expected
    if original is not None:np.testing.assert_array_equal(array,original)
    return actual

def test_every_int16_grid_value_and_saturation_endpoint_are_byte_exact():
    grid=np.arange(-32768,32768,dtype=np.int32).astype(np.float32)/np.float32(32768)
    actual=compare(grid,16000,True)
    assert len(actual)==44+65536*2
    assert actual[44:]==np.arange(-32768,32768,dtype='<i2').tobytes()
    compare(np.array([-1.,-0.,0.,1.],dtype=np.float32),16000,True)

@pytest.mark.parametrize('rate',[1,8000,16000,22050,44100,48000,192000,2147483647])
def test_native_grid_rates_empty_and_noncontiguous(rate):
    compare(np.array([],dtype=np.float32),rate,True)
    compare(np.array([-1,0,.5,1],dtype=np.float32)[::-1],rate,True)

@pytest.mark.parametrize('array',[
    np.array([.1,.2,.3],dtype=np.float32),
    np.array([np.nextafter(np.float32(1/32768),np.float32(1))],dtype=np.float32),
    np.array([-2,-1,0,1,2],dtype=np.float32),
    np.array([np.nan,np.inf,-np.inf,0],dtype=np.float32),
    np.array([0,.5],dtype=np.float64),np.array([0,1],dtype=np.int16),
    np.array([[0,.5]],dtype=np.float32),np.array([0,.5],dtype='>f4'),[0.,.5],
])
def test_other_inputs_keep_original_soundfile_bytes(array):compare(array,16000,False)

def test_random_non_grid_values_keep_original_soundfile_bytes():
    rng=np.random.default_rng(20261119)
    for size in(1,7,127,4800,128160):compare(rng.uniform(-1,1,size).astype(np.float32),16000,False)

@pytest.mark.parametrize('rate',[0,-1,2147483648,None,float('nan'),'bad'])
def test_invalid_rates_keep_original_error_type(rate):
    array=np.zeros(1,dtype=np.float32)
    # Existing protected sample_rate:int rejects off-type arguments before the function body.
    if COMPILED and type(rate) is not int:
        with patch.object(sf,'write',wraps=sf.write)as writer:
            with pytest.raises(TypeError,match="sample_rate.*expected int"):
                encode_audio_clip_wav(array,rate)
            assert writer.call_count==0
        return
    try:reference(array,rate)
    except Exception as expected:
        with patch.object(sf,'write',wraps=sf.write):
            with pytest.raises(type(expected)):encode_audio_clip_wav(array,rate)
    else:pytest.fail('Fixture rate unexpectedly accepted')

@pytest.mark.parametrize('rate',[16000.0,'16000',np.int64(16000),True])
def test_non_builtin_integer_rate_preserves_existing_source_or_protected_contract(rate):
    array=np.array([0,.5],dtype=np.float32)
    if COMPILED:
        with patch.object(sf,'write',wraps=sf.write)as writer:
            with pytest.raises(TypeError,match="sample_rate.*expected int"):
                encode_audio_clip_wav(array,rate)
            assert writer.call_count==0
    else:
        compare(array,rate,False)
