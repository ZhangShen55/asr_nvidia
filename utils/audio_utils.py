import os
import uuid
import asyncio
import time
import logging
import math
import struct
import numpy as np
import torch
from io import BytesIO
import soundfile as sf
from fastapi import File, Form

import tempfile
import subprocess

SERVICE_CACHE_DIR = os.path.join(tempfile.gettempdir(), "seacraft_asr")


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("audio")

def load_audio_tensor(path: str):
    """
    加载音频为 tensor。优先 ffmpeg 后端；容器内 torchaudio 若未编译 ffmpeg 则回退 soundfile。
    """
    import torchaudio

    try:
        return torchaudio.load(path, backend="ffmpeg")
    except ValueError:
        return torchaudio.load(path)


def check_audio_format(audio_bytes: bytes) -> dict:
    try:
        with sf.SoundFile(BytesIO(audio_bytes)) as f:
            return {
                "samplerate": f.samplerate,
                "channels": f.channels,
                "subtype": f.subtype
            }
    except Exception:
        return {}


def standardize_audio(audio_bytes: bytes, suffix: str, force_resample: bool = False) -> bytes:
    info = check_audio_format(audio_bytes)
    need_resample = force_resample or not (
        info.get("samplerate") == 16000 and
        info.get("channels") == 1 and
        info.get("subtype") == "PCM_16"
    )

    if not need_resample:
        logger.info("[音频检查] 已符合要求，跳过重采样")
        return audio_bytes

    logger.info(f"[音频检查] 转换前参数：{info}，开始重采样")
    # 目标环境锁定PyAV，不依赖宿主PATH中的ffmpeg二进制；全程内存解码。
    import av
    import wave

    pcm = BytesIO()
    with av.open(BytesIO(audio_bytes)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            for converted in resampler.resample(frame):
                pcm.write(converted.to_ndarray().astype("<i2", copy=False).tobytes())
        for converted in resampler.resample(None):
            pcm.write(converted.to_ndarray().astype("<i2", copy=False).tobytes())
    buf = BytesIO()
    with wave.open(buf, "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", ""))
        output.writeframes(pcm.getvalue())
    return buf.getvalue()

async def preprocess_audio(audio_bytes: bytes, suffix: str, force_resample: bool = False) -> bytes:
    start = time.perf_counter()
    result = await asyncio.to_thread(standardize_audio, audio_bytes, suffix, force_resample)
    duration = (time.perf_counter() - start) * 1000
    logger.info(f"[音频预处理] 耗时：{duration:.2f}ms")
    return result

def audio_sample_bounds(start_time: float, end_time: float, sample_rate: int, total_samples: int):
    """将秒端点分别round到采样格后限制到源范围；与VAD/路由门限口径一致。

    必须分别round起止，不能int截断，也不能用round(duration*sr)替代端点差；
    毫秒格点转浮点秒会出现1.001*16000略小于整数的表示误差。EOF只裁短，绝不补零。
    """
    start, end = float(start_time), float(end_time)
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("裁剪端点必须是有限秒值")
    if sample_rate <= 0 or total_samples < 0:
        raise ValueError("采样率必须为正且源样本数非负")
    left = max(0, min(total_samples, round(start * sample_rate)))
    right = max(0, min(total_samples, round(end * sample_rate)))
    if right <= left:
        raise ValueError("裁剪区间为空")
    return left, right


# 旧调用方的毫秒端点接口与正式内存裁剪使用同一采样边界。
def crop_audio(audio_data: torch.Tensor, start_time, end_time, sample_rate):
    left, right = audio_sample_bounds(start_time / 1000, end_time / 1000,
                                     sample_rate, audio_data.shape[len(audio_data.shape) - 1])
    return audio_data[:, left:right]



def extract_audio_clip(input_path: str, start_time: float, duration: float, suffix=".wav") -> str:
    """
    旧FFmpeg兼容通道：按秒转码并返回路径，正式v3使用下方内存采样格裁剪。

    压缩文件/非WAV转码不保证逐样本等长；不为此旧通道逐段重读并标准化整条大文件。
    """
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-loglevel", "error",
        output_file
    ]
    subprocess.run(command, check=True)
    return output_file


logger = logging.getLogger(__name__)


def extract_audio_clip_from_tensor(audio_tensor, sample_rate: int, start_time: float, duration: float) -> str:
    """从已加载的音频张量裁剪出一段 16kHz 单声道 WAV，返回临时文件路径。

    相比逐段调用 ffmpeg（实测 118 段 7.97s），张量切片仅需 0.25s。
    """
    array, sample_rate = extract_audio_clip_array(
        audio_tensor, sample_rate, start_time, duration
    )
    return save_audio_clip_array(array, sample_rate)


def save_audio_clip_array(array, sample_rate: int) -> str:
    """把已经裁剪并折叠为单声道的一维数组保存为 WAV。"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_file = tmp.name
    try:
        import torchaudio

        torchaudio.save(output_file, torch.from_numpy(array).reshape(1, -1), sample_rate)
    except ModuleNotFoundError:
        sf.write(output_file, array, int(sample_rate), format="WAV", subtype="PCM_16")
    return output_file


def encode_audio_clip_wav(array, sample_rate: int) -> bytes:
    """把单声道 float32 样本编码为内存中的 PCM16 WAV。"""
    # PCM16源的样本在float32里精确落于1/32768格点；该分支不近似量化。
    # 非格点/非原生dtype/非单声道及特殊采样率继续使用原libsndfile路径。
    if (
        type(array) is np.ndarray
        and array.ndim == 1
        and array.dtype == np.dtype("float32")
        and array.dtype.isnative
        and type(sample_rate) is int
        and 0 < sample_rate <= 0x7FFFFFFF
        and array.size * 2 <= 0xFFFFFFFF - 36
        and np.isfinite(array).all()
        and (array >= -1).all()
        and (array <= 1).all()
    ):
        scaled = array * np.float32(32768)
        integral = scaled.astype(np.int32)
        exact_grid = np.array_equal(scaled, integral)
        del scaled
        if exact_grid:
            # +1与libsndfile一致映射至32767；不修改调用方的CAM/emo原数组。
            np.clip(integral, -32768, 32767, out=integral)
            pcm = integral.astype("<i2").tobytes()
            del integral
            data_size = len(pcm)
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + data_size,
                b"WAVE", b"fmt ", 16, 1, 1, sample_rate,
                sample_rate * 2, 2, 16, b"data", data_size,
            )
            return header + pcm
        del integral
    if getattr(array, "ndim", 1) != 1:
        array = array.reshape(-1)
    buffer = BytesIO()
    sf.write(buffer, array, int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def extract_audio_clip_array(audio_tensor, sample_rate: int, start_time: float, duration: float):
    """从已加载的tensor或numpy数组裁剪，返回(float32 numpy数组, 采样率)。

    本地模型（CAM++）直接吃 numpy 数组，比走文件路径快约 2 倍：
    funasr 的文件分支里还包含一次额外的音频读取与重采样开销。
    """
    # Cython 的 boundscheck=False 会把部分动态序列的负索引优化成越界访问；
    # 显式取最后一个维度，保证源码和扩展运行时一致。
    shape = getattr(audio_tensor, "shape", ())
    if not shape:
        raise ValueError("音频张量缺少 shape")
    start_sample, end_sample = audio_sample_bounds(
        start_time, start_time + duration, sample_rate, shape[len(shape) - 1]
    )

    tensor = audio_tensor
    if getattr(tensor, "ndim", 0) == 1:
        clipped = tensor[start_sample:end_sample]
    elif getattr(tensor, "ndim", 0) == 2:
        # torchaudio 使用 [channels, samples]；裁剪后统一折叠为单声道，
        # 避免立体声张量在 torchaudio.save 中变成三维 [1, C, T]。
        channels = tensor[:, start_sample:end_sample]
        clipped = channels.mean(dim=0) if hasattr(channels, "detach") else channels.mean(axis=0)
    else:
        raise ValueError(f"音频张量必须是 1D 或 2D，实际维度为 {getattr(tensor, 'ndim', None)}")
    if hasattr(clipped, "detach"):
        clipped = clipped.detach().cpu().numpy()
    array = clipped.reshape(-1).astype("float32")
    return array, sample_rate


def read_audio_array(path: str):
    """读取音频文件为 (float32 numpy 数组, 采样率)。"""
    array, sample_rate = sf.read(path, dtype="float32")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, sample_rate


def write_audio_bytes_to_temp_file(audio_bytes: bytes, file_name: str, suffix=".mp3", unique: bool = False) -> str:
    """

    file_name：上游传过来的文件名（带后缀）

    将音频字节写入临时文件，若后缀为 .aac 则转换为 16kHz 单声道 WAV

    Args:
        audio_bytes: 音频字节数据
        suffix: 文件后缀，默认为 .mp3
        unique: 是否为并发请求生成唯一文件名，避免同名文件互相覆盖

    Returns:
        临时文件路径
    """
    if not suffix.startswith("."):
        suffix = "." + suffix

    os.makedirs(SERVICE_CACHE_DIR, exist_ok=True)
    # 服务缓存只写入专属目录和专属前缀，便于精确清理和重启扫描。
    safe_name = os.path.basename(file_name or "upload")
    prefix = f"seacraft-asr-{uuid.uuid4().hex}-"
    if unique:
        prefix = f"seacraft-asr-v3-{uuid.uuid4().hex}-"
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, prefix=prefix, dir=SERVICE_CACHE_DIR
    ) as tmp:
        tmp.write(audio_bytes)
        original_file = tmp.name

    # 如果是 aac 格式，转换为 16kHz 单声道 WAV
    if suffix.lower() == ".aac":
        converted_file = os.path.splitext(original_file)[0] + ".wav"
        
        try:
            # 使用 ffmpeg 进行转换
            subprocess.run([
                "ffmpeg", "-y",  # -y 表示覆盖已存在文件
                "-i", original_file,
                "-ar", "16000",  # 设置采样率为 16kHz
                "-ac", "1",      # 设置为单声道
                "-f", "wav",     # 输出格式为 WAV
                converted_file
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            logger.info(f"Converted AAC to WAV: {original_file} -> {converted_file}")
            
            # 删除原始 AAC 文件
            os.remove(original_file)
            return converted_file
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert AAC to WAV: {e.stderr.decode()}")
            # 转换失败时返回原始文件
            return original_file
    
    return original_file



def plan_audio_chunks(
    duration_s: float,
    chunk_minutes: float = 60,
    min_last_minutes: float = 15,
    overlap_s: float = 15,
) -> list:
    """
    规划长音频分块方案，返回 [(clean_start_s, clean_end_s), ...] 列表。

    规则：
    - 按 chunk_minutes 等分，最后一块不足 min_last_minutes 则并入倒数第二块；
    - clean_start/clean_end 为该块对外"有效区间"，不含 overlap；
    - 实际送进模型的音频区间由调用方根据 overlap_s 扩展：
        actual_start = max(0, clean_start - overlap_s)   (非第一块)
        actual_end   = min(duration_s, clean_end + overlap_s) (非最后块)
    """
    chunk_s = chunk_minutes * 60
    min_last_s = min_last_minutes * 60

    chunks = []
    pos = 0.0
    while pos < duration_s:
        end = min(pos + chunk_s, duration_s)
        chunks.append((pos, end))
        if end >= duration_s:
            break
        pos = end

    # 最后一块过短则合并到倒数第二块
    if len(chunks) >= 2:
        last_chunk = chunks[len(chunks) - 1]
        previous_chunk = chunks[len(chunks) - 2]
        if (last_chunk[1] - last_chunk[0]) >= min_last_s:
            return chunks
        prev_start, _ = previous_chunk
        _, last_end = last_chunk
        chunks[len(chunks) - 2] = (prev_start, last_end)
        chunks.pop()

    return chunks
