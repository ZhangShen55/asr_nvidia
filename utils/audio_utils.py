import os
import uuid
import asyncio
import time
import logging
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
    from pydub import AudioSegment

    audio = AudioSegment.from_file(BytesIO(audio_bytes), format=suffix)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)

    # 导出成 WAV 容器，不能是 raw_data
    buf = BytesIO()
    audio.export(buf, format="wav")          # 默认 PCM_16
    return buf.getvalue()

async def preprocess_audio(audio_bytes: bytes, suffix: str, force_resample: bool = False) -> bytes:
    start = time.perf_counter()
    result = await asyncio.to_thread(standardize_audio, audio_bytes, suffix, force_resample)
    duration = (time.perf_counter() - start) * 1000
    logger.info(f"[音频预处理] 耗时：{duration:.2f}ms")
    return result

# 分割音频
def crop_audio(audio_data:torch.Tensor, start_time, end_time, sample_rate):
    start_sample = int(start_time * sample_rate / 1000)  # 转换为样本数
    end_sample = int(end_time * sample_rate / 1000)  # 转换为样本数
    return audio_data[:,start_sample:end_sample]



def extract_audio_clip(input_path: str, start_time: float, duration: float, suffix=".wav") -> str:
    """
    用 ffmpeg 截取一段音频片段，返回新文件路径
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
    if getattr(array, "ndim", 1) != 1:
        array = array.reshape(-1)
    buffer = BytesIO()
    sf.write(buffer, array, int(sample_rate), format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def extract_audio_clip_array(audio_tensor, sample_rate: int, start_time: float, duration: float):
    """从已加载的音频张量裁剪出一段，返回 (float32 numpy 数组, 采样率)。

    本地模型（CAM++）直接吃 numpy 数组，比走文件路径快约 2 倍：
    funasr 的文件分支里还包含一次额外的音频读取与重采样开销。
    """
    start_sample = max(0, int(start_time * sample_rate))
    end_sample = min(audio_tensor.shape[-1], int((start_time + duration) * sample_rate))
    if end_sample <= start_sample:
        raise ValueError("裁剪区间为空")

    tensor = audio_tensor
    if getattr(tensor, "ndim", 0) == 1:
        clipped = tensor[start_sample:end_sample]
    elif getattr(tensor, "ndim", 0) == 2:
        # torchaudio 使用 [channels, samples]；裁剪后统一折叠为单声道，
        # 避免立体声张量在 torchaudio.save 中变成三维 [1, C, T]。
        clipped = tensor[:, start_sample:end_sample].mean(dim=0)
    else:
        raise ValueError(f"音频张量必须是 1D 或 2D，实际维度为 {getattr(tensor, 'ndim', None)}")
    array = clipped.detach().cpu().numpy().reshape(-1).astype("float32")
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
        prefix = f"seacraft-asr-v2-{uuid.uuid4().hex}-"
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
    if len(chunks) >= 2 and (chunks[-1][1] - chunks[-1][0]) < min_last_s:
        prev_start, _ = chunks[-2]
        _, last_end = chunks[-1]
        chunks[-2] = (prev_start, last_end)
        chunks.pop()

    return chunks
