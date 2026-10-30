"""v2 链路共用的请求上下文准备与临时文件清理。

`/v1.1.8/seacraft_asr` 下线后，本模块只保留 v2 需要的部分：
请求参数解析后的上下文构建、情绪标签映射与临时文件清理。
"""

import os
import time
import uuid
import logging
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from entity.data import AsrRequestParams
from utils.audio_utils import (
    preprocess_audio,
    write_audio_bytes_to_temp_file,
    load_audio_tensor,
    SERVICE_CACHE_DIR,
)
from utils.asr_stats import update_fail_task

logger = logging.getLogger(__name__)

# emotion2vec 原始标签 -> 业务情绪标签 映射
EMOTION_LABEL_MAP = {
    "中立/neutral": "平淡",
    "其他/other": "平淡",
    "<unk>": "平淡",
    "开心/happy": "积极",
    "吃惊/surprised": "兴奋",
    "生气/angry": "强调",
    "难过/sad": "思考",
    "厌恶/disgusted": "思考",
    "恐惧/fearful": "疑问",
}


@dataclass
class AsrContext:
    request: AsrRequestParams
    content: bytes
    audio_bytes: bytes
    suffix: str
    filename: str
    tmp_paths: List[str] = field(default_factory=list)
    tmp_path: str = ""
    audio_tensor: Any = None
    sample_rate: int = 0
    audio_total_s: float = 0.0
    task_id: str = ""
    load_audio_time_ms: float = 0.0


def map_emotion(label: str) -> str:
    """将 emotion2vec 原始标签映射为业务情绪标签；未知或空值归为平淡。"""
    if not label:
        return "平淡"
    return EMOTION_LABEL_MAP.get(label, "平淡")


def generate_task_id() -> str:
    return str(uuid.uuid4())


def cleanup_temp_files(tmp_paths: List[str]):
    """幂等清理当前请求登记的服务缓存文件。"""
    paths = list(dict.fromkeys(tmp_paths or []))
    # AAC 等格式转换会在同一专属前缀下产生源文件和 WAV 文件；即使转换
    # 过程中某一步异常，也要把同一请求前缀下的中间文件一并清理。
    for p in list(paths):
        if os.path.dirname(os.path.abspath(p)) == os.path.abspath(SERVICE_CACHE_DIR):
            prefix = os.path.basename(p).split(".", 1)[0]
            paths.extend(
                str(candidate)
                for candidate in Path(SERVICE_CACHE_DIR).glob(prefix + "*")
                if candidate.is_file()
            )
    for p in list(dict.fromkeys(paths)):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"删除临时文件失败: {p}, 错误: {e}")


def cleanup_stale_service_cache() -> int:
    """启动时只清除带服务专属前缀的残留文件。"""
    root = Path(SERVICE_CACHE_DIR)
    if not root.is_dir():
        return 0
    removed = 0
    for path in root.iterdir():
        if not path.name.startswith("seacraft-asr-") or not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            logger.warning("[缓存] 清理启动残留失败: %s, 错误: %s", path, exc)
    if removed:
        logger.warning("[缓存] 启动时清理服务残留 %d 个文件", removed)
    return removed


async def prepare_asr_context(
    request: AsrRequestParams,
    unique_tmp: bool = False,
) -> Tuple[Optional[dict], Optional[AsrContext]]:
    """读取并预处理音频，构建 ASR 上下文。失败时返回 (error_response, None)。

    unique_tmp=True 时使用唯一临时文件名，供 v2 链路的并发请求使用。
    """
    if request.audioFile is None:
        logger.error("音频为空")
        return {"msg": "音频文件不能为空", "code": 4001}, None

    logger.info(f"request: \n{request}")

    try:
        start_time = time.perf_counter()
        content = await request.audioFile.read()
        suffix = request.audioFile.filename.split(".")[-1].lower()
        filename = request.audioFile.filename
        if suffix == "m4a":
            suffix = "mp4"
        audio_bytes = await preprocess_audio(content, suffix, force_resample=False)
        if len(audio_bytes) < 1024:
            logger.error(
                f"音频文件过小，疑似空白音频：{request.audioFile.filename}，"
                f"大小：{len(audio_bytes)}"
            )
            update_fail_task()
            return {"msg": "音频文件内容为空，请重新录制", "code": 4006}, None
        load_audio_time_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"加载音频文件耗时：{load_audio_time_ms:.2f} ms")
    except Exception as e:
        logger.error(
            f"读取音频文件发生错误，文件名:{request.audioFile.filename}, 错误信息：\n{e}"
        )
        return {"msg": "读取音频文件发生错误", "code": 4002}, None

    tmp_paths: List[str] = []
    # VAD、裁剪和本地模型必须读取已经完成标准化的音频。此前这里误写回
    # 原始 content：立体声/非 16 kHz 音频会绕过标准化，既可能拖慢 VAD，
    # 也会让段裁剪得到多声道张量并触发 torchaudio 的 2D 校验错误。
    try:
        tmp_path = write_audio_bytes_to_temp_file(
            audio_bytes, file_name=filename, suffix=f"{suffix}", unique=unique_tmp
        )
        tmp_paths.append(tmp_path)
        audio_tensor, sample_rate = await asyncio.to_thread(load_audio_tensor, tmp_path)
    except Exception as exc:
        cleanup_temp_files(tmp_paths)
        logger.error("准备标准化音频失败，文件名:%s，错误信息：%s", filename, exc)
        update_fail_task()
        return {"msg": "读取音频文件发生错误", "code": 4002}, None
    audio_total_s = audio_tensor.shape[-1] / sample_rate if sample_rate else 0.0
    task_id = f"{generate_task_id()}_{filename}"

    ctx = AsrContext(
        request=request,
        content=content,
        audio_bytes=audio_bytes,
        suffix=suffix,
        filename=filename,
        tmp_paths=tmp_paths,
        tmp_path=tmp_path,
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        audio_total_s=audio_total_s,
        task_id=task_id,
        load_audio_time_ms=load_audio_time_ms,
    )
    return None, ctx


__all__ = [
    "AsrContext",
    "cleanup_temp_files",
    "cleanup_stale_service_cache",
    "generate_task_id",
    "map_emotion",
    "prepare_asr_context",
]
