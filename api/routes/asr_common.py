"""v2 链路共用的请求上下文准备与临时文件清理。

`/v1.1.8/seacraft_asr` 下线后，本模块只保留 v2 需要的部分：
请求参数解析后的上下文构建、情绪标签映射与临时文件清理。
"""

import os
import time
import uuid
import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

from entity.data import AsrRequestParams
from utils.audio_utils import (
    preprocess_audio,
    write_audio_bytes_to_temp_file,
    load_audio_tensor,
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
    """统一清理临时文件"""
    for p in tmp_paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"删除临时文件失败: {p}, 错误: {e}")


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
    tmp_path = write_audio_bytes_to_temp_file(
        content, file_name=filename, suffix=f"{suffix}", unique=unique_tmp
    )
    tmp_paths.append(tmp_path)

    audio_tensor, sample_rate = await asyncio.to_thread(load_audio_tensor, tmp_path)
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
    "generate_task_id",
    "map_emotion",
    "prepare_asr_context",
]
