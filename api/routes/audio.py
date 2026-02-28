import os
import time
import shutil
import tempfile
import logging
from typing import Annotated
from fastapi import UploadFile, File, Form
from fastapi import APIRouter, UploadFile, File, Form

from utils.audio_analyze import analyze_audio_auto
from utils.audio_utils import preprocess_audio2wav, split_audio
from utils.asr_stats import update_stat
from core.models import get_whisper_model

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/audio/db_snr")
async def audio_analyze(
    audioFile: Annotated[UploadFile, File(..., description="音频文件(wav/pcm)")],
    time_size: Annotated[int, Form(description="检测粒度，单位秒")] = 10,
):
    filename = audioFile.filename
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[-1]) as tmp:
        shutil.copyfileobj(audioFile.file, tmp)
        tmp_path = tmp.name

    start_time = time.time()
    result = analyze_audio_auto(tmp_path, window_size_sec=time_size)
    end_time = time.time()

    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        logger.warning(f"临时文件已不存在：{tmp_path}")
    except Exception as e:
        logger.error(f"删除临时文件失败：{tmp_path}，错误：{e}")

    update_stat("offline")
    return {
        "result": result,
        "task_id": f"task_{filename}",
        "process_time_ms": int((end_time - start_time) * 1000),
        "timestamp": int(time.time()),
    }


@router.post("/audio/detect_mandarin")
async def audio_mandarin_detect(
    audioFile: Annotated[UploadFile, File(..., description="音频文件(wav/pcm)")],
    time_size: Annotated[int, Form(description="检测粒度，单位秒")] = 30,
):
    logger.info(f"收到音频检测请求: 文件名={audioFile.filename}, time_size={time_size}")

    tmp_paths = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audioFile.filename.split('.')[-1]}") as temp_file:
        shutil.copyfileobj(audioFile.file, temp_file)
        temp_path = temp_file.name
        tmp_paths.append(temp_path)

    start_time = time.time()
    results = []
    all_chunk_paths = []  # 记录所有 chunk 文件路径用于统一清理
    
    try:
        processed_wav = preprocess_audio2wav(temp_path)
        tmp_paths.append(processed_wav)

        # 分段
        chunks = split_audio(processed_wav, time_size)

        # 需要 whisper model
        model_whisper = get_whisper_model()
        
        for chunk_path, start_sec, end_sec in chunks:
            all_chunk_paths.append(chunk_path)
            # 语言检测（使用 whisper 的 transcribe metadata）
            _, info = model_whisper.transcribe(chunk_path, language=None)
            prob = info.language_probability or 0.0
            results.append({
                "st": start_sec,
                "ed": end_sec,
                "evaluate": "优秀" if prob >= 0.9 else "良好" if prob >= 0.6 else "一般",
                "score": int(prob * 100)
            })
    finally:
        # 统一清理所有 chunk 临时文件
        for chunk_path in all_chunk_paths:
            try:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                logger.warning(f"删除 chunk 临时文件失败: {chunk_path}, 错误: {e}")
        
        # 统一清理所有临时文件（包括原始上传文件和processed_wav）
        for p in tmp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {p}, 错误: {e}")
    
    end_time = time.time()

    top = "优秀" if max(results, key=lambda x: x["score"])["score"] >= 90 else "良好" if max(results, key=lambda x: x["score"])["score"] >= 60 else "一般"
    lowest = "优秀" if min(results, key=lambda x: x["score"])["score"] >= 90 else "良好" if min(results, key=lambda x: x["score"])["score"] >= 60 else "一般"
    avg_score = sum(item["score"] for item in results) / len(results) if results else 0
    avg = "优秀" if avg_score >= 90 else "良好" if avg_score >= 60 else "一般"

    return {
        "results": results,
        "highest": top,
        "lowest": lowest,
        "avg": avg,
        "process_time_ms": int((end_time - start_time) * 1000),
        "timestamp": int(time.time())
    }
