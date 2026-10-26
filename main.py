from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from core.config import settings
from core.logging import setup_logging
from core.qwen_asr_client import close_qwen_client
from core.vad_pool import start_vad_pool, stop_vad_pool
from api.routes.asr_v2 import router as asr_v2_router
from api.routes.status import router as status_router
from utils.asr_stats import reset_stats

# 服务进程的统一进程名：nvidia-smi 的进程列表读 /proc/<pid>/cmdline 的 argv[0]，
# 覆写后运维能直接定位「哪一行是本服务占的显存」。
# 本地模型（VAD / CAM++ / emotion2vec）同进程承载，因此不再按模型拆分进程名。
PROCESS_NAME = "asr-offline-model"


def _set_process_name() -> None:
    """把进程名覆写为 PROCESS_NAME（失败不影响服务启动）。"""
    try:
        import setproctitle

        setproctitle.setproctitle(PROCESS_NAME)
    except Exception as exc:  # noqa: BLE001 - 进程名只是运维标识，不该阻塞启动
        print(f"[WARN] 设置进程名失败: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) 进程命名（运维在 nvidia-smi 中据此定位本服务的显存占用）
    _set_process_name()

    # 2) 日志初始化
    setup_logging(settings.log_path)

    # 3) 重置统计
    reset_stats()

    # 4) 记录启动时刻
    app.state.run_start_time = datetime.utcnow()

    # 5) 预热 VAD worker 进程池：全部核心 worker 就绪后服务才对外就绪，
    #    避免第一个请求承担 FSMN-VAD 的模型加载延迟
    await start_vad_pool()

    try:
        yield
    finally:
        # 释放 VAD worker 进程池与 Qwen 客户端连接池
        await stop_vad_pool()
        await close_qwen_client()


def create_app() -> FastAPI:
    app = FastAPI(title="SeaCraftASR", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由装配（路径与契约保持不变）
    app.include_router(status_router)
    app.include_router(asr_v2_router)

    return app


app = create_app()

# 方便本地调试：容器里用 start.sh，多进程由 Nginx + 多 uvicorn 实例负责
if __name__ == "__main__":
    import uvicorn

    _set_process_name()
    uvicorn.run("main:app", host="0.0.0.0", port=8083, reload=False)
