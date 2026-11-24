import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

# 在Settings读取缺省路径及任何模型模块导入之前拒绝容器配置兜底。
from core.deployment import validate_npu_runtime, validate_source_container_config
validate_source_container_config()

from core.config import settings
from core.logging import setup_logging
from core.qwen_asr_client import close_qwen_client, preload_qwen_client, get_qwen_client
from core.models import close_v3_model_pools, preload_v3_model_pools
from core.qwen_asr_queue import get_slot_manager
from core.vad_pool import start_vad_pool, stop_vad_pool
from core.async_utils import await_completion
from core.resource_diagnostics import ResourceSnapshotSampler
from api.routes.asr_v3 import router as asr_v3_router
from api.routes.asr_common import cleanup_stale_service_cache
from api.routes.status import router as status_router
from utils.asr_stats import reset_stats

# 主服务与独立 VAD worker 使用不同名称，设备资源按进程分别归因。
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
    app.state.shutdown_failed = False
    # 配置错误在任何模型预热和对外就绪之前暴露。
    _ = settings.qwen_audio_spk_emo_concurrency
    settings.validate_model_backends()
    app.state.deployment_preflight = validate_npu_runtime(settings)

    # 命名仅用于运维识别，不代表包含独立 VAD worker 的占用。
    _set_process_name()

    # 2) 日志初始化
    setup_logging(settings.log_path)

    # 只扫描并清理服务专属缓存目录，避免误删其它应用的临时文件。
    cleanup_stale_service_cache()

    # 3) 重置统计
    reset_stats()

    # 4) 记录启动时刻
    app.state.run_start_time = datetime.utcnow()
    # 只构造本app缓存，不启动NPU/procfs I/O；首个GET才发起一次后台采样。
    sampler = ResourceSnapshotSampler(interval_s=settings.resource_sample_interval_s)
    app.state.resource_sampler = sampler

    # 5) 预热 VAD worker 进程池：全部核心 worker 就绪后服务才对外就绪，
    #    避免第一个请求承担 FSMN-VAD 的模型加载延迟
    async def shutdown():
        failures = []
        # 先发布closing，重资源I/O与模型收尾并行；不能等npu-smi/procfs才归还模型。
        sampler_close = asyncio.create_task(sampler.aclose())
        qwen_close = (asyncio.create_task(close_qwen_client())
                      if settings.qwen_execution_mode == "process" else None)
        models_close = (asyncio.create_task(close_v3_model_pools())
                        if getattr(settings, "model_device_speech_campplus_execution_mode", "thread") == "process"
                        else None)
        try:
            await asyncio.sleep(0)
            closes = [stop_vad_pool]
            if models_close is None:
                closes.append(close_v3_model_pools)
            if qwen_close is None:
                closes.append(close_qwen_client)
            for close in closes:
                try:
                    await close()
                except Exception as exc:
                    failures.append((close.__name__, str(exc)))
            for name, task in (("close_qwen_client", qwen_close),
                               ("close_v3_model_pools", models_close)):
                if task is not None:
                    try:
                        await await_completion(task)
                    except Exception as exc:
                        failures.append((name, str(exc)))
        finally:
            try:
                await await_completion(sampler_close)
            except Exception as exc:
                failures.append(("resource_sampler", str(exc)))
        if failures:
            raise RuntimeError(f"服务收尾失败: {failures}")

    try:
        await start_vad_pool()
        # 放在同一个finally内且先于就绪：加载失败也关闭部分VAD/Qwen，不留半就绪服务。
        if getattr(settings, "qwen_preload_on_startup", False) and settings.qwen_asr_enabled:
            await preload_qwen_client()
            if settings.qwen_generation_mode == "continuous":
                get_slot_manager().bind_owner_pool(await get_qwen_client())
        await preload_v3_model_pools()
        yield
    finally:
        # 启动部分失败同样回收；调用方重复取消也不能遗弃正在关闭的资源。
        try:
            await await_completion(shutdown())
        except BaseException:
            # Uvicorn会记录lifespan失败，但TERM重放仍可能返回0；显式保留失败供CLI退出码判定。
            app.state.shutdown_failed = True
            raise


def create_app() -> FastAPI:
    app = FastAPI(title="SeaCraftASR", lifespan=lifespan)
    app.state.shutdown_failed = False
    app.state.resource_sampler = None

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 路由装配（路径迁移为 v3，业务契约保持不变）
    app.include_router(status_router)
    app.include_router(asr_v3_router)

    return app


app = create_app()

def run_server(argv=None):
    """统一启动入口；TERM重放必须走Python退出，保证NPU运行时atexit。"""
    import argparse
    import signal
    import uvicorn

    parser = argparse.ArgumentParser(description="SeaCraftASR Ascend local service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--workers", type=int, choices=(1,), default=1)
    parser.add_argument("--timeout-graceful-shutdown", type=int, default=30)
    args = parser.parse_args(argv)

    def exit_after_shutdown(_signum, _frame):
        # Uvicorn先执行lifespan，再重放原信号；默认TERM会绕过Python atexit。
        # NPU运行时还可能接管INT，因此正式关闭使用TERM + Python SystemExit。
        raise SystemExit(0)

    original_term = signal.signal(signal.SIGTERM, exit_after_shutdown)
    _set_process_name()
    try:
        # 单worker不需要字符串再导入，保证CLI读取的就是实际lifespan所在app的失败状态。
        uvicorn.run(app, host=args.host, port=args.port, workers=1, reload=False,
                    timeout_graceful_shutdown=args.timeout_graceful_shutdown)
    finally:
        signal.signal(signal.SIGTERM, original_term)
        if getattr(app.state, "shutdown_failed", False):
            # 不允许TERM的SystemExit(0)洗白lifespan收尾失败；正常TERM仍保持0。
            raise SystemExit(1)


if __name__ == "__main__":
    run_server()
