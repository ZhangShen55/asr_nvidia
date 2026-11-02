"""等待本请求已启动的工作收尾，避免取消后提前归还资源。"""

import asyncio


async def await_completion(awaitable):
    """保留工作任务，延迟传播调用方取消直到该任务真正结束。

    适用于不可中断的线程工作和必须完成的资源回收；底层任务自身取消仍正常传播。
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancelled = exc
        except BaseException:
            if cancelled is not None:
                raise cancelled from None
            raise
    if cancelled is not None:
        raise cancelled
    return result


async def run_in_thread(function, /, *args, **kwargs):
    """线程无法强行停止，调用方取消时仍等它完成再回收关联资源。"""
    return await await_completion(asyncio.to_thread(function, *args, **kwargs))
