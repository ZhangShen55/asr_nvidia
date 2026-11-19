"""新process模式懒状态与lifespan关闭顺序；全部替身，不使用NPU。"""
import asyncio
from contextlib import ExitStack
from unittest.mock import AsyncMock,PropertyMock,patch

import pytest


def test_process_default_factory_constructor_and_global_status_stay_lazy(tmp_path):
    from core.config import Settings
    from core import qwen_asr_client as module
    from core.qwen_asr_client import QwenAsrClient,get_qwen_status
    from core.qwen_asr_process import LocalQwenProcessRuntime
    async def scenario():
        with patch('core.qwen_asr_process.multiprocessing.get_context',side_effect=AssertionError('must not spawn')):
            client=QwenAsrClient(local_model_dir=str(tmp_path),local_device='npu:0',execution_mode='process')
            assert isinstance(client._runtimes[0],LocalQwenProcessRuntime)
            state=client.snapshot()
            assert state['execution_mode']=='process' and state['memory_process_scope']=='model_worker'
            assert state['configured_model_workers']==1 and state['model_workers'][0]['worker_pid'] is None
            await client.aclose()
        with patch.object(module,'_client_instance',None), \
             patch.object(Settings,'qwen_execution_mode',new_callable=PropertyMock,return_value='process'), \
             patch('core.qwen_asr_process.LocalQwenProcessRuntime',side_effect=AssertionError('global status must not construct')):
            status=get_qwen_status()
            assert status['execution_mode']=='process' and status['model_workers']==[]
    asyncio.run(scenario())


@pytest.mark.parametrize('mode',['thread','process'])
@pytest.mark.parametrize('fail_nn',[False,True])
def test_lifespan_process_starts_qwen_closing_before_nn_and_aggregates_failure(mode,fail_nn):
    import importlib.util
    import sys
    from pathlib import Path
    # Other repository tests may prepend the authoritative root; load this draft/stage main exactly.
    spec=importlib.util.spec_from_file_location('process_shutdown_main',Path(__file__).resolve().parents[1]/'main.py')
    main=importlib.util.module_from_spec(spec);sys.modules[spec.name]=main;spec.loader.exec_module(main)
    from core.config import Settings
    events=[]
    async def scenario():
        began=asyncio.Event()
        async def stop_vad():events.append('VAD');await asyncio.sleep(0)
        async def close_qwen():events.append('Qwen');began.set();await asyncio.sleep(0)
        async def close_nn():
            events.append('NN')
            if mode=='process':await asyncio.wait_for(began.wait(),1)
            else:assert not began.is_set()
            if fail_nn:raise RuntimeError('controlled NN close failure')
        with ExitStack() as stack:
            stack.enter_context(patch.object(Settings,'qwen_execution_mode',new_callable=PropertyMock,return_value=mode))
            for name in ('_set_process_name','setup_logging','cleanup_stale_service_cache','reset_stats'):
                stack.enter_context(patch.object(main,name))
            stack.enter_context(patch.object(main.settings,'validate_model_backends'))
            stack.enter_context(patch.object(main,'validate_npu_runtime',return_value={'cpu_test':True}))
            stack.enter_context(patch.object(main,'start_vad_pool',AsyncMock()))
            stack.enter_context(patch.object(main,'stop_vad_pool',stop_vad))
            stack.enter_context(patch.object(main,'close_v3_model_pools',close_nn))
            stack.enter_context(patch.object(main,'close_qwen_client',close_qwen))
            app=main.create_app()
            async def context():
                async with main.lifespan(app):pass
            if fail_nn:
                with pytest.raises(RuntimeError,match='服务收尾失败'):await context()
                assert app.state.shutdown_failed
            else:
                await context();assert not app.state.shutdown_failed
        assert set(events)=={'VAD','Qwen','NN'}
        assert (events.index('Qwen')<events.index('NN')) is (mode=='process')
    asyncio.run(scenario())
