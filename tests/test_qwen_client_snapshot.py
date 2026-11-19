"""状态观测只复制一次批历史，保留全部原响应字段和计数语义。"""
import asyncio
from unittest.mock import patch

from core.qwen_asr_client import QwenAsrClient


def test_client_snapshot_reads_batch_snapshot_once_and_preserves_response():
    class Runtime:
        def __init__(self, *args):
            self.failure = None
        def transcribe_batch(self, *args):
            raise AssertionError('snapshot must not execute inference')
        def snapshot(self):
            return {'loaded':True, 'loading':False, 'failure':None,
                    'memory_allocated_mb':2, 'memory_reserved_mb':3}
        def close(self):
            pass
    async def scenario():
        client = QwenAsrClient(execution_mode="thread", runtime_factory=Runtime, local_instance_count=2,
                               local_model_dir='/cpu-test-no-model', local_device='npu:7', local_dtype='bfloat16')
        batch_state = {'waiting':37, 'recent_batches':[
            {'batch_id':batch, 'members':[{'request_id':str(i//16), 'segment_index':i%16} for i in range(128)]}
            for batch in range(64)]}
        client._waiting = 3
        try:
            with patch.object(client._batch_scheduler, 'snapshot', return_value=batch_state) as batch_snapshot:
                actual = client.snapshot()
            batch_snapshot.assert_called_once_with()
            assert actual['batch_scheduler'] is batch_state
            assert actual == {
                'execution_mode':'thread','memory_process_scope':'service_process',
                'configured_model_workers':0,'model_workers':[],'shutdown':{'phase':'open','forced_cleanup':False},
                'backend':'transformers_torch_npu', 'model_dir':'/cpu-test-no-model',
                'device':'npu:7', 'dtype':'bfloat16', 'loaded':True, 'loading':False,
                'configured_instances':2, 'capacity':2, 'busy':0, 'waiting':40,
                'batch_scheduler':batch_state, 'failed_instances':0, 'execution_failures':0,
                'closing':False, 'memory_allocated_mb':4, 'memory_reserved_mb':6,
                'memory_source':'pytorch_load_delta_estimate',
                'instances':[dict(Runtime().snapshot(), index=i, busy=False) for i in range(2)],
            }
        finally:
            client._waiting = 0
            await client.aclose()
    asyncio.run(scenario())
