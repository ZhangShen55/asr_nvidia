"""真实批边界、跨请求公平与取消所有权；全为CPU桩，不加载NPU。"""
import asyncio
import io
import wave
import unittest
from core.qwen_batch_scheduler import BatchScheduler


def wav(seconds=.3):
    output=io.BytesIO()
    with wave.open(output,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000)
        f.writeframes(b'\0\0'*round(seconds*16000))
    return output.getvalue()


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    def scheduler(self, execute, **kwargs):
        options=dict(workers=1,batch_size=128,wait_ms=10,audio_frame_budget=102400,
                     token_budget=78336,max_new_tokens=256,capacity=128)
        options.update(kwargs)
        return BatchScheduler(execute,**options)

    async def test_actual128_has_8_request_owners_and_results(self):
        batches=[]
        async def execute(members,batch_id):
            batches.append([(m.request_id,m.segment_index) for m in members])
            return [f'{m.request_id}:{m.segment_index}' for m in members]
        scheduler=self.scheduler(execute)
        outputs=await asyncio.gather(*(scheduler.submit(wav(),None,request_id=str(r),segment_index=i)
                                      for r in range(8) for i in range(16)))
        self.assertEqual(len(batches),1); self.assertEqual(len(batches[0]),128)
        self.assertEqual([x[0] for x in batches[0][:8]],[str(i) for i in range(8)])
        self.assertEqual(outputs,[f'{r}:{i}' for r in range(8) for i in range(16)])
        self.assertEqual(scheduler.snapshot()['batch_sizes'],{128:1})
        await scheduler.close()

    async def test_only32_ready_do_not_wait_for128(self):
        async def execute(members,_): return [m.segment_index for m in members]
        scheduler=self.scheduler(execute)
        out=await asyncio.wait_for(asyncio.gather(*(scheduler.submit(wav(),None,request_id=str(i%2),segment_index=i)
                                                  for i in range(32))),.3)
        self.assertEqual(len(out),32)
        self.assertEqual(scheduler.snapshot()['batch_sizes'],{32:1})
        await scheduler.close()

    async def test_frame_budget_splits_without_modifying_members(self):
        seen=[]
        async def execute(members,_):
            seen.extend(m.segment_index for m in members)
            return [m.audio for m in members]
        scheduler=self.scheduler(execute,audio_frame_budget=200)
        source=wav()
        out=await asyncio.gather(*(scheduler.submit(source,None,request_id='a',segment_index=i) for i in range(5)))
        self.assertEqual(out,[source]*5); self.assertEqual(seen,list(range(5)))
        self.assertEqual(scheduler.snapshot()['batch_sizes'],{2:2,1:1})
        await scheduler.close()

    async def test_queued_cancel_is_not_executed(self):
        entered=asyncio.Event(); release=asyncio.Event(); seen=[]
        async def execute(members,_):
            seen.extend(m.segment_index for m in members)
            entered.set(); await release.wait()
            return [1]*len(members)
        scheduler=self.scheduler(execute,batch_size=1)
        first=asyncio.create_task(scheduler.submit(wav(),None,segment_index=1))
        await entered.wait()
        second=asyncio.create_task(scheduler.submit(wav(),None,segment_index=2))
        await asyncio.sleep(.01); second.cancel()
        with self.assertRaises(asyncio.CancelledError): await second
        release.set(); await first
        self.assertEqual(seen,[1]); self.assertEqual(scheduler.snapshot()['inflight'],0)
        await scheduler.close()

    async def test_member_repeated_cancel_waits_for_shared_batch(self):
        entered=asyncio.Event(); release=asyncio.Event()
        async def execute(members,_):
            entered.set(); await release.wait()
            return [m.segment_index for m in members]
        scheduler=self.scheduler(execute,batch_size=2)
        first=asyncio.create_task(scheduler.submit(wav(),None,segment_index=1))
        second=asyncio.create_task(scheduler.submit(wav(),None,segment_index=2))
        await entered.wait(); first.cancel(); await asyncio.sleep(.01); first.cancel()
        await asyncio.sleep(.01); self.assertFalse(first.done())
        release.set()
        with self.assertRaises(asyncio.CancelledError): await first
        self.assertEqual(await second,2)
        self.assertEqual(scheduler.snapshot()['inflight'],0)
        await scheduler.close()

    async def test_two_workers_take_shared_work(self):
        busy=0; peak=0
        async def execute(members,_):
            nonlocal busy,peak
            busy+=1; peak=max(peak,busy)
            await asyncio.sleep(.01)
            busy-=1
            return [m.segment_index for m in members]
        scheduler=self.scheduler(execute,workers=2,batch_size=2)
        out=await asyncio.gather(*(scheduler.submit(wav(),None,request_id='a',segment_index=i) for i in range(8)))
        self.assertEqual(out,list(range(8))); self.assertEqual(peak,2)
        await scheduler.close()

    async def test_single_oversize_rejected_without_leaking(self):
        async def execute(*_): raise AssertionError('not called')
        scheduler=self.scheduler(execute,audio_frame_budget=1)
        with self.assertRaises(ValueError): await scheduler.submit(wav(),None)
        self.assertEqual(scheduler.snapshot()['inflight'],0)
        await scheduler.close()


    async def test_compatible_tails_are_not_blocked_by_alternating_heads(self):
        batches=[]
        async def execute(members,_):
            batches.append([m.frames for m in members])
            return [m.segment_index for m in members]
        scheduler=self.scheduler(execute,duration_ratio=4)
        tasks=[scheduler.submit(wav(1 if i%2==0 else 8),None,request_id=str(r),segment_index=i)
               for r in range(8) for i in range(16)]
        result=await asyncio.gather(*tasks)
        self.assertEqual(result,list(range(16))*8)
        self.assertEqual([len(x) for x in batches],[64,64])
        self.assertTrue(all(max(x)/min(x)<=4 for x in batches))
        await scheduler.close()

    async def test_wider_duration_ratio_allows_actual128_mixed_batch(self):
        batches=[]
        async def execute(members,_):
            batches.append(len(members)); return [1]*len(members)
        scheduler=self.scheduler(execute,duration_ratio=32)
        await asyncio.gather(*(scheduler.submit(wav(.3 if i%2==0 else 8),None,request_id=str(r))
                              for r in range(8) for i in range(16)))
        self.assertEqual(batches,[128])
        self.assertEqual(scheduler.snapshot()['recent_batches'][0]['estimated_padded_audio_frames'],102400)
        await scheduler.close()

    async def test_padding_budget_uses_full_single_feature_axis(self):
        batches=[]
        async def execute(members,_):
            batches.append(len(members));return [1]*len(members)
        scheduler=self.scheduler(execute,audio_frame_budget=600)
        await asyncio.gather(*(scheduler.submit(wav(3.01),None,request_id='a') for _ in range(3)))
        self.assertEqual(batches,[1,1,1]) # 301有效帧，完整single400帧；不能按301*2凑批。
        self.assertEqual(scheduler.snapshot()['recent_batches'][0]['estimated_padded_audio_frames'],400)
        await scheduler.close()

    async def test_incompatible_old_head_runs_before_replenished_tail(self):
        seen=[];tasks=[]
        scheduler=None
        async def execute(members,_):
            seen.append([(m.request_id,m.segment_index) for m in members])
            if len(seen)==1:
                tasks.extend(asyncio.create_task(scheduler.submit(wav(1),None,request_id='a',segment_index=i))
                             for i in range(20,28))
                await asyncio.sleep(0)
            return [1]*len(members)
        scheduler=self.scheduler(execute,batch_size=4)
        initial=[asyncio.create_task(scheduler.submit(wav(8 if i==1 else 1),None,request_id='a',segment_index=i))
                 for i in range(8)]
        await asyncio.gather(*initial)
        await asyncio.gather(*tasks)
        self.assertIn(('a',1),seen[1])
        await scheduler.close()

    def test_duration_ratio_rejects_nonfinite_bool_and_out_of_range(self):
        for value in [True,0,65,float('inf'),float('nan'),'4']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                self.scheduler(None,duration_ratio=value)


    async def test_other_request_old_head_anchors_despite_short_refill(self):
        seen=[]; extra=[]; scheduler=None
        async def execute(members,_):
            seen.append([(m.request_id,m.segment_index) for m in members])
            if len(seen)<=3:
                extra.extend(asyncio.create_task(scheduler.submit(wav(1),None,request_id='a',segment_index=100+len(extra)+i))
                             for i in range(8))
                await asyncio.sleep(0)
            return [1]*len(members)
        scheduler=self.scheduler(execute,batch_size=8)
        initial=[asyncio.create_task(scheduler.submit(wav(1),None,request_id='a',segment_index=i)) for i in range(8)]
        initial += [asyncio.create_task(scheduler.submit(wav(8 if i==0 else 1),None,request_id='b',segment_index=i)) for i in range(8)]
        await asyncio.gather(*initial)
        await asyncio.gather(*extra)
        owner_batch=next(i for i,b in enumerate(seen) if ('b',0) in b)
        self.assertLessEqual(owner_batch,2)
        await scheduler.close()

    async def test_dynamic_window16_refills512_segments_without_alternating_11_and_5(self):
        # 测生产event loop的10ms调度预算；IsolatedAsyncioTestCase默认debug记录
        # 每个Task栈会人为耗尽该预算，实际服务默认debug=False。其它测试仍开debug。
        loop=asyncio.get_running_loop();previous_debug=loop.get_debug();loop.set_debug(False)
        self.addCleanup(loop.set_debug,previous_debug)
        first_started=asyncio.Event();batch_sizes=[];active=0;peak=0
        async def execute(members,_):
            batch_sizes.append(len(members));first_started.set()
            await asyncio.sleep(.02)
            return [m.segment_index for m in members]
        scheduler=self.scheduler(execute)
        audio=wav(1)
        async def producer(slot):
            nonlocal active,peak
            if slot>=11:await first_started.wait()
            values=[]
            for index in range(slot,512,16):
                # 有限16生产者；已有第一批11段，余5段及后续段模拟1~3ms切片。
                if index>=16 or slot>=11:await asyncio.sleep((index%3+1)/1000)
                active+=1;peak=max(peak,active)
                try:values.append(await scheduler.submit(audio,None,request_id='audio',segment_index=index))
                finally:active-=1
            return values
        try:
            results=await asyncio.wait_for(asyncio.gather(*(producer(i) for i in range(16))),5)
            self.assertEqual(sorted(value for group in results for value in group),list(range(512)))
            self.assertLessEqual(peak,16)
            steady=batch_sizes[1:-1] # 初始11和末尾不足16段不是稳定补位窗口。
            self.assertTrue(steady)
            self.assertGreaterEqual(sum(size==16 for size in steady)/len(steady),.9,batch_sizes)
            self.assertEqual(scheduler.snapshot()['inflight'],0)
        finally:await scheduler.close()

    async def test_new_arrivals_do_not_extend_fixed_refill_deadline(self):
        entered=asyncio.Event();release=asyncio.Event();completed=asyncio.Event();filler_done=asyncio.Event()
        dispatched=[];extra=[]
        async def execute(members,_):
            dispatched.append(len(members))
            if len(dispatched)==1:
                entered.set();await release.wait();completed.set()
            elif len(dispatched)==2:
                self.assertFalse(filler_done.is_set(),'arrival通知不能重置补位deadline并等到生产结束')
            return [1]*len(members)
        scheduler=self.scheduler(execute)
        first=asyncio.create_task(scheduler.submit(wav(1),None,request_id='a'))
        await entered.wait()
        tail=asyncio.create_task(scheduler.submit(wav(1),None,request_id='b'))
        await asyncio.sleep(.02) # 旧tail的原始入队10ms窗口已经过期。
        async def filler():
            await completed.wait()
            for _ in range(20):
                await asyncio.sleep(.003)
                extra.append(asyncio.create_task(scheduler.submit(wav(1),None,request_id='c')))
            filler_done.set()
        feeding=asyncio.create_task(filler());release.set()
        try:
            await asyncio.wait_for(asyncio.gather(first,tail,feeding),1)
            await asyncio.gather(*extra)
            self.assertGreaterEqual(len(dispatched),2)
        finally:await scheduler.close()

    async def test_close_wakes_worker_during_post_completion_refill_wait(self):
        entered=asyncio.Event();release=asyncio.Event()
        async def execute(members,_):
            entered.set();await release.wait();return [1]*len(members)
        scheduler=self.scheduler(execute,wait_ms=100)
        first=asyncio.create_task(scheduler.submit(wav(1),None))
        await entered.wait()
        tail=asyncio.create_task(scheduler.submit(wav(1),None))
        await asyncio.sleep(.01);release.set();await first
        await asyncio.wait_for(scheduler.close(),.05)
        with self.assertRaises(RuntimeError):await tail
