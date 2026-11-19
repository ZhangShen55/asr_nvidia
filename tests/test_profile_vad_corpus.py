import asyncio
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("profile_vad_corpus", Path(__file__).resolve().parents[1]/"scripts/profile_vad_corpus.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def make_samples(tmp_path, count=10):
    result = []
    for i in range(count):
        raw = os.fsencode(tmp_path)+b"/source-\xff-"+str(i).encode()
        data = bytes([i])
        with open(raw, "wb") as stream:
            stream.write(data)
        result.append({"path": os.fsdecode(raw), "path_bytes_b64": base64.b64encode(raw).decode(),
                       "sha256": hashlib.sha256(data).hexdigest(), "audio_seconds": 2,
                       "sample_id": str(i)})
    return result


class FakePool:
    def __init__(self, workers=2, stuck=False, start_fail=False):
        self.workers, self.stuck, self.start_fail = workers, stuck, start_fail
        self.active = self.peak = 0
        self.stopped = False
        self.events = []
        self._workers = {}
        self.paths = []
    async def start(self):
        if self.start_fail:
            raise RuntimeError("fake startup failure")
    async def stop(self, timeout_s):
        # 取消时worker仍在读取完整源，源必须保留至stop被调用。
        if self.stuck:
            assert self.paths and all(Path(path).is_file() for path in self.paths)
        self.stopped = True
    def stats(self):
        return {"loaded_instances":self.workers, "failed_instances":0,"busy":self.active}
    async def detect(self, path):
        assert path.isascii()
        self.paths.append(path)
        index = Path(path).read_bytes()[0]
        self.events.append((index,"start"))
        self.active += 1; self.peak=max(self.peak,self.active)
        try:
            await asyncio.sleep(10 if self.stuck else .04 if index == 0 else .001)
            self.events.append((index,"end"))
            return [{"value":[[0,299],[300,600],[700,2000]]}]
        finally:
            self.active -= 1


def options(**kwargs):
    values = dict(workers=2, wall_timeout_s=420, no_progress_s=120, monitor_interval_s=.002, stop_timeout_s=.1)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_inventory_deduplicates_sha_and_legacy_paths(tmp_path):
    samples = make_samples(tmp_path,2)
    rows = [dict(s,format="WAV",duration_s=2) for s in samples]
    rows.extend([rows[0], dict(rows[1],format="MP3")])
    inventory = tmp_path/"inventory.json"; tool.save(inventory,rows)
    selected, excluded = tool.inventory_samples(inventory)
    assert len(selected)==2 and len(excluded)==2
    assert os.fsencode(selected[0]["path"])==base64.b64decode(rows[0]["path_bytes_b64"])


def test_copy_uses_new_ascii_cache_and_preserves_source(tmp_path):
    sample = make_samples(tmp_path,1)[0]
    cache=tmp_path/"owned.wav"
    tool.prepare_copy(sample,cache)
    assert tool.digest(cache)==sample["sha256"]==tool.digest(sample["path"])
    with pytest.raises(FileExistsError):
        tool.prepare_copy(sample,cache)


def test_segment_300ms_boundary_and_unknown_language():
    result=tool.segment_stats([{"value":[[0,299],[300,600],[700,1001],[1100,2000]]}],2)
    assert result["segment_count"]==3 and result["short_dropped"]==1
    assert result["kept_intervals_ms"][0]==[300,600]
    assert result["effective_seconds"]==pytest.approx(1.501)
    assert result["acoustic_language"] is None


def test_window_refills_and_stops_pool_and_clears_only_owned_cache(tmp_path):
    samples=make_samples(tmp_path)
    output=tmp_path/"output"; output.mkdir()
    pool=FakePool()
    result=asyncio.run(tool.profile(pool,samples,output,options()))
    assert result["complete"] and pool.stopped and pool.peak==2
    assert pool.events.index((2,"start")) < pool.events.index((0,"end"))
    assert all(Path(sample["path"]).is_file() for sample in samples)
    assert not list((output/"owned-full-source-cache").iterdir())
    assert len(result["records"])==10


def test_watchdog_does_not_queue_entire_corpus_and_cache_lifetime(tmp_path):
    samples=make_samples(tmp_path)
    output=tmp_path/"output"; output.mkdir()
    pool=FakePool(stuck=True)
    result=asyncio.run(tool.profile(pool,samples,output,options(wall_timeout_s=.02)))
    assert result["status"]=="diagnostic_aborted" and pool.stopped
    assert len(pool.paths)==2
    assert sum(row["status"]=="not_submitted" for row in result["records"])==8
    assert not result["shutdown"]["cache_remaining"]


def test_failed_start_still_stops(tmp_path):
    output=tmp_path/"output"; output.mkdir()
    pool=FakePool(start_fail=True)
    with pytest.raises(RuntimeError):
        asyncio.run(tool.profile(pool,[],output,options()))
    assert pool.stopped
    assert json.loads((output/"summary.json").read_text())["status"]=="incomplete"
