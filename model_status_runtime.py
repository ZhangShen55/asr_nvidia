"""运行时状态字典的 Python helper。

VAD reader 线程会从 multiprocessing 回调进入状态登记。把这段纯字典操作
放在未编译的小 helper 中，避免 Cython 扩展在解释器退出时处理临时引用。
"""


def update_vad_runtime(
    entry,
    device,
    instances,
    loaded,
    allocated_mb,
    configured_instances=None,
    successful_instances=None,
    failed_instances=0,
    devices=None,
    busy=0,
    waiting=0,
    failures=None,
):
    entry["loaded"] = bool(loaded)
    entry["device"] = device[0] if isinstance(device, list) and device else device
    if devices is None:
        devices = device if isinstance(device, list) else [device]
    entry["devices"] = list(devices)
    entry["instances"] = max(0, int(instances))
    entry["allocated_mb"] = round(max(0.0, allocated_mb), 2)
    if configured_instances is None:
        configured_instances = instances or 0
    if successful_instances is None:
        successful_instances = instances
    entry["configured_instances"] = int(configured_instances)
    entry["available_instances"] = max(0, int(successful_instances) - int(busy))
    entry["failed_instances"] = int(failed_instances)
    entry["busy"] = int(busy)
    entry["waiting"] = int(waiting)
    entry["failure"] = (failures or [None])[-1]
