# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_assign_device 单元测试：覆盖单卡/多卡 RR 分配、busy 跳过、全占阻塞、CPU 模式及并发场景。"""
import threading

from ttk.remote.server import xpu_server
from ttk.remote.server.xpu_server import XpuRequestHandler


def _make_handler(device_ids, gpu_locks):
    """Construct a handler with given device_ids + pre-built gpu_locks.

    不设 _device_rr_counter/_device_rr_lock 实例属性——让它们走类属性
    （真实 server 每个请求一个 handler 实例，counter 必须跨实例共享）。
    """
    h = XpuRequestHandler.__new__(XpuRequestHandler)
    h.device_ids = device_ids
    xpu_server._device_locks = gpu_locks
    # 重置类属性（测试隔离）
    XpuRequestHandler._device_rr_counter = 0
    XpuRequestHandler._device_rr_lock = threading.Lock()
    return h


def test_concurrent_two_requests_different_devices():
    """2 个线程同时 _assign_device → 分到不同 device（spec §6 多卡并发）。"""
    locks = {0: threading.Lock(), 1: threading.Lock()}
    h = _make_handler([0, 1], locks)
    results = []
    barrier = threading.Barrier(2)

    def t():
        barrier.wait()   # 确保同时出发
        results.append(h._assign_device())
    ths = [threading.Thread(target=t) for _ in range(2)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(timeout=2)
    assert sorted(results) == [0, 1]   # 分到不同 device
    for d in results:
        locks[d].release()


def test_data_does_not_concurrent_with_perf_same_device():
    """PERF 占 device 0 → DATA try-lock 0 失败 → 跳到 device 1（spec §6 DATA+PERF 不并发）。"""
    locks = {0: threading.Lock(), 1: threading.Lock()}
    locks[0].acquire()   # 模拟 PERF 占 device 0
    h = _make_handler([0, 1], locks)
    h._device_rr_counter = 0   # 强制 RR 起点为 0
    dev = h._assign_device()   # DATA 请求
    assert dev == 1   # PERF 在 0，DATA 跳到 1
    locks[1].release()
    locks[0].release()
