from unittest.mock import MagicMock, patch
from ttk.remote.server import xpu_server
from ttk.remote.server.xpu_server import XpuRequestHandler
import threading


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


def test_assign_single_gpu():
    """单卡：try-lock 直接成功。"""
    h = _make_handler([0], {0: threading.Lock()})
    dev = h._assign_device()
    assert dev == 0
    xpu_server._device_locks[dev].release()


def test_assign_rr_distribution():
    """2 卡：连续 4 次分配交替到 0,1,0,1。"""
    locks = {0: threading.Lock(), 1: threading.Lock()}
    h = _make_handler([0, 1], locks)
    devs = []
    for _ in range(4):
        d = h._assign_device()
        devs.append(d)
        locks[d].release()
    assert devs == [0, 1, 0, 1]


def test_assign_try_lock_skips_busy():
    """device 0 被占 → 跳到 device 1。"""
    locks = {0: threading.Lock(), 1: threading.Lock()}
    locks[0].acquire()  # 占住 device 0
    h = _make_handler([0, 1], locks)
    dev = h._assign_device()
    assert dev == 1  # 跳到空闲的 device 1
    locks[1].release()
    locks[0].release()


def test_assign_non_contiguous_ids():
    """device_ids=[2,5]（非连续）：返回实际 id，不是索引。"""
    locks = {2: threading.Lock(), 5: threading.Lock()}
    h = _make_handler([2, 5], locks)
    dev = h._assign_device()
    assert dev in (2, 5)
    locks[dev].release()


def test_assign_all_busy_blocks_then_succeeds():
    """全占 → 阻塞 → 另一线程释放后获得。"""
    locks = {0: threading.Lock()}
    locks[0].acquire()
    h = _make_handler([0], locks)

    result = []
    def t():
        result.append(h._assign_device())
    th = threading.Thread(target=t)
    th.start()
    # 等一小会让阻塞线程就位
    import time; time.sleep(0.1)
    locks[0].release()  # 释放 → 阻塞线程获得
    th.join(timeout=2)
    assert result == [0]
    # 锁已被阻塞线程 acquire，不再 release（它会在 finally release）


def test_assign_cpu_mode():
    """CPU 模式 device_ids=["cpu"] → 返回 "cpu"，不锁。"""
    h = _make_handler(["cpu"], {})
    dev = h._assign_device()
    assert dev == "cpu"


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
