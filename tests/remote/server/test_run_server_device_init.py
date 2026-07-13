from ttk.remote.server.xpu_server import _init_device_locks
import threading


def test_init_device_locks_for_gpu():
    """_init_device_locks([0,1]) → _device_locks 有 2 个 Lock（调真函数）。"""
    _init_device_locks([0, 1])
    from ttk.remote.server import xpu_server
    assert set(xpu_server._device_locks.keys()) == {0, 1}
    assert all(isinstance(v, type(threading.Lock())) for v in xpu_server._device_locks.values())


def test_init_device_locks_empty_for_cpu():
    """_init_device_locks(["cpu"]) → _device_locks 为空。"""
    _init_device_locks(["cpu"])
    from ttk.remote.server import xpu_server
    assert xpu_server._device_locks == {}


def test_init_device_locks_non_contiguous():
    """_init_device_locks([2,5]) → keys={2,5}。"""
    _init_device_locks([2, 5])
    from ttk.remote.server import xpu_server
    assert set(xpu_server._device_locks.keys()) == {2, 5}


def test_init_device_locks_clears_stale():
    """_init_device_locks 先 clear 再填——不残留旧 device 的锁。"""
    from ttk.remote.server import xpu_server
    xpu_server._device_locks = {99: threading.Lock()}  # stale
    _init_device_locks([0, 1])
    assert 99 not in xpu_server._device_locks
    assert set(xpu_server._device_locks.keys()) == {0, 1}
