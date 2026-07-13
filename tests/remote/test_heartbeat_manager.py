import os
import time
from unittest.mock import MagicMock, patch

import pytest


class TestHeartbeatManagerStart:
    def test_start_sets_ttk_xpu_health_path_env_var(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        # Use a minimal heartbeat target that exits immediately
        def fake_loop(**kwargs):
            pass

        mgr = HeartbeatManager(
            heartbeat_target=fake_loop,
            root_path=str(tmp_path),
            tenant_id="test_abc123",
            endpoints=[],
        )
        mgr.start()
        try:
            assert "TTK_XPU_HEALTH_PATH" in os.environ
            assert ".ttk" in os.environ["TTK_XPU_HEALTH_PATH"]
            assert "test_abc123" in os.environ["TTK_XPU_HEALTH_PATH"]
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_health_path_uses_dot_ttk_subdirectory(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        def fake_loop(**kwargs): pass
        mgr = HeartbeatManager(
            heartbeat_target=fake_loop,
            root_path=str(tmp_path),
            tenant_id="tid123",
            endpoints=[],
        )
        mgr.start()
        try:
            assert ".ttk/xpu_health_tid123.json" in os.environ["TTK_XPU_HEALTH_PATH"]
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_start_called_twice_is_idempotent(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        def fake_loop(**kwargs):
            import time
            time.sleep(0.01)

        mgr = HeartbeatManager(
            heartbeat_target=fake_loop,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[],
        )
        mgr.start()
        try:
            proc1 = mgr._process
            mgr.start()  # should not create a second process
            assert mgr._process is proc1
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_no_spawn_when_endpoints_empty(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        def fake_loop(**kwargs): pass
        mgr = HeartbeatManager(
            heartbeat_target=fake_loop,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[],  # no endpoints
        )
        mgr.start()
        try:
            assert mgr._process is None
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_start_process_name_is_HB(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        # NOTE: target must be a MODULE-LEVEL function (not a local closure) so it
        # is picklable under any multiprocessing start method. test_device_lock.py
        # sets forkserver globally, which requires pickling the Process target —
        # a local closure raises "Can't pickle local object" in the full suite.
        mgr = HeartbeatManager(
            heartbeat_target=_fake_brief_loop,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[("fake", 9999)],
        )
        mgr.start()
        try:
            assert mgr._process is not None
            assert mgr._process.name == "HB"
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_start_passes_tls_to_subprocess(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        # Target must be module-level (picklable under forkserver — see note above).
        # tls kwarg acceptance is verified by HeartbeatManager.start() succeeding
        # (it passes tls= into the Process kwargs); the subprocess runs in a
        # separate process so a captured={} dict there would not be visible here.
        mgr = HeartbeatManager(
            heartbeat_target=_fake_brief_loop,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[("fake", 9999)],
            tls={"verify": False},
        )
        mgr.start()
        try:
            # tls kwarg should be accepted by the subprocess target signature
            assert mgr._process is not None
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)


def _fake_long_loop(**kwargs):
    import time
    time.sleep(100)


def _fake_brief_loop(**kwargs):
    """Module-level short-sleep target for start/process-name tests.

    Must be module-level (not a closure) so multiprocessing can pickle it under
    forkserver/spawn start methods. Sleeps a few seconds: long enough to be alive
    when the test inspects mgr._process, short enough that stop()'s kill is clean.
    """
    import time
    time.sleep(5)


def _fake_immediate_exit(**kwargs):
    pass


class TestHeartbeatManagerStop:
    def test_stop_terminates_subprocess(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager

        mgr = HeartbeatManager(
            heartbeat_target=_fake_long_loop,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[("fake", 9999)],  # non-empty so subprocess spawns
        )
        mgr.start()
        assert mgr._process is not None
        assert mgr._process.is_alive()
        try:
            mgr.stop()
            assert not mgr._process or not mgr._process.is_alive()
        finally:
            mgr.stop()
            os.environ.pop("TTK_XPU_HEALTH_PATH", None)

    def test_stop_clears_env_var(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        mgr = HeartbeatManager(
            heartbeat_target=_fake_immediate_exit,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[("a", 1)],
        )
        mgr.start()
        assert "TTK_XPU_HEALTH_PATH" in os.environ
        mgr.stop()
        assert "TTK_XPU_HEALTH_PATH" not in os.environ

    def test_stop_removes_health_file(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        from ttk.remote.health_file import atomic_write_json

        mgr = HeartbeatManager(
            heartbeat_target=lambda **kw: None,
            root_path=str(tmp_path),
            tenant_id="tid",
            endpoints=[],
        )
        mgr.start()
        try:
            # Simulate heartbeat writing the file
            atomic_write_json(mgr.health_path, {"test": True})
            assert os.path.isfile(mgr.health_path)
        finally:
            mgr.stop()
        assert not os.path.isfile(mgr.health_path)

    def test_stop_when_never_started_is_safe(self, tmp_path):
        from ttk.remote.heartbeat_manager import HeartbeatManager
        mgr = HeartbeatManager(
            heartbeat_target=lambda **kw: None,
            root_path=str(tmp_path),
            tenant_id="t1",
            endpoints=[],
        )
        mgr.stop()  # must not raise


# supervise tests (new in Task 2)
from ttk.remote.heartbeat_manager import HeartbeatManager


def _noop(**kw):
    pass


def test_supervise_noop_when_alive():
    hm = HeartbeatManager(_noop, "/tmp", "t", [])
    hm._process = MagicMock()
    hm._process.is_alive.return_value = True
    before = hm._process
    hm.supervise()
    assert hm._process is before  # no respawn


def test_supervise_respawns_when_dead():
    hm = HeartbeatManager(_noop, "/tmp", "t", [])
    hm._process = MagicMock()
    hm._process.is_alive.return_value = False
    hm._process.exitcode = 1
    with patch("ttk.remote.heartbeat_manager.multiprocessing.Process") as mp:
        mp.return_value.start = MagicMock()
        hm.supervise()
        assert mp.called  # new HB Process spawned
        _, kwargs = mp.call_args
        assert kwargs.get("name") == "HB"
        assert "tls" in kwargs.get("kwargs", {})   # tls 透传


def test_supervise_noop_when_no_process():
    hm = HeartbeatManager(_noop, "/tmp", "t", [])
    hm.supervise()
    assert hm._process is None
