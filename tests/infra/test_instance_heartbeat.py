import time
from unittest.mock import MagicMock, patch, call

import pytest


class TestInstanceBaseHeartbeat:
    def test_profile_starts_heartbeat_before_setup_profile_object(self):
        """profile() must call _start_heartbeat_process BEFORE setup_profile_object,
        and AFTER _parse_testcases (and after validate_only early-return)."""
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.switches.device_count = 1
        ins.switches.process_per_device = 1
        ins.switches.test_mode = "op"
        ins.switches.validate_only = False

        manager = MagicMock()
        # Wire profile()'s collaborators to the same mock so we can assert call order.
        # Each step is a distinct child-mock; manager.mock_calls preserves order.
        ins.env_prepare = manager.env_prepare
        ins.get_device_platform = manager.get_device_platform
        ins.get_device_count = manager.get_device_count
        ins._parse_testcases = manager._parse_testcases
        ins._start_heartbeat_process = manager._start_heartbeat_process
        ins.setup_profile_object = manager.setup_profile_object
        ins.profile_object = manager.profile_object
        ins._open_result_file = manager._open_result_file
        ins.prepare_subprocesses = manager.prepare_subprocesses
        ins._prepare_tasks = manager._prepare_tasks
        ins._update_processes = manager._update_processes
        ins._handle_completed_process = manager._handle_completed_process
        ins._push_task_to_process = manager._push_task_to_process
        ins._close_idle_processes = manager._close_idle_processes
        ins._summary_print = manager._summary_print
        ins.close_subprocesses = manager.close_subprocesses
        ins._pre_exit = manager._pre_exit
        # make the main loop exit immediately
        ins.total_case_count = 1
        ins.completed_case_count = 1

        ins.profile()

        names = [c[0] for c in manager.mock_calls]
        # _start_heartbeat_process must come before setup_profile_object
        assert "_start_heartbeat_process" in names
        assert "setup_profile_object" in names
        assert names.index("_start_heartbeat_process") < names.index("setup_profile_object")
        # and it must come after _parse_testcases
        assert names.index("_parse_testcases") < names.index("_start_heartbeat_process")

    def test_start_heartbeat_constructs_manager_with_tls_when_remote_configured(self):
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.switches.device_count = 1

        mock_ep = MagicMock()
        mock_ep.host = "10.0.0.1"
        mock_ep.port = 9090

        mock_config = MagicMock()
        mock_config.endpoints = [mock_ep]
        mock_config.tls_ca = ""
        mock_config.tls_cert = ""
        mock_config.tls_key = ""

        # Patch at the source module since method uses lazy import
        with patch("ttk.remote.is_remote_configured", return_value=True), \
             patch("ttk.remote.config.get_remote_config", return_value=mock_config), \
             patch("ttk.remote.tls.tls_from_config", return_value={"ca_cert": ""}) as tls_mock, \
             patch("ttk.remote.heartbeat_manager.HeartbeatManager") as HBM:
            mock_mgr = MagicMock()
            HBM.return_value = mock_mgr

            ins._start_heartbeat_process()

            HBM.assert_called_once()
            _, kwargs = HBM.call_args
            assert "tls" in kwargs
            tls_mock.assert_called_once_with(mock_config)
            mock_mgr.start.assert_called_once()

    def test_prepare_subprocesses_no_longer_starts_heartbeat(self):
        """prepare_subprocesses must NOT call _start_heartbeat_process anymore
        (moved to profile())."""
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.switches.device_count = 1
        ins.switches.process_per_device = 1
        ins.used_device = [0]
        ins._prepare_device_locks = MagicMock()
        ins.process_groups = {}
        ins._start_heartbeat_process = MagicMock()

        # Stub the readiness loop so no real ProcessGroup poll happens
        with patch.object(ins.__class__, "prepare_subprocesses",
                          wraps=ins.prepare_subprocesses):
            # prepare_subprocesses accesses process_groups values; with empty dict the
            # readiness loop is skipped.
            ins.prepare_subprocesses()

        ins._start_heartbeat_process.assert_not_called()

    def test_start_heartbeat_skips_when_no_remote(self):
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.switches.device_count = 1

        with patch("ttk.remote.is_remote_configured", return_value=False):
            ins._start_heartbeat_process()
            assert ins.heartbeat_manager is None

    def test_start_heartbeat_raises_on_cert_without_key(self):
        """tls_from_config raises RuntimeError on cert-without-key; that raise must
        propagate out of _start_heartbeat_process (loud startup fail), NOT be
        swallowed by the ImportError try/except."""
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.switches.device_count = 1

        mock_config = MagicMock()
        mock_config.endpoints = [MagicMock(host="h", port=1)]
        mock_config.tls_ca = ""
        mock_config.tls_cert = "/etc/cert.pem"
        mock_config.tls_key = ""  # cert without key -> tls_from_config raises

        with patch("ttk.remote.is_remote_configured", return_value=True), \
             patch("ttk.remote.config.get_remote_config", return_value=mock_config):
            with pytest.raises(RuntimeError):
                ins._start_heartbeat_process()

    def test_close_subprocesses_stops_heartbeat(self):
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        mock_mgr = MagicMock()
        ins.heartbeat_manager = mock_mgr
        ins.process_groups = {}

        ins._stop_heartbeat_process()

        mock_mgr.stop.assert_called_once()


class TestSuperviseHeartbeat:
    def test_supervise_noop_when_no_heartbeat(self):
        """_supervise_heartbeat must be None-safe (heartbeat_manager is None
        when remote is off)."""
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        ins.heartbeat_manager = None
        # must not raise
        ins._supervise_heartbeat()

    def test_supervise_throttled(self):
        """_supervise_heartbeat must throttle ~1s: within the window it should
        NOT call supervise(), after the window it should."""
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        mock_mgr = MagicMock()
        ins.heartbeat_manager = mock_mgr

        # First call: no prior timestamp -> should call supervise immediately.
        ins._supervise_heartbeat()
        assert mock_mgr.supervise.call_count == 1

        # Second call immediately after: within 1s window -> throttled, not called.
        ins._supervise_heartbeat()
        assert mock_mgr.supervise.call_count == 1

        # Advance the last-supervise timestamp past the window -> should call again.
        ins._last_supervise_ts = time.time() - 1.1
        ins._supervise_heartbeat()
        assert mock_mgr.supervise.call_count == 2
