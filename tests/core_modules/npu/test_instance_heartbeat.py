# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for NpuInstance heartbeat lifecycle."""
from unittest.mock import MagicMock


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

    def test_close_subprocesses_stops_heartbeat(self):
        from ttk.core_modules.npu.instance_refactor import NpuInstance

        ins = NpuInstance()
        mock_mgr = MagicMock()
        ins.heartbeat_manager = mock_mgr
        ins.process_groups = {}

        ins._stop_heartbeat_process()

        mock_mgr.stop.assert_called_once()
