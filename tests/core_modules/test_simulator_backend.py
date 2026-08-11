#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
"""Unit tests for the NPUSim simulator backend (no device / simulation needed)."""
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ttk.cli.sim_args import apply_sim_args
from ttk.utilities.classes import MODE, SWITCHES


class _DummyRTSProfilingParam:
    """Minimal stand-in for RTSProfilingParam (module-level: picklable)."""
    switch = True
    compile_result = "SUCC"
    is_valid = True
    fail_reason = ""
    block_dim = 1

    def clear_atomic_output_workspace(self):
        pass


def _sim_args(**overrides):
    base = dict(
        backend="npusim",
        sim_soc_version="Ascend950",
        sim_output_dir=None,
        sim_report=False,
        sim_cores=None,
        sim_object_file=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestSimArgs:
    def test_default_backend_is_npu(self):
        sw = SWITCHES()
        assert sw.backend == "npu"

    def test_npusim_backend_normalizes_switches(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args())
        assert sw.backend == "npusim"
        assert sw.mode == MODE.ASCEND_CAMODEL
        # SoC name maps to the platform ini used by get_npu_hw_info.
        assert sw.dev_plat == "Ascend950PR_9589"
        assert sw.short_soc_version is None  # filled later by get_device_platform
        assert sw.warmup is False
        assert sw.TASK_PROFILING is False
        assert sw.deterministic_level == 0
        assert sw.process_per_device == 1
        assert sw.sim_output_dir == str(Path(sw.root_path) / "sim_output")

    def test_unknown_soc_passthrough(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args(sim_soc_version="Ascend999"))
        assert sw.dev_plat == "Ascend999"

    def test_npu_backend_leaves_mode_untouched(self):
        sw = SWITCHES()
        apply_sim_args(sw, _sim_args(backend="npu"))
        assert sw.mode == MODE.ASCEND_ONBOARD
        assert sw.dev_plat == "AUTO"

    def test_no_prof_rejected(self):
        sw = SWITCHES()
        with pytest.raises(ValueError, match="--backend npusim"):
            apply_sim_args(sw, _sim_args(no_prof=True))


class TestSwitchPickle:
    def test_pickle_roundtrip_with_sim_fields(self):
        sw = SWITCHES()
        sw.backend = "npusim"
        sw.sim_soc_version = "Ascend950"
        sw.sim_report = True
        restored = pickle.loads(pickle.dumps(sw))
        assert restored.backend == "npusim"
        assert restored.sim_soc_version == "Ascend950"
        assert restored.sim_report is True
        # Non-sim fields survive too.
        assert restored.deterministic_level == sw.deterministic_level


class TestCaseWriter:
    def test_case_dir_uses_safe_name(self):
        from ttk.core_modules.simulator import case_writer

        sw = SWITCHES()
        sw.sim_output_dir = "/tmp/ttk_sim_out"
        d = case_writer.case_dir(sw, "aclnnAdd/00 x")
        assert str(d) == "/tmp/ttk_sim_out/aclnnAdd_00_x" or "-" in str(d)

    def test_enabled_kernel_modes(self):
        from ttk.core_modules.simulator import case_writer

        sw = SWITCHES()
        # dyn enabled+prof on by default; cst/bin disabled.
        assert case_writer.enabled_kernel_modes(sw) == ("dyn",)
        sw.cst_switches.enabled = True
        assert case_writer.enabled_kernel_modes(sw) == ("dyn", "cst")


class TestWrapperGeneration:
    @pytest.fixture
    def sw(self, tmp_path):
        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        return sw

    def test_kernel_wrapper_generated(self, sw):
        from ttk.core_modules.simulator import case_writer
        from ttk.core_modules.simulator.wrapper import write_kernel_wrapper

        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_kernel_wrapper(sw, case_path)
        assert wrapper.is_file()
        assert wrapper.name == "caseA_kernel.py"
        text = wrapper.read_text()
        assert "OnlineRtsProfiling" in text
        assert "os._exit(0)" in text
        # camodel teardown (rtCtxDestroy/rtDeviceReset) busy-spins on some cases;
        # the wrapper skips it and lets os._exit reclaim the process.
        assert "skip_teardown=True" in text

    def test_aclnn_wrapper_generated(self, sw):
        from ttk.core_modules.simulator import case_writer
        from ttk.core_modules.simulator.wrapper import write_aclnn_wrapper

        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_aclnn_wrapper(sw, case_path)
        assert wrapper.is_file()
        assert wrapper.name == "caseA_aclnn.py"
        text = wrapper.read_text()
        assert "AclOpExecutor" in text
        assert "os._exit(0)" in text
        assert "skip_teardown=True" in text


class TestResultLoading:
    def test_kernel_result_ok(self, tmp_path):
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_mode_result

        mdir = tmp_path / "dyn"
        mdir.mkdir()
        (mdir / "result.json").write_text(
            json.dumps({"ok": True, "cycle": "UNKNOWN", "oob": "OK,OK"}))
        (mdir / "output_0.bin").write_bytes(b"\x00\x01")
        (mdir / "output_1.bin").write_bytes(b"\x02\x03")
        res = _load_mode_result(tmp_path, "dyn")
        assert isinstance(res, RTSProfilingResult)
        assert res.output_bytes == [b"\x00\x01", b"\x02\x03"]
        assert res.oob == "OK,OK"

    def test_kernel_result_failure(self, tmp_path):
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_mode_result

        mdir = tmp_path / "dyn"
        mdir.mkdir()
        (mdir / "result.json").write_text(
            json.dumps({"ok": False, "error": "SIM_EXECUTION_FAILED"}))
        res = _load_mode_result(tmp_path, "dyn")
        assert isinstance(res, RTSProfilingResult)
        assert res.cycle == "SIM_EXECUTION_FAILED"

    def test_aclnn_result_ok(self, tmp_path):
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_aclnn_result

        (tmp_path / "final_result.json").write_text(json.dumps(
            {"ok": True, "api_prof": "UNKNOWN", "op_prof": "TOTAL_CYCLE_TODO",
             "oob": "UNKNOWN", "deterministic_status": None}))
        (tmp_path / "output_0.bin").write_bytes(b"\x00\x01")
        with (tmp_path / "output_view_shapes.pkl").open("wb") as f:
            pickle.dump([("float32", [1, 2])], f)
        res = _load_aclnn_result(tmp_path)
        assert isinstance(res, ApiProfilingResult)
        assert res.success is True
        assert res.output_bytes == [b"\x00\x01"]
        assert res.output_view_shapes == (("float32", [1, 2]),)
        # json null round-trips to None (not the literal string "None").
        assert res.deterministic_status is None

    def test_aclnn_result_missing(self, tmp_path):
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_aclnn_result

        res = _load_aclnn_result(tmp_path)
        assert isinstance(res, ApiProfilingResult)
        assert res.success is False
        assert "SIM_RESULT_MISSING" in str(res.api_prof)

    def test_kernel_result_corrupt(self, tmp_path):
        """A truncated result.json surfaces as SIM_RESULT_CORRUPT, not a crash."""
        from ttk.core_modules.simulator.sim_profiling import _load_mode_result

        mdir = tmp_path / "dyn"
        mdir.mkdir()
        (mdir / "result.json").write_text('{"ok": true, "cyc')  # truncated json
        res = _load_mode_result(tmp_path, "dyn")
        assert res.cycle.startswith("SIM_RESULT_CORRUPT")

    def test_aclnn_result_corrupt(self, tmp_path):
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_aclnn_result

        (tmp_path / "final_result.json").write_text('{"ok": true')  # truncated json
        res = _load_aclnn_result(tmp_path)
        assert isinstance(res, ApiProfilingResult)
        assert res.failed()
        assert "SIM_RESULT_CORRUPT" in str(res.api_prof)

    def test_kernel_result_missing_with_wrapper_error(self, tmp_path):
        """A wrapper crash must surface its traceback, not a bare MISSING."""
        from ttk.core_modules.simulator.sim_profiling import _load_mode_result

        (tmp_path / "dyn").mkdir()
        (tmp_path / "wrapper_error.json").write_text(
            'Traceback (most recent call last):\n'
            '  File "wrapper.py", line 12, in main\n'
            'RuntimeError: boom')
        res = _load_mode_result(tmp_path, "dyn")
        assert res.cycle.startswith("SIM_RESULT_MISSING")
        assert "RuntimeError: boom" in res.cycle

    def test_aclnn_result_missing_with_wrapper_error(self, tmp_path):
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult
        from ttk.core_modules.simulator.sim_profiling import _load_aclnn_result

        (tmp_path / "wrapper_error.json").write_text("RuntimeError: boom")
        res = _load_aclnn_result(tmp_path)
        assert isinstance(res, ApiProfilingResult)
        assert res.failed()
        assert "RuntimeError: boom" in str(res.api_prof)


class TestSkipTeardown:
    """skip_teardown guards the camodel teardown busy-spin (rtCtxDestroy)."""

    def test_kernel_wrapper_skips_teardown(self, tmp_path):
        from ttk.core_modules.simulator import case_writer
        from ttk.core_modules.simulator.wrapper import write_kernel_wrapper

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_kernel_wrapper(sw, case_path)
        assert "skip_teardown=True" in wrapper.read_text()

    def test_aclnn_wrapper_skips_teardown(self, tmp_path):
        from ttk.core_modules.simulator import case_writer
        from ttk.core_modules.simulator.wrapper import write_aclnn_wrapper

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        case_path = case_writer.ensure_case_dir(sw, "caseA")
        wrapper = write_aclnn_wrapper(sw, case_path)
        assert "skip_teardown=True" in wrapper.read_text()

    def test_rts_skip_teardown_noop(self):
        """skip_teardown=True: destroy_context/reset skip api_call entirely."""
        from ttk.core_modules.runtime.rts_interface import RTSInterface

        dev = object.__new__(RTSInterface)  # bypass ctypes.CDLL load in __init__
        dev.skip_teardown = True
        dev.context = None
        dev.context_storage = []
        dev.device_id = 0
        calls = []
        dev.api_call = lambda *a, **k: calls.append(a)
        dev.destroy_context()
        dev.reset()
        assert calls == []

    def test_rts_teardown_still_active_by_default(self):
        """skip_teardown=False keeps the original validation before api_call."""
        from ttk.core_modules.runtime.rts_interface import RTSInterface

        dev = object.__new__(RTSInterface)
        dev.skip_teardown = False
        dev.context = None
        dev.context_storage = []
        # Attributes __del__->reset() touches when the object is GC'd.
        dev.device_id = None
        dev.kernel_binary_storage = {}
        dev.memory_manager = {}
        dev.size_info_storage = []
        calls = []
        dev.api_call = lambda *a, **k: calls.append(a)
        with pytest.raises(ValueError):
            dev.destroy_context()  # context not in storage -> raises, no api_call
        assert calls == []

    def test_aclnn_interface_passes_skip_teardown(self, monkeypatch):
        import ttk.core_modules.aclnn.acl_interface as acl_mod

        captured = {}

        class FakeRTS:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def reset(self, *a, **k):
                pass

            def destroy_context(self, *a, **k):
                pass

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        acl_mod.AclInterface("Ascend950", True, skip_teardown=True)
        assert captured.get("skip_teardown") is True

    def test_aclnn_reset_skipped_when_skip_teardown(self, monkeypatch):
        """skip_teardown=True: AclInterface.reset() must not call
        _acl_reset_device() (aclrtResetDevice can busy-spin on camodel)."""
        import ttk.core_modules.aclnn.acl_interface as acl_mod

        calls = []

        class FakeRTS:
            def __init__(self, **kwargs):
                pass

            def reset(self, *a, **k):
                calls.append("rts_reset")

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        dev = acl_mod.AclInterface("Ascend950", True, skip_teardown=True)
        dev._owns_acl_runtime = True
        dev._acl_reset_device = lambda: calls.append("acl_reset_device")
        dev.reset()
        assert calls == []  # neither aclrtResetDevice nor rts reset is issued

    def test_aclnn_reset_still_active_by_default(self, monkeypatch):
        """skip_teardown=False keeps the original aclrtResetDevice path."""
        import ttk.core_modules.aclnn.acl_interface as acl_mod

        calls = []

        class FakeRTS:
            def __init__(self, **kwargs):
                pass

            def reset(self, *a, **k):
                calls.append("rts_reset")

        monkeypatch.setattr(acl_mod, "RTSInterface", FakeRTS)
        monkeypatch.setattr(acl_mod.ctypes, "CDLL", lambda *a, **k: object())
        monkeypatch.setattr(acl_mod.AclInterface, "_on_exit", lambda self: None)
        dev = acl_mod.AclInterface("Ascend950", True)
        dev._owns_acl_runtime = True
        dev._acl_reset_device = lambda: calls.append("acl_reset_device")
        dev.reset()
        assert "acl_reset_device" in calls
        assert "rts_reset" in calls


class TestCannsimBackend:
    """record/report run the CANN-bundled cannsim, not the repo source."""

    def test_latest_record_dir_matches_cannsim_prefix(self, tmp_path):
        from ttk.core_modules.simulator.npusim_runner import _latest_record_dir

        (tmp_path / "cannsim_20260810_abc").mkdir()
        (tmp_path / "npusim_20260810_old").mkdir()  # legacy prefix must not match
        d = _latest_record_dir(tmp_path)
        assert d.name == "cannsim_20260810_abc"

    def test_latest_record_dir_empty_raises(self, tmp_path):
        from ttk.core_modules.simulator.npusim_runner import _latest_record_dir

        with pytest.raises(RuntimeError, match="cannsim record produced no"):
            _latest_record_dir(tmp_path)

    def test_locate_cannsim_prefers_toolkit_home(self, tmp_path, monkeypatch):
        from ttk.core_modules.simulator.npusim_runner import locate_cannsim_executable

        cand = tmp_path / "ascend" / "bin" / "cannsim"
        cand.parent.mkdir(parents=True)
        cand.write_text("#!/usr/bin/env python3\nfrom cannsim.main import main\n")
        cand.chmod(0o755)
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "ascend"))
        monkeypatch.setenv("PATH", str(tmp_path))  # no cannsim on PATH
        assert locate_cannsim_executable() == cand.resolve()

    def test_locate_cannsim_rejects_non_python_script(self, tmp_path, monkeypatch):
        from ttk.core_modules.simulator.npusim_runner import locate_cannsim_executable

        cand = tmp_path / "ascend" / "bin" / "cannsim"
        cand.parent.mkdir(parents=True)
        cand.write_bytes(b"\x7fELF\x02\x01")  # an unrelated binary on PATH
        cand.chmod(0o755)
        monkeypatch.setenv("ASCEND_TOOLKIT_HOME", str(tmp_path / "ascend"))
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(RuntimeError, match="CANN cannsim not found"):
            locate_cannsim_executable()

    def test_latest_record_dir_excludes_pre_existing(self, tmp_path):
        from ttk.core_modules.simulator.npusim_runner import _latest_record_dir

        old = tmp_path / "cannsim_20260809_old"
        new = tmp_path / "cannsim_20260810_new"
        old.mkdir()
        new.mkdir()
        # before 只含旧目录 → 返回本次新建的新目录
        d = _latest_record_dir(tmp_path, before=[old.name])
        assert d.name == new.name

    def test_locate_cannsim_raises_when_missing(self, tmp_path, monkeypatch):
        from ttk.core_modules.simulator.npusim_runner import locate_cannsim_executable

        monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(RuntimeError, match="CANN cannsim not found"):
            locate_cannsim_executable()

    def test_build_sim_env_has_no_repo_path(self, tmp_path, monkeypatch):
        from ttk.core_modules.simulator.npusim_runner import build_sim_env

        monkeypatch.delenv("PYTHONPATH", raising=False)
        sw = SWITCHES()
        sw.root_path = str(tmp_path)
        env = build_sim_env(sw)
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert str(tmp_path) in parts  # ttk root present
        assert not any("npu-simulator" in p for p in parts)
        assert env.get("CANNSIM_NO_DELAY") == "1"
        assert env.get("NPUSIM_NO_DELAY") == "1"

    def test_cannsim_cmd_uses_sys_executable_and_script(self, monkeypatch):
        from ttk.core_modules.simulator import npusim_runner

        fake_script = Path("/opt/ascend/bin/cannsim")
        monkeypatch.setattr(npusim_runner, "locate_cannsim_executable",
                            lambda: fake_script)
        cmd = npusim_runner._cannsim_cmd()
        assert cmd == [sys.executable, str(fake_script)]


class TestSimRecordFailure:
    """run_record failures surface as structured fail results, not crashes."""

    def test_run_aclnn_sim_returns_fail_when_record_raises(self, tmp_path, monkeypatch):
        import ttk.core_modules.simulator.sim_profiling as sim_mod
        from ttk.core_modules.npu.op_api.profiling_structure import ApiProfilingResult

        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        monkeypatch.setattr(sim_mod, "get_global_storage", lambda: sw)

        def _boom(*a, **k):
            raise RuntimeError("cannsim not found")

        monkeypatch.setattr(sim_mod, "run_record", _boom)
        context = SimpleNamespace(is_valid=True, testcase_name="caseA")
        res = sim_mod.run_aclnn_sim(context, 0)
        assert isinstance(res, ApiProfilingResult)
        assert res.failed()
        assert "SIM_RECORD_FAILED" in str(res.api_prof)

    def test_run_kernel_sim_returns_fail_when_record_raises(self, tmp_path, monkeypatch):
        import ttk.core_modules.operator as operator_pkg
        import ttk.core_modules.simulator.sim_profiling as sim_mod
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult

        # OpInfoKeeper lazily loads op-info config which needs a real CANN env.
        class _FakeOpInfoKeeper:
            def op_output_defined(self, op_name):
                return False

        monkeypatch.setattr(operator_pkg, "OpInfoKeeper", _FakeOpInfoKeeper)
        sw = SWITCHES()
        sw.sim_output_dir = str(tmp_path)
        monkeypatch.setattr(sim_mod, "get_global_storage", lambda: sw)
        monkeypatch.setattr(sim_mod, "_construct_param",
                            lambda *a, **k: _DummyRTSProfilingParam())
        monkeypatch.setattr(sim_mod, "_validate_param", lambda p: None)

        def _boom(*a, **k):
            raise RuntimeError("record crashed")

        monkeypatch.setattr(sim_mod, "run_record", _boom)
        context = SimpleNamespace(testcase_name="caseA", op_name="Add")
        dyn, cst, bin_ = sim_mod.run_kernel_sim(context)
        assert isinstance(dyn, RTSProfilingResult)
        assert "SIM_RECORD_FAILED" in str(dyn.cycle)
        assert cst.cycle == "SUPPRESSED"
        assert bin_.cycle == "SUPPRESSED"


class TestPerfStatusSimUnknown:
    """NPUSim reports cycle=UNKNOWN; perf_status must not fail on it."""

    @staticmethod
    def _ctx(monkeypatch, mode):
        import ttk.core_modules.npu.op.profiling as profiling_mod
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult
        from ttk.utilities.classes import SWITCHES

        sw = SWITCHES()
        sw.mode = mode
        monkeypatch.setattr(profiling_mod, "get_global_storage", lambda: sw)
        return SimpleNamespace(
            dyn_prof_result=RTSProfilingResult("UNKNOWN"),
            cst_prof_result=RTSProfilingResult.fail("SUPPRESSED"),
            bin_prof_result=RTSProfilingResult.fail("SUPPRESSED"),
        )

    def test_sim_unknown_cycle_is_not_a_failure(self, monkeypatch):
        from ttk.core_modules.npu.op.profiling import handle_profiling_result
        from ttk.utilities.classes import MODE

        ctx = self._ctx(monkeypatch, MODE.ASCEND_CAMODEL)
        assert handle_profiling_result(ctx) == "PASS"

    def test_real_device_unknown_cycle_still_fails(self, monkeypatch):
        from ttk.core_modules.npu.op.profiling import handle_profiling_result
        from ttk.utilities.classes import MODE

        ctx = self._ctx(monkeypatch, MODE.ASCEND_ONBOARD)
        assert handle_profiling_result(ctx) == "FAIL"

    def test_sim_real_cycle_still_passes(self, monkeypatch):
        import ttk.core_modules.npu.op.profiling as profiling_mod
        from ttk.core_modules.npu.op.profiling import handle_profiling_result
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult
        from ttk.utilities.classes import MODE, SWITCHES

        sw = SWITCHES()
        sw.mode = MODE.ASCEND_CAMODEL
        monkeypatch.setattr(profiling_mod, "get_global_storage", lambda: sw)
        ctx = SimpleNamespace(
            dyn_prof_result=RTSProfilingResult(100.5),
            cst_prof_result=RTSProfilingResult.fail("SUPPRESSED"),
            bin_prof_result=RTSProfilingResult.fail("SUPPRESSED"),
        )
        assert handle_profiling_result(ctx) == "PASS"

    def test_real_device_real_cycle_still_passes(self, monkeypatch):
        import ttk.core_modules.npu.op.profiling as profiling_mod
        from ttk.core_modules.npu.op.profiling import handle_profiling_result
        from ttk.core_modules.npu.op.profiling_structure import RTSProfilingResult
        from ttk.utilities.classes import MODE, SWITCHES

        sw = SWITCHES()
        sw.mode = MODE.ASCEND_ONBOARD
        monkeypatch.setattr(profiling_mod, "get_global_storage", lambda: sw)
        ctx = SimpleNamespace(
            dyn_prof_result=RTSProfilingResult(100.5),
            cst_prof_result=RTSProfilingResult.fail("CST_OFF"),
            bin_prof_result=RTSProfilingResult.fail("BIN_OFF"),
        )
        assert handle_profiling_result(ctx) == "PASS"
