#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.npu.op.profiling_structure:
- ComparisonResult: set/get (5 fields: dyn, cst, bin, bre, passed)
- RTSProfilingResult: constructor, fail(), oob_status
- ProfilingReturnStructure: defaults, kernel_execute_failed, get_titles
"""

import pytest
from unittest.mock import patch, MagicMock

from ttk.core_modules.npu.op.profiling_structure import (
    ComparisonResult, RTSProfilingResult, ProfilingReturnStructure
)


@pytest.fixture(autouse=True)
def _mock_global_storage():
    mock_gs = MagicMock()
    mock_gs.custom_columns = None
    mock_gs.dev_plat = "Ascend910B2"
    mock_gs.oom_enabled.return_value = False
    with patch('ttk.core_modules.npu.op.profiling_structure.get_global_storage', return_value=mock_gs):
        yield mock_gs


class TestComparisonResult:

    def test_default_values(self):
        cr = ComparisonResult("INIT")
        assert cr.dyn_precision == "INIT"
        assert cr.cst_precision == "INIT"
        assert cr.bin_precision == "INIT"
        assert cr.passed == "INIT"

    def test_set_and_get(self):
        cr = ComparisonResult(None)
        result = cr.set("d", "c", "b", "PASS")
        assert result is cr
        assert cr.dyn_precision == "d"
        assert cr.cst_precision == "c"
        assert cr.bin_precision == "b"
        assert cr.passed == "PASS"

    def test_get_returns_all_slots(self):
        cr = ComparisonResult("X")
        values = cr.get()
        assert len(values) == len(ComparisonResult.__slots__)
        assert all(v == "X" for v in values)


class TestRTSProfilingResult:

    def test_constructor_defaults(self):
        r = RTSProfilingResult()
        assert r.cycle is None
        assert r.output_bytes == (None,)
        assert r.oob == "UNKNOWN"

    def test_constructor_with_values(self):
        r = RTSProfilingResult(cycle=100, output_bytes=(200,), oob="PASS")
        assert r.cycle == 100
        assert r.output_bytes == (200,)
        assert r.oob == "PASS"

    def test_fail_classmethod(self):
        r = RTSProfilingResult.fail("TIMEOUT")
        assert r.cycle == "TIMEOUT"
        assert r.output_bytes == ("TIMEOUT",)
        assert r.oob == "UNKNOWN"

    def test_oob_status_pass_when_empty(self):
        r = RTSProfilingResult(oob="")
        assert r.oob_status == "PASS"

    def test_oob_status_pass_when_none(self):
        r = RTSProfilingResult(oob=None)
        assert r.oob_status == "PASS"

    def test_oob_status_pass_normal(self):
        r = RTSProfilingResult(oob="PASS,SOMETHING")
        assert r.oob_status == "PASS"

    def test_oob_status_fail(self):
        r = RTSProfilingResult(oob="FAIL,SOMETHING")
        assert r.oob_status == "FAIL"


class TestProfilingReturnStructure:

    def test_constructor_defaults(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        assert prs.dyn_perf_us is None
        assert prs.cst_perf_us is None
        assert prs.bin_perf_us is None
        assert prs.perf_status is None
        assert prs.precision_status is None
        assert prs.memory_oob_status is None
        assert prs.soc == "Ascend910B2"

    def test_constructor_with_default_value(self, _mock_global_storage):
        prs = ProfilingReturnStructure("N/A")
        assert prs.dyn_perf_us == "N/A"
        assert prs.cst_perf_us == "N/A"
        assert prs.precision_status == "N/A"

    def test_kernel_execute_failed_no_failure(self, _mock_global_storage):
        prs = ProfilingReturnStructure("100")
        assert not prs.kernel_execute_failed()

    def test_kernel_execute_failed_with_trap(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        prs.dyn_perf_us = "TRAP"
        prs.cst_perf_us = "100"
        prs.bin_perf_us = "100"
        assert prs.kernel_execute_failed()

    def test_kernel_execute_failed_with_launch_failed(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        prs.dyn_perf_us = "100"
        prs.cst_perf_us = "LAUNCH_FAILED"
        prs.bin_perf_us = "100"
        assert prs.kernel_execute_failed()

    def test_kernel_execute_failed_with_timeout(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        prs.dyn_perf_us = "100"
        prs.cst_perf_us = "100"
        prs.bin_perf_us = "TIMEOUT"
        assert prs.kernel_execute_failed()

    def test_kernel_execute_failed_with_comma_separated(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        prs.dyn_perf_us = "100,TRAP"
        prs.cst_perf_us = "100"
        prs.bin_perf_us = "100"
        assert prs.kernel_execute_failed()

    def test_get_titles(self, _mock_global_storage):
        titles = ProfilingReturnStructure.get_titles()
        assert len(titles) > 0
        assert "dyn_perf_us" in titles
        assert "cst_perf_us" in titles
        assert "precision_status" in titles

    def test_get_titles_custom(self, _mock_global_storage):
        _mock_global_storage.custom_columns = ["dyn_perf_us", "cst_perf_us"]
        titles = ProfilingReturnStructure.get_titles(custom=True)
        assert titles == ("dyn_perf_us", "cst_perf_us")

    def test_get_returns_tuple_of_values(self, _mock_global_storage):
        prs = ProfilingReturnStructure("N/A")
        values = prs.get()
        assert isinstance(values, tuple)
        assert len(values) == len(ProfilingReturnStructure.__slots__)

    def test_no_stc_slots_present(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        for slot in ("stc_name", "stc_block_dim", "stc_perf_us", "stc_throughput_gbs",
                     "stc_kernel_name",
                     "stc_compile_s", "stc_precision", "stc_oob_result",
                     "stc_workspaces", "stc_op_pattern", "stc_rl_query_result"):
            assert not hasattr(prs, slot), f"Stc slot should be removed: {slot}"

    def test_no_rel_rst_slots(self, _mock_global_storage):
        prs = ProfilingReturnStructure()
        assert not hasattr(prs, "rel_precision")
        assert not hasattr(prs, "rst_precision")
