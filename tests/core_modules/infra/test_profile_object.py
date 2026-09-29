# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for ttk.core_modules.infra.profile_object: header assembly & task result dispatch."""

from unittest.mock import MagicMock, patch

import pytest

from ttk.core_modules.infra.profile_object import ProfileObject
from ttk.core_modules.infra.task import TaskA, TaskType
from ttk.core_modules.testcase_manager.testcase_base import TestcaseBase


class _FakeProfileObject(ProfileObject):
    def setup(self):
        pass

    def possible_result_titles(self):
        return ("dyn_perf_us", "precision")

    def init_tasks(self, testcases):
        pass

    def apply_profile_success_result(self, testcase, result):
        return (("PASS", "PASS"), False)


@pytest.fixture
def switches():
    sw = MagicMock()
    sw.preserve_original_csv = False
    sw.custom_columns = None
    with patch("ttk.core_modules.infra.profile_object.get_global_storage", return_value=sw):
        yield sw


def _make_case(name="t0"):
    case = MagicMock(spec=TestcaseBase)
    case.testcase_name = name
    case.pick_data.return_value = (name, "Add")
    case.get_all_visible_headers.return_value = ("testcase_name", "op_name", "api_name", "input_shape")
    return case


def _make_po(switches):
    po = _FakeProfileObject(MagicMock(), MagicMock())
    po.case_input_title = ("testcase_name", "op_name")
    po.case_result_title = ("dyn_perf_us", "precision")
    return po


def test_profile_object_is_abstract():
    with pytest.raises(TypeError):
        ProfileObject(MagicMock(), MagicMock())


def test_reorder_moves_identity_headers_to_front(switches):
    po = _make_po(switches)
    out = po._reorder_input_titles(("testcase_name", "input_shape", "op_name", "api_name", "extra"))
    assert out == ("testcase_name", "op_name", "api_name", "input_shape", "extra")
    assert po._front_input_count == 3


def test_reorder_skips_when_custom_columns_set(switches):
    switches.custom_columns = ("testcase_name", "precision")
    po = _make_po(switches)
    out = po._reorder_input_titles(("testcase_name", "input_shape"))
    assert out == ("testcase_name", "input_shape")
    assert po._front_input_count == 0


def test_reorder_noop_when_testcase_name_not_first(switches):
    po = _make_po(switches)
    out = po._reorder_input_titles(("input_shape", "op_name"))
    assert out == ("input_shape", "op_name")
    assert po._front_input_count == 0


def test_assemble_row_splits_front(switches):
    po = _make_po(switches)
    po._front_input_count = 2
    out = po._assemble_row(("tc", "op", "extra"), ("perf", "prec"))
    assert out == ("tc", "op", "perf", "prec", "extra")


def test_assemble_row_no_split_when_front_zero(switches):
    po = _make_po(switches)
    po._front_input_count = 0
    out = po._assemble_row(("tc", "op"), ("perf", "prec"))
    assert out == ("tc", "op", "perf", "prec")


def test_profile_fail_result_puts_details_first(switches):
    po = _make_po(switches)
    out = po._profile_fail_result("FAILURE", "boom")
    assert out == ("boom", "FAILURE")


def test_profile_fail_result_empty_titles(switches):
    po = _make_po(switches)
    po.case_result_title = ()
    assert po._profile_fail_result("FAILURE", "boom") == ()


def test_handle_complete_profile_path(switches):
    po = _make_po(switches)
    task = TaskA(_make_case(), int, (), type=TaskType.PROFILE)
    out, kill = po.handle_task_result_complete(task, object())
    assert out == ("t0", "Add", "PASS", "PASS")
    assert kill is False


def test_handle_runtime_error_profile_path(switches):
    po = _make_po(switches)
    task = TaskA(_make_case(), int, (), type=TaskType.PROFILE)
    out = po.handle_task_result_runtime_error(task, RuntimeError("crashed"), pid=99)
    assert out == ("t0", "Add", "crashed", "FAILURE")


def test_output_titles_filters_by_custom_columns(switches):
    switches.custom_columns = ("testcase_name", "precision")
    po = _make_po(switches)
    case = _make_case()
    header = po.output_titles(case, ["testcase_name", "op_name", "input_shape"])
    assert "testcase_name" in header
    assert "precision" in header
    assert "op_name" not in header
    assert "dyn_perf_us" not in header


def test_e2e_crash_keeps_failure_status(switches):
    po = _make_po(switches)
    po.case_result_title = ("precision_status", "eager_precision")
    assert po._profile_fail_result("PROFILE_CRASH", "exit -9") == ("FAIL", "PROFILE_CRASH")


def test_e2e_failure_keeps_schema_without_duplicate_parent_error(monkeypatch, caplog):
    from ttk.core_modules.framework_api.object import FrameworkApiProfileObject
    from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

    obj = object.__new__(FrameworkApiProfileObject)
    monkeypatch.setattr(obj, "_print_new_compare_failures", lambda *args: None)
    result = FrameworkApiReturnStructure()
    result.construct("0%", "FAIL", None)
    values, _ = obj.apply_profile_success_result(_make_case(), result)
    row = dict(zip(result.get_titles(), values))
    assert row["precision_status"] == "FAIL"
    assert "error_info" not in row
    assert not caplog.records


def test_e2e_missing_and_invalid_worker_results(monkeypatch, caplog):
    from ttk.core_modules.framework_api.object import FrameworkApiProfileObject
    from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

    obj = object.__new__(FrameworkApiProfileObject)
    monkeypatch.setattr(obj, "_print_new_compare_failures", lambda *args: None)
    task = TaskA(_make_case(), int, (), type=TaskType.PROFILE)
    values, _ = obj.apply_profile_success_result(task.testcase, object())
    row = dict(zip(FrameworkApiReturnStructure.get_titles(), values))
    assert row["precision_status"] == "FAIL"
    assert row["eager_precision"] == "INVALID_WORKER_RESULT"
    assert len(caplog.records) == 1
    caplog.clear()
    monkeypatch.setattr(obj, "_profile_normal_complete", lambda task, result: (result, False))
    result = obj.handle_task_result_none(task)
    assert result.precision_status == "FAIL"
    assert result.eager_precision == "NO_WORKER_RESULT"
    assert len(caplog.records) == 1


@pytest.mark.parametrize("status", ["FAIL", "FAILURE", "PROFILE_CRASH", "COMPILE_FAILURE", "TIMEOUT"])
def test_failure_summary_does_not_repeat_diagnostics(status, caplog):
    import csv
    import io
    from types import SimpleNamespace

    from ttk.core_modules.infra.instance_base import InstanceBase

    stream = io.StringIO()
    obj = SimpleNamespace(
        result_csv_writer=csv.writer(stream),
        result_csv_file=stream,
        _header_flushed=True,
        _precision_status_idx=0,
        pass_count=0,
        fail_count=0,
        other_count=0,
    )
    obj._update_summary = lambda row: InstanceBase._update_summary(obj, row)
    InstanceBase._flush(obj, (status,))
    assert obj.fail_count == 1
    assert not caplog.records
