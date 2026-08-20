# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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
