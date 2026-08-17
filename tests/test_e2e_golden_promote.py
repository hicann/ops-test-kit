#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Unit tests for the E2E golden Promote path.

Covers the three functions added when E2E golden gained `golden_mode=Promote`
under a cross_check tolerance:

  * `golden_generation._promote_raw_inputs`    - dtype lift itself
  * `profiling._needs_golden_promote`          - "is the standard cross_check?"
  * `profiling._generate_golden_maybe_promote` - set / restore the override

Existing promote tests (`test_golden_promote_wrap.py`, `test_aclnn_promote.py`)
only exercise the KERNEL and ACLNN paths, so none of the above was covered.
"""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from ttk.core_modules.framework_api import profiling as _prof
from ttk.core_modules.framework_api.golden_generation import (
    _promote_raw_inputs as promote_raw_inputs,
)
from ttk.core_modules.framework_api.profiling import (
    _generate_golden_maybe_promote as generate_golden_maybe_promote,
)
from ttk.core_modules.framework_api.profiling import (
    _needs_golden_promote as needs_golden_promote,
)

# 被测函数以别名导入而非 `模块._私有` 属性访问:后者会被判受保护成员访问(G.CLS.11)。
# 仍保留 _prof 模块引用,供 patch.object 按名字打桩使用(传的是字符串,不构成属性访问)。


class _Testcase:
    """Minimal stand-in for TestcaseE2e (plain attributes, no __slots__)."""

    def __init__(self, flat_tensor_dtypes=None, golden_mode_override=None):
        self.testcase_name = "ut_case"
        self.api_name = "torch.add"
        self.flat_tensor_dtypes = flat_tensor_dtypes
        self.golden_mode_override = golden_mode_override
        self.flat_precision_tolerances = None
        self.flat_absolute_precision = None


class _Switches:
    """Minimal stand-in for the SWITCHES object."""

    def __init__(self, golden_mode=None):
        self.golden_mode = golden_mode
        self.plugin_path = None
        self.compare_method = None


def _standard(token):
    """构造一个只带 token 字段的 tolerance standard 替身。"""
    return type("_Std", (), {"token": token})()


# ----------------------------------------------------------------------------
# _promote_raw_inputs
# ----------------------------------------------------------------------------
def test_promote_lifts_float_dtypes():
    """fp16/bf16 -> fp32, fp32 -> fp64, per DTYPE_PROMOTE_MAP."""
    testcase = _Testcase(flat_tensor_dtypes=["float16", "float32"])
    raw = [np.ones(4, dtype=np.float16), np.ones(4, dtype=np.float32)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float64
    # 原数组不被就地改写
    assert raw[0].dtype == np.float16
    assert raw[1].dtype == np.float32


def test_promote_leaves_integer_dtypes_untouched():
    """整型不在 DTYPE_PROMOTE_MAP 中,必须原样保留(含对象本身)。"""
    testcase = _Testcase(flat_tensor_dtypes=["int32", "int64", "bool"])
    raw = [
        np.ones(2, dtype=np.int32),
        np.ones(2, dtype=np.int64),
        np.ones(2, dtype=bool),
    ]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert [a.dtype for a in out] == [np.dtype(np.int32), np.dtype(np.int64), np.dtype(bool)]
    for got, want in zip(out, raw):
        assert got is want


def test_promote_skips_none_entry():
    """可选入参为 None 时跳过,不得抛异常,且后续下标不错位。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16", "float16", "float16"])
    raw = [np.ones(2, dtype=np.float16), None, np.ones(2, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1] is None
    assert out[2].dtype == np.float32


@pytest.mark.parametrize("mode", [None, "Enable", "Disable"])
def test_promote_returns_original_object_when_mode_off(mode):
    """非 Promote 模式必须原样返回同一个 list 对象,不产生拷贝开销。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])
    raw = [np.ones(2, dtype=np.float16)]

    assert promote_raw_inputs(testcase, raw, _Switches(golden_mode=mode)) is raw


def test_promote_override_takes_precedence_over_switches():
    """testcase.golden_mode_override 优先于 switches.golden_mode。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"], golden_mode_override="Promote")
    raw = [np.ones(2, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode=None))

    assert out[0].dtype == np.float32


def test_promote_returns_original_when_dtype_list_missing():
    """拿不到 flat_tensor_dtypes 时原样返回,不能凭空猜 dtype。"""
    testcase = _Testcase(flat_tensor_dtypes=None)
    raw = [np.ones(2, dtype=np.float16)]

    assert promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote")) is raw


def test_promote_leaves_extra_inputs_alone():
    """raw_inputs 比 dtype 列表长时,多出的部分保持原样而非报错。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])
    raw = [np.ones(2, dtype=np.float16), np.ones(2, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float16


def test_promote_keeps_original_and_warns_when_astype_fails(caplog):
    """astype 抛 TypeError/ValueError 时保留原值并给出 warning(不再是静默 debug)。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])

    class _BadArray(np.ndarray):
        def astype(self, *args, **kwargs):
            raise ValueError("boom")

    bad = np.ones(2, dtype=np.float16).view(_BadArray)
    with caplog.at_level(logging.WARNING):
        out = promote_raw_inputs(testcase, [bad], _Switches(golden_mode="Promote"))

    assert out[0] is bad
    assert any("promote input#0" in record.message for record in caplog.records)


# ----------------------------------------------------------------------------
# _needs_golden_promote
# ----------------------------------------------------------------------------
def test_needs_promote_false_without_reference():
    """没有竞品参考数据时不需要 Promote。"""
    assert needs_golden_promote(_Testcase(), _Switches(), None) is False
    assert needs_golden_promote(_Testcase(), _Switches(), []) is False


def test_needs_promote_true_for_cross_check():
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", return_value=[_standard("cross_check")]
    ):
        assert needs_golden_promote(_Testcase(), _Switches(), [np.ones(2)]) is True


def test_needs_promote_false_for_other_standard():
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", return_value=[_standard("stat_rel_err")]
    ):
        assert needs_golden_promote(_Testcase(), _Switches(), [np.ones(2)]) is False


def test_needs_promote_true_when_any_output_is_cross_check():
    standards = [_standard("stat_rel_err"), _standard("cross_check")]
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", return_value=standards
    ):
        assert needs_golden_promote(_Testcase(), _Switches(), [np.ones(2), np.ones(2)]) is True


@pytest.mark.parametrize("exc", [KeyError("k"), TypeError("t"), ValueError("v")])
def test_needs_promote_falls_back_with_warning_on_spec_error(exc, caplog):
    """tolerance spec 配置有误时回退 False,但必须留 WARNING —— 静默跳过正是本次要修的症状。"""
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", side_effect=exc
    ):
        with caplog.at_level(logging.WARNING):
            got = needs_golden_promote(_Testcase(), _Switches(), [np.ones(2)])

    assert got is False
    assert any("golden Promote skipped" in record.message for record in caplog.records)


def test_needs_promote_does_not_swallow_unexpected_exception():
    """非配置类异常不应被吞掉,以免掩盖真实缺陷。"""
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", side_effect=RuntimeError("unexpected")
    ):
        with pytest.raises(RuntimeError):
            needs_golden_promote(_Testcase(), _Switches(), [np.ones(2)])


# ----------------------------------------------------------------------------
# _generate_golden_maybe_promote
# ----------------------------------------------------------------------------
def test_maybe_promote_leaves_override_untouched_when_not_needed():
    testcase = _Testcase()
    seen = {}

    def _fake_generate(case, raw_inputs, switches, backend):
        seen["mode"] = case.golden_mode_override
        return ["golden"]

    with patch.object(_prof, "_needs_golden_promote", return_value=False), patch.object(
        _prof, "_generate_golden_data", side_effect=_fake_generate
    ):
        out = generate_golden_maybe_promote(testcase, [], _Switches(), "cpu", [np.ones(2)])

    assert out == ["golden"]
    assert seen["mode"] is None
    assert testcase.golden_mode_override is None


def test_maybe_promote_sets_promote_then_restores():
    testcase = _Testcase(golden_mode_override="Enable")
    seen = {}

    def _fake_generate(case, raw_inputs, switches, backend):
        seen["mode"] = case.golden_mode_override
        return ["golden"]

    with patch.object(_prof, "_needs_golden_promote", return_value=True), patch.object(
        _prof, "_generate_golden_data", side_effect=_fake_generate
    ):
        out = generate_golden_maybe_promote(testcase, [], _Switches(), "cpu", [np.ones(2)])

    assert out == ["golden"]
    assert seen["mode"] == "Promote"              # 生成期间已抬升
    assert testcase.golden_mode_override == "Enable"  # 结束后恢复原值


def test_maybe_promote_restores_override_even_when_generation_raises():
    testcase = _Testcase(golden_mode_override=None)

    with patch.object(_prof, "_needs_golden_promote", return_value=True), patch.object(
        _prof, "_generate_golden_data", side_effect=RuntimeError("golden boom")
    ):
        with pytest.raises(RuntimeError):
            generate_golden_maybe_promote(testcase, [], _Switches(), "cpu", [np.ones(2)])

    assert testcase.golden_mode_override is None
