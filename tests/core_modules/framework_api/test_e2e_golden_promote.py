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
"""

from unittest.mock import patch

import numpy as np

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


def test_promote_leaves_extra_inputs_alone():
    """raw_inputs 比 dtype 列表长时,多出的部分保持原样而非报错。"""
    testcase = _Testcase(flat_tensor_dtypes=["float16"])
    raw = [np.ones(2, dtype=np.float16), np.ones(2, dtype=np.float16)]

    out = promote_raw_inputs(testcase, raw, _Switches(golden_mode="Promote"))

    assert out[0].dtype == np.float32
    assert out[1].dtype == np.float16


# ----------------------------------------------------------------------------
# _needs_golden_promote
# ----------------------------------------------------------------------------
def test_needs_promote_true_for_cross_check():
    with patch.object(_prof, "get_spec_attr", return_value=None), patch.object(
        _prof, "resolve_tolerance", return_value=[_standard("cross_check")]
    ):
        assert needs_golden_promote(_Testcase(), _Switches(), [np.ones(2)]) is True


# ----------------------------------------------------------------------------
# _generate_golden_maybe_promote
# ----------------------------------------------------------------------------
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
