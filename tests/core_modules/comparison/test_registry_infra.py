# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""registry 基础设施单元测试：EachCompareResult 默认值、_to_numpy 转换、
ComparisonBase.compare() 4-tuple + _check_empty。"""
import numpy as np
import pytest

from ttk.core_modules.comparison.registry import ComparisonBase, EachCompareResult, _to_numpy, register_comparison


def test_each_compare_result_defaults():
    """EachCompareResult 默认值：precision/diff_index/is_pass/log/standard/metrics/error_info 及 metrics 独立 dict。"""
    r = EachCompareResult(1)
    assert r.precision == 1
    assert r.diff_index is None
    assert r.is_pass is False
    assert r.log == ""
    assert r.standard == ""
    assert r.metrics == {}
    assert r.error_info is None
    # distinct default dicts (not shared)
    assert EachCompareResult(1).metrics is not EachCompareResult(1).metrics


def test_to_numpy_passthrough():
    """numpy 数组直接透传（同一对象）。"""
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert _to_numpy(a) is a


def test_to_numpy_torch_float():
    """torch float32 tensor → numpy float32 数组。"""
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _to_numpy(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [1.0, 2.0])


# 一个最小 ComparisonBase 子类用于测 _check_empty
@register_comparison("__test_dummy")
class _Dummy(ComparisonBase):
    STANDARD_NAME = "dummy"

    def compare_impl(self):
        return EachCompareResult(0.5, is_pass=True, standard="dummy",
                                 metrics={"k": 1})


@pytest.mark.parametrize("actual, golden, expected", [
    pytest.param(np.array([]), np.array([]), (True, 1), id="both_empty"),
])
def test_check_empty(actual, golden, expected):
    """_check_empty: 双方空 → PASS(precision=1)。"""
    c = _Dummy(actual, golden, 0, "float32", {})
    r = c._check_empty()
    assert r.is_pass is expected[0] and r.precision == expected[1]
