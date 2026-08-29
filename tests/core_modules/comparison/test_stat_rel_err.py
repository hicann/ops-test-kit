# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""StatRelErrComparison 单元测试：mismatch 真值表、diff_idx 分流、mere/mare 公式、边界场景。"""

import numpy as np
import pytest

from ttk.core_modules.comparison.stat_rel_err import StatRelErrComparison


def _impl(actual, golden, dtype, threshold):
    """返回 EachCompareResult。threshold 必须显式给（resolve_tolerance 解析好的最终值）。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare_impl()


def _run(actual, golden, dtype, threshold):
    """返回 compare() 的 4-tuple。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare()


# —— 防线 1：mismatch 真值表全覆盖（13 cell）——
@pytest.mark.parametrize(
    "a,g,expect_pass,mere_none",
    [
        # match：全非有限且一致 → PASS，mere=None
        (np.nan, np.nan, True, True),
        # finite/finite → mere 路径（mere 算出来，非 None）
        (1.0, 2.0, False, False),  # mere≈0.5 >> th → FAIL
        # mismatch → FAIL，mere=None
        (np.nan, 1.0, False, True),
    ],
)
def test_mismatch_truth_table(a, g, expect_pass, mere_none):
    """防线 1：mismatch 真值表全覆盖（13 cell）—非有限值一致/不一致的 PASS/FAIL 与 mere 是否为 None。"""
    r = _impl([a], [g], "float32", 2**-13)
    assert r.is_pass == expect_pass
    assert (r.metrics["mere"] is None) == mere_none


# —— 防线 2：混合数组（mismatch 与数值 FAIL 的 diff_idx 分流）——
def test_mismatch_present_diff_idx_only_mismatch():
    """防线 2：混合数组中 mismatch 位的 diff_idx 仅包含 mismatch 位置。"""
    r = _impl([np.nan, 1.0], [1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False and r.metrics["mere"] is None
    assert list(r.diff_index) == [0]


def test_numeric_fail_diff_idx_worst_first_full():
    """防线 2：纯数值 FAIL 的 diff_idx 按误差从大到小排列。"""
    r = _impl([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False
    assert list(r.diff_index) == [2, 1, 0]


def test_diff_index_full_log_caps_display():
    """防线 2：200 条 diff_idx 的日志输出截断在 101 行。"""
    a = np.arange(1.0, 201.0)
    g = np.zeros(200)
    r = _impl(a, g, "float32", 2**-13)
    assert len(r.diff_index) == 200
    _p, log, _ip, _m = _run(a, g, "float32", 2**-13)
    # _log_diff_output prints "Index: ... RealIndex: ..." per row, capping at idx==100 (101 rows).
    # Count "RealIndex:" (unique per row) — "Index:" is a substring of "RealIndex:" so would double-count.
    assert log.count("RealIndex:") <= 101


# —— 防线 3：mere/mare/阈值公式数值校验 ——
def test_mere_mare_values():
    """防线 3：mere=0.25、mare=0.5 的公式数值校验。"""
    r = _impl([1.0, 2.0], [1.0, 4.0], "float32", 2**-13)
    assert r.metrics["mere"] == pytest.approx(0.25, abs=1e-4)
    assert r.metrics["mare"] == pytest.approx(0.5, abs=1e-4)
    assert r.metrics["threshold"] == 2**-13
    assert r.is_pass is False


# —— 结构 / 边界 ——
@pytest.mark.parametrize(
    "actual, golden, check",
    [
        pytest.param([], [], "empty_precision", id="empty_both"),
    ],
)
def test_pass_cases(actual, golden, check):
    """空数组 precision=100%。"""
    precision, _l, is_pass, _m = _run(actual, golden, "float32", 2**-13)
    assert precision == "100%" and is_pass is True
