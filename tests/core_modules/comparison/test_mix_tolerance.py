# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""MixToleranceComparison 单元测试：逐元素条件、matched_ratio 边界、max_abs_error 硬上限、NaN/Inf 真值表。"""

import numpy as np
import pytest

from ttk.core_modules.comparison.mix_tolerance import MixToleranceComparison

# float32 表值（resolve_tolerance 解析的最终值）
FP32 = {"rtol": 2**-10, "atol": 2**-16, "required_matched_ratio": 0.99, "max_abs_error_limit": 1e-2}


def _impl(actual, golden, dtype="float32", options=None):
    """返回 EachCompareResult。options 缺省用 float32 表值（resolve_tolerance 解析好的最终值）。"""
    c = MixToleranceComparison(np.asarray(actual), np.asarray(golden), 0, dtype, options or FP32)
    return c.compare_impl()


def _run(actual, golden, dtype="float32", options=None):
    """返回 compare() 的 4-tuple。"""
    c = MixToleranceComparison(np.asarray(actual), np.asarray(golden), 0, dtype, options or FP32)
    return c.compare()


# —— 逐元素通过条件：|actual - golden| <= atol + rtol * |golden| ——
def test_exact_match_passes():
    r = _impl([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0
    assert r.metrics["max_abs_error"] == 0.0


def test_atol_covers_small_golden_no_divzero():
    """golden=0 时 atol 兜底（天然避免除零）：err=atol 过、err=2*atol 不过。"""
    atol = FP32["atol"]
    ok = _impl([atol], [0.0])
    assert ok.is_pass is True
    bad = _impl([2 * atol], [0.0])
    assert bad.is_pass is False  # 单元素 ratio=0 < 0.99


def test_rtol_covers_large_golden():
    """大值场景走相对容差：golden=100、err=5e-3（> atol，< 硬上限）在 rtol 预算内通过；rtol=0 则失败。"""
    r = _impl([100.005], [100.0])  # 预算 = atol + rtol*100 ≈ 0.098
    assert r.is_pass is True
    r0 = _impl([100.005], [100.0], options=dict(FP32, rtol=0.0))
    assert r0.is_pass is False  # atol=2^-16 兜不住 5e-3


# —— 整体通过条件：matched_ratio >= required 且 max_abs_error <= 硬上限 ——
def test_ratio_boundary_99_percent():
    """100 元素 1 个超元素容差（err=5e-3 < 硬上限）→ ratio=0.99 恰好达标（>=）；2 个 → 0.98 FAIL。"""
    g = np.ones(100)
    a1 = g.copy()
    a1[0] = 1.0 + 5e-3  # 超元素预算（≈9.9e-4）但低于硬上限 1e-2
    r1 = _impl(a1, g)
    assert r1.metrics["matched_ratio"] == 0.99
    assert r1.is_pass is True

    a2 = g.copy()
    a2[:2] = 1.0 + 5e-3
    r2 = _impl(a2, g)
    assert r2.metrics["matched_ratio"] == 0.98
    assert r2.is_pass is False


def test_max_abs_error_hard_limit_beats_ratio():
    """ratio 达标但单点绝对误差超硬上限 → FAIL（硬上限拦灾难性离群点）。"""
    g = np.ones(100)
    a = g.copy()
    a[0] = 11.0  # err=10 > 1e-2，ratio=0.99 达标
    r = _impl(a, g)
    assert r.is_pass is False
    assert r.metrics["max_abs_error"] == pytest.approx(10.0)
    assert "max_abs_error" in r.metrics["reason"]


# —— NaN/Inf 真值表 ——
@pytest.mark.parametrize(
    ("a", "g", "expect_pass"),
    [
        (np.nan, np.nan, True),  # 都 NaN → 一致
        (np.inf, np.inf, True),  # 都 +Inf → 一致
        (np.inf, -np.inf, False),  # Inf 异号 → 无界误差
        (np.nan, 1.0, False),  # NaN vs 有限 → 无界误差
        (1.0, np.inf, False),  # 有限 vs Inf → 无界误差
    ],
)
def test_nan_inf_truth_table(a, g, expect_pass):
    r = _impl([a], [g])
    assert r.is_pass == expect_pass


def test_mismatch_sets_max_abs_error_none():
    """NaN/Inf 不一致 → max_abs_error 无界，metrics 中以 None 表示（保 CSV eval 往返）。"""
    r = _impl([np.nan, 1.0], [1.0, 1.0])
    assert r.is_pass is False
    assert r.metrics["max_abs_error"] is None
    assert "NaN/Inf mismatch" in r.metrics["reason"]


def test_all_nan_consistent_passes():
    """全 NaN 且一致 → matched（max_abs_error=0）。"""
    r = _impl([np.nan, np.nan], [np.nan, np.nan])
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0


# —— diff_idx 排序：误差降序（NaN 视作 +inf 最前）——
def test_diff_idx_worst_first():
    g = np.ones(4)
    a = np.array([1.001, np.nan, 1.02, 1.004])  # err: 1e-3, nan, 2e-2, 4e-3
    r = _impl(a, g)
    assert r.is_pass is False
    assert list(r.diff_index) == [1, 2, 3, 0]


# —— 结构 / 边界 ——
def test_empty_arrays_pass():
    precision, _l, is_pass, _m = _run([], [])
    assert precision == "100%"
    assert is_pass is True


def test_size_mismatch_fails():
    r = _impl([1.0, 2.0], [1.0])
    assert r.is_pass is False
    assert r.precision == "2 vs 1"


def test_metrics_are_plain_literals():
    """metrics 全 Python 字面量（CSV eval 往返）。"""
    r = _impl([1.0, 2.0], [1.0, 2.0])
    assert eval(repr(r.metrics)) == r.metrics  # noqa: S307


def test_spec_override_params_take_effect():
    """resolve 解析后的 override 参数生效：rtol=0.5 时 err=4e-3（> 默认预算 ≈9.9e-4，< 硬上限）可通过。"""
    r = _impl([1.004], [1.0], options=dict(FP32, rtol=0.5))
    assert r.is_pass is True
    assert r.metrics["rtol"] == 0.5
    assert _impl([1.004], [1.0]).is_pass is False


def test_fp8_output_vs_high_precision_golden():
    """fp8 输出走混合容差：ml_dtypes fp8 promote 到 float32 后与高精度 golden 比对。"""
    ml_dtypes = pytest.importorskip("ml_dtypes")
    # float8_e5m2 表值：rtol=2^-1, atol=2^-3, max_abs_error_limit=max(1e-1, 32*2^-2)=8.0
    opts = {"rtol": 2**-1, "atol": 2**-3, "required_matched_ratio": 0.99, "max_abs_error_limit": 8.0}
    a = np.array([1.0, 2.0, 4.0], ml_dtypes.float8_e5m2)
    g = np.array([1.0, 2.1, 4.0], np.float32)  # 单标杆：高精度 golden
    r = MixToleranceComparison(a, g, 0, "float8_e5m2", opts).compare_impl()
    assert r.is_pass is True
    assert r.metrics["matched_ratio"] == 1.0
    assert r.metrics["max_abs_error"] == pytest.approx(0.1, abs=1e-6)
