#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
mix_tolerance comparison — 生态算子开源精度标准（混合容差）
"""

# Standard Packages
import numpy as np

# Third-party Packages
from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


@register_comparison(["mixed", "mix_tolerance"])  # mixed=CLI 简写；mix_tolerance=Spec.tolerance 官方标准名
class MixToleranceComparison(ComparisonBase):
    STANDARD_NAME = "mix_tolerance"

    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        # rtol/atol/required_matched_ratio/max_abs_error_limit 均为 resolve_tolerance 解析好的最终值（不查表）
        actual, golden = self.output, self.golden
        if actual.size != golden.size:
            return EachCompareResult(
                f"{actual.size} vs {golden.size}",
                is_pass=False,
                standard="mix_tolerance",
                metrics={"standard": "mix_tolerance", "pass": False, "reason": "output size != golden size"},
            )
        T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
        a = actual.astype(T)
        g = golden.astype(T)
        rtol = self.tol_options["rtol"]
        atol = self.tol_options["atol"]
        required_ratio = self.tol_options["required_matched_ratio"]
        max_err_limit = self.tol_options["max_abs_error_limit"]

        with np.errstate(invalid="ignore", divide="ignore"):
            err = np.abs(a - g)
            a_nan, g_nan = np.isnan(a), np.isnan(g)
            a_inf, g_inf = np.isinf(a), np.isinf(g)
            same_nan = a_nan & g_nan  # 都 NaN（NaN 无符号）
            same_inf = a_inf & g_inf & (np.sign(a) == np.sign(g))  # 都 Inf 且同号
            mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
            finite = np.isfinite(a) & np.isfinite(g)
            # 逐元素通过条件：|actual - golden| <= atol + rtol * |golden|（atol 天然避免除零）
            elem_ok = same_nan | same_inf | (finite & (err <= atol + rtol * np.abs(g)))
            matched_ratio = float(elem_ok.sum()) / a.size

            # max_abs_error 取有限元素对的最大绝对误差；NaN/Inf 不一致视为无界误差 → 必超硬上限
            if mismatch.any():
                max_abs_error = float("inf")
            elif finite.any():
                max_abs_error = float(err[finite].max())
            else:  # 全非有限且一致
                max_abs_error = 0.0
            # 整体通过条件：matched_ratio 达标 且 max_abs_error 不超硬上限
            passed = matched_ratio >= required_ratio and max_abs_error <= max_err_limit

            if passed:
                diff_idx, reason = None, None
            else:  # 未通过元素按误差降序（NaN 视作 +inf 排最前）；完整列表不截断，log 控制打印数
                err_for_sort = np.where(np.isnan(err), np.inf, err)
                bad_idx = np.where(~elem_ok)[0]
                diff_idx = bad_idx[np.argsort(-err_for_sort[bad_idx], kind="stable")]
                parts = []
                if mismatch.any():
                    parts.append(FAIL_REASONS["nan_inf_mismatch"])
                exceeded = []
                if matched_ratio < required_ratio:
                    exceeded.append(f"matched_ratio({matched_ratio:.4f}<{required_ratio})")
                if max_abs_error > max_err_limit:
                    exceeded.append(f"max_abs_error({max_abs_error:.2e}>{max_err_limit:.2e})")
                if exceeded:
                    parts.append(FAIL_REASONS["threshold_exceeded"].format(metrics=", ".join(exceeded)))
                reason = "; ".join(parts)

        return EachCompareResult(
            matched_ratio,
            diff_index=diff_idx,
            is_pass=passed,
            standard="mix_tolerance",
            metrics=_metrics(rtol, atol, matched_ratio, required_ratio, max_abs_error, max_err_limit, passed, reason),
        )


def _metrics(rtol, atol, matched_ratio, required_ratio, max_abs_error, max_err_limit, passed, reason=None):
    # 全 Python 字面量，保 CSV eval 往返；max_abs_error=None（inf）当 NaN/Inf mismatch
    m = {
        "standard": "mix_tolerance",
        "rtol": rtol,
        "atol": atol,
        "matched_ratio": matched_ratio,
        "required_matched_ratio": required_ratio,
        "max_abs_error": None if np.isinf(max_abs_error) else max_abs_error,
        "max_abs_error_limit": max_err_limit,
        "pass": bool(passed),
    }
    if reason:
        m["reason"] = reason
    return m
