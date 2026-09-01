# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""cross_check: 三方交叉校验（NPU / golden / third_party）。"""

import numpy as np

from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


def safe_div(num, den, err):
    """三比值的除法:分母一律夹小值域阈值 err。

    精度标准:正常值域的 mare/mere/rmse 三比值,分母按 dtype 取的 small_value
    (见 resolve.py)夹底。err 既防除零,也保证竞品误差落在噪声地板以下时不把比值
    放大——固定常数会让判定随输出量级漂移。
    """
    if np.isnan(num) or np.isnan(den):
        return float("inf")
    if np.isinf(num) and np.isinf(den):
        return 1.0  # both overflow -> ratio=1 (consistent)
    if np.isinf(num):
        return float("inf")  # NPU overflow, third_party finite -> NPU worse
    if num == 0 and den == 0:
        return 1.0  # both perfect match -> ratio=1 (consistent)
    return float(num / max(den, err))


@register_comparison(["cross_check"])
class CrossCheckComparison(ComparisonBase):
    STANDARD_NAME = "cross_check"

    def compare_impl(self) -> EachCompareResult:
        t, g, b = self.output, self.golden, self.third_party
        if b is None:
            return EachCompareResult(
                "GOLDEN_FAILURE",
                is_pass=False,
                standard="cross_check",
                metrics={
                    "standard": "cross_check",
                    "pass": False,
                    "level": self.tol_options.get("level"),
                    "reason": FAIL_REASONS["third_party_unavailable"],
                },
            )

        T = np.promote_types(np.float32, np.promote_types(t.dtype, np.promote_types(g.dtype, b.dtype)))
        t, g, b = t.astype(T), g.astype(T), b.astype(T)

        sv = {"small_value": self.tol_options["small_value"], "small_value_atol": self.tol_options["small_value_atol"]}
        limits = {
            "mare": self.tol_options["mare_ratio"],
            "mere": self.tol_options["mere_ratio"],
            "rmse": self.tol_options["rmse_ratio"],
        }

        special = np.isnan(t) | np.isinf(t) | np.isnan(g) | np.isinf(g) | np.isnan(b) | np.isinf(b)
        finite = ~special
        large = finite & (np.abs(g) >= sv["small_value"])
        small = finite & (np.abs(g) < sv["small_value"])

        special_ok = self._check_special(t, g, b, special)
        large_result = self._check_large(t, g, b, large, sv, limits)
        small_result = self._check_small(t, g, b, small, sv)

        passed = large_result.ratio_ok and special_ok and small_result.ok
        diff_idx = large_result.diff_idx + small_result.diff_idx

        reason = self._build_reason(passed, special_ok, small_result, large_result)
        metrics = self._build_metrics(limits, sv, large_result, small_result, passed, reason)

        _diff_arr = np.array(diff_idx, dtype=int) if diff_idx else None
        return EachCompareResult(
            "PASS" if passed else "FAIL",
            is_pass=passed,
            diff_index=_diff_arr if not passed else None,
            standard="cross_check",
            metrics=metrics,
        )

    def _check_special(self, t, g, b, special):
        """Path A: 特殊位（NaN/Inf）三方交叉校验。

        规则（满足任一即通过）：
          1. NPU 与 third_party 一致（不论 golden 为何值）→ 通过
          2. golden 为 nan/inf/-inf 且 NPU 与 golden 一致（不论 third_party）→ 通过
          以上都不满足 → 不通过
        """
        if not special.any():
            return True
        t_nan, g_nan, b_nan = np.isnan(t), np.isnan(g), np.isnan(b)
        t_pinf, g_pinf, b_pinf = np.isposinf(t), np.isposinf(g), np.isposinf(b)
        t_ninf, g_ninf, b_ninf = np.isneginf(t), np.isneginf(g), np.isneginf(b)
        t_fin, g_fin, b_fin = np.isfinite(t), np.isfinite(g), np.isfinite(b)

        # 规则1：NPU 与 third_party 一致（同为 nan/+inf/-inf，或同为有限且数值相等）
        tb_match = (t_nan & b_nan) | (t_pinf & b_pinf) | (t_ninf & b_ninf) | (t_fin & b_fin & (t == b))
        # 规则2：golden 非有限 且 NPU 与 golden 一致（同为 nan/+inf/-inf）
        g_special = ~g_fin
        tg_match = (t_nan & g_nan) | (t_pinf & g_pinf) | (t_ninf & g_ninf) | (t_fin & g_fin & (t == g))

        pass_pos = tb_match | (g_special & tg_match)
        fail_pos = special & ~pass_pos
        return not fail_pos.any()

    def _check_large(self, t, g, b, large, sv, limits):
        """Path B-1: 大值域 mare/mere/rmse 三比值判定。"""
        if not large.any():
            return _LargeResult(ratio_ok=True, diff_idx=[])

        with np.errstate(invalid="ignore", divide="ignore"):
            rel_npu = np.abs(t[large] - g[large]) / (np.abs(g[large]) + 1e-7)
            rel_party = np.abs(b[large] - g[large]) / (np.abs(g[large]) + 1e-7)
            # third_party 在 large 位置全 inf/nan 时（算子数学溢出，如
            # npu_quant_matmul 用负 scale），ratio 不参与判定，只看 target vs golden
            b_party_invalid = np.all(np.isinf(rel_party) | np.isnan(rel_party))
            if b_party_invalid:
                return _LargeResult(ratio_ok=True, diff_idx=self._top_diff(t, g, large))

            mare_ratio = safe_div(rel_npu.max(), rel_party.max(), sv["small_value"])
            mere_ratio = safe_div(rel_npu.mean(), rel_party.mean(), sv["small_value"])
            # float32 平方溢出检测：差值 > ~1.8e19 时 **2 溢出为 Inf → float64 重算
            rmse_npu = np.sqrt(np.mean((t[large] - g[large]) ** 2))
            rmse_party = np.sqrt(np.mean((b[large] - g[large]) ** 2))
            if np.isinf(rmse_npu) or np.isinf(rmse_party):
                rmse_npu = np.sqrt(np.mean(((t[large] - g[large]).astype(np.float64)) ** 2))
                rmse_party = np.sqrt(np.mean(((b[large] - g[large]).astype(np.float64)) ** 2))
            rmse_ratio = safe_div(rmse_npu, rmse_party, sv["small_value"])

        exceeded = []
        if mare_ratio > limits["mare"]:
            exceeded.append(f"mare({mare_ratio:.2f}>{limits['mare']})")
        if mere_ratio > limits["mere"]:
            exceeded.append(f"mere({mere_ratio:.2f}>{limits['mere']})")
        if rmse_ratio > limits["rmse"]:
            exceeded.append(f"rmse({rmse_ratio:.2f}>{limits['rmse']})")

        return _LargeResult(
            ratio_ok=not exceeded,
            mare=mare_ratio,
            mere=mere_ratio,
            rmse=rmse_ratio,
            exceeded=exceeded,
            diff_idx=self._top_diff(t, g, large),
        )

    def _check_small(self, t, g, b, small, sv):
        """Path B-2: 小值域 ErrorCount 比值判定。"""
        if not small.any():
            return _SmallResult(ok=True, err_target=0, err_third=0, ratio=0.0, diff_idx=[])
        err_target = int(np.sum(small & (np.abs(t - g) > sv["small_value_atol"])))
        err_third = int(np.sum(small & (np.abs(b - g) > sv["small_value_atol"])))
        small_ratio = err_target / max(err_third, 1)
        return _SmallResult(
            ok=small_ratio <= 2.0,
            err_target=err_target,
            err_third=err_third,
            ratio=small_ratio,
            diff_idx=self._top_diff(t, g, small),
        )

    @staticmethod
    def _build_reason(passed, special_ok, small_result, large_result):
        if passed:
            return None
        if not special_ok:
            return FAIL_REASONS["nan_inf_mismatch"]
        if not small_result.ok:
            return FAIL_REASONS["small_value_exceeded"].format(ratio=f"{small_result.ratio:.2f}>2.0")
        return FAIL_REASONS["ratio_exceeded"].format(exceeded=", ".join(large_result.exceeded))

    def _build_metrics(self, limits, sv, large_result, small_result, passed, reason):
        metrics = {
            "standard": "cross_check",
            "level": self.tol_options.get("level"),
            "config": {
                "mare": limits["mare"],
                "mere": limits["mere"],
                "rmse": limits["rmse"],
                "small_value": sv["small_value"],
                "small_value_atol": sv["small_value_atol"],
            },
            "result": {
                "mare": float(large_result.mare) if large_result.mare is not None else None,
                "mere": float(large_result.mere) if large_result.mere is not None else None,
                "rmse": float(large_result.rmse) if large_result.rmse is not None else None,
                "small_err_cnt_target": small_result.err_target,
                "small_err_cnt_third": small_result.err_third,
                "small_err_ratio": float(small_result.ratio),
            },
            "pass": passed,
        }
        if reason:
            metrics["reason"] = reason
        return metrics

    def _top_diff(self, t, g, mask):
        """top MAX_DIFF_OUTPUT |t-g|>0 positions (worst-first) under mask. Returns list."""
        _diff = np.abs(t[mask] - g[mask])
        _li = np.where(mask)[0]
        _mis = _diff > 0
        if not _mis.any():
            return []
        _md, _ml = _diff[_mis], _li[_mis]
        _k = min(self.MAX_DIFF_OUTPUT, _md.size)
        _top = np.argpartition(-_md, _k - 1)[:_k]
        result = _ml[_top[np.argsort(-_md[_top])]].tolist()
        del _md, _ml, _top, _diff, _li, _mis
        return result

    def _log_diff_output(self, diff_index):
        """Three-party diff (t/g/b + |t-g| + |b-g|), worst-first.
        large positions use %.6e, small positions use %.10e (small values need
        more digits). [L]/[S] tag distinguishes large/small in each row."""
        log = ""
        if diff_index is None or len(diff_index) == 0:
            return log
        t, g, b = self.output, self.golden, self.third_party
        small_value = self.tol_options.get("small_value", 1e-7)
        n = len(diff_index)
        log += f"Output {self.output_idx} Compare Difference length {n} (worst-first)\n"
        for idx in range(n):
            i = int(diff_index[idx])
            is_small = abs(float(g[i])) < small_value
            fmt = "%.10e" if is_small else "%.6e"
            tag = "S" if is_small else "L"
            log += (
                f"Index: {idx:03d} RealIndex: {i:06d} [{tag}] t={fmt % float(t[i])} g={fmt % float(g[i])} "
                f"b={fmt % float(b[i])} |t-g|={fmt % float(abs(t[i] - g[i]))} |b-g|={fmt % float(abs(b[i] - g[i]))}\n"
            )
        return log


class _LargeResult:
    __slots__ = ("ratio_ok", "mare", "mere", "rmse", "exceeded", "diff_idx")

    def __init__(self, ratio_ok, mare=None, mere=None, rmse=None, exceeded=None, diff_idx=None):
        self.ratio_ok = ratio_ok
        self.mare = mare
        self.mere = mere
        self.rmse = rmse
        self.exceeded = exceeded or []
        self.diff_idx = diff_idx or []


class _SmallResult:
    __slots__ = ("ok", "err_target", "err_third", "ratio", "diff_idx")

    def __init__(self, ok, err_target, err_third, ratio, diff_idx):
        self.ok = ok
        self.err_target = err_target
        self.err_third = err_third
        self.ratio = ratio
        self.diff_idx = diff_idx or []
