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
stat_rel_err comparison — 相对误差统计精度标准（MERE/MARE）
"""

import numpy as np

from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison

# 算法常量（不是配置——配置由 resolve_tolerance 解析后经 params 传入）
MARE_RATIO = 10
FLOOR = 1e-7
# 分块大小：所有中间数组均为 O(CHUNK_SIZE)，不随输出元素数增长（issue #166）。
# diff_index 仅被 _log_diff_output 消费（打印前 MAX_DIFF_OUTPUT+1 行），故只保留 top-K
# (worst-first)；旧实现的全量 int64 索引 + argsort 约占 52 字节/元素，会把 worker 推向 OOM。
CHUNK_SIZE = 1_048_576


@register_comparison("stat_rel_err")
class StatRelErrComparison(ComparisonBase):
    STANDARD_NAME = "stat_rel_err"

    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        # threshold 是 resolve_tolerance 解析好的最终值（经 params → options 传入），不查表
        actual, golden = self.output, self.golden
        T = np.promote_types(np.dtype(np.float32), np.promote_types(actual.dtype, golden.dtype))
        th = self.tol_options["threshold"]
        k = self.MAX_DIFF_OUTPUT + 1  # _log_diff_output 实际打印的行数（idx 0..100）
        n = actual.size
        mism = []  # 前 K 个 NaN/Inf 不一致位（全局升序），够 K 个即停
        cand_err, cand_idx = [], []  # 各块局部 top-K 候选，循环后归并取全局 top-K
        n_finite = 0
        err_sum = 0.0  # float64 累加各块有限元素 rel_err 之和
        mare = 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            for s in range(0, n, CHUNK_SIZE):
                # 切片上标自动截断到 n，最后一块可小于 CHUNK_SIZE；dtype 提升在块内做免全量拷贝
                ac = actual[s : s + CHUNK_SIZE].astype(T, copy=False)
                gc = golden[s : s + CHUNK_SIZE].astype(T, copy=False)
                a_nan, g_nan = np.isnan(ac), np.isnan(gc)
                a_inf, g_inf = np.isinf(ac), np.isinf(gc)
                same_nan = a_nan & g_nan  # 都 NaN（NaN 无符号）
                same_inf = a_inf & g_inf & (np.sign(ac) == np.sign(gc))  # 都 Inf 且同号
                mismatch = (a_nan | g_nan | a_inf | g_inf) & ~same_nan & ~same_inf
                if mismatch.any():
                    need = k - len(mism)
                    mism.extend((np.flatnonzero(mismatch)[:need] + s).tolist())
                    if len(mism) >= k:
                        break
                    continue
                if mism:  # 结论已定（mismatch FAIL），后续块只为补齐前 K 个位置
                    continue
                fin = np.isfinite(ac) & np.isfinite(gc)
                nf = int(fin.sum())
                if not nf:
                    continue
                n_finite += nf
                re = np.abs(ac - gc) / (np.abs(gc) + FLOOR)
                re_f = re[fin]
                err_sum += float(re_f.sum(dtype=np.float64))
                mare = max(mare, float(re_f.max()))
                idx_f = np.flatnonzero(fin) + s
                if re_f.size > k:  # 块内只留 top-K：全局 top-K 必在诸块 top-K 的并集内
                    part = np.argpartition(re_f, re_f.size - k)[re_f.size - k :]
                    re_f, idx_f = re_f[part], idx_f[part]
                cand_err.append(re_f)
                cand_idx.append(idx_f)

        if mism:  # NaN/Inf 不一致：污染，不算 mere/mare
            return EachCompareResult(
                "FAIL",
                diff_index=np.asarray(mism, dtype=np.int64),
                is_pass=False,
                standard="stat_rel_err",
                metrics=_metrics(th, None, None, False, FAIL_REASONS["nan_inf_mismatch"]),
            )
        if n_finite == 0:  # 全非有限且一致
            return EachCompareResult(
                "PASS", is_pass=True, standard="stat_rel_err", metrics=_metrics(th, None, None, True)
            )

        mere = err_sum / n_finite
        passed = mere < th and mare < MARE_RATIO * th
        if passed:
            diff_idx = None
            reason = None
        else:  # 数值 FAIL：top-K 按 rel_err 降序(worst first)
            err = np.concatenate(cand_err)
            idx = np.concatenate(cand_idx)
            diff_idx = idx[np.argsort(-err)[:k]]
            exceeded = []
            if mere >= th:
                exceeded.append(f"mere({mere:.2e}>={th})")
            if mare >= MARE_RATIO * th:
                exceeded.append(f"mare({mare:.2e}>={MARE_RATIO * th})")
            reason = FAIL_REASONS["threshold_exceeded"].format(metrics=", ".join(exceeded))

        return EachCompareResult(
            "PASS" if passed else "FAIL",
            diff_index=diff_idx,
            is_pass=passed,
            standard="stat_rel_err",
            metrics=_metrics(th, mere, mare, passed, reason),
        )


def _metrics(th, mere, mare, passed, reason=None):
    # 全 Python 字面量，保 CSV eval 往返；mere/mare=None（非 NaN）当 mismatch 或全 match
    m = {"standard": "stat_rel_err", "mere": mere, "mare": mare, "threshold": th, "pass": bool(passed)}
    if reason:
        m["reason"] = reason
    return m
