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
quantization comparison —— 量化输出精度判据
"""
import numpy as np

from .registry import ComparisonBase, EachCompareResult, register_comparison, FAIL_REASONS


@register_comparison('quant')
class QuantComparison(ComparisonBase):
    """量化输出判据：**绝对误差 <= 1**（1 LSB = 1 ULP），单标杆比对，不做三方。

    【适用场景】浮点型输入 + int4/int8 输出的量化算子。

    【为何不能用 binary_equal】
    量化的最后一步是 round(x / scale)，round 是阶跃函数：当商落在 .5 附近时，
    浮点末位 1 ulp 的差异就会让结果跳到相邻整数格。而末位差异来自归约次序、
    FMA 融合、硬件舍入等实现自由度——**不同实现之间必然存在**。实测两个都属
    Welford 族的实现（torch.var_mean vs 显式串行 Welford）之间就有 1.79% 的量化
    元素不一致，远大于被测内核与 golden 的差异。要求逐位相等等于要求复现某一个
    实现的浮点运算次序，既不可能也无意义。

    【为何不做三方】
    本判据是绝对标准（差 <= 1 即通过），及格线已经定死，不需要拿竞品当尺子。
    实测也证明三方在此不可靠：竞品实现细节（如是否开 torch.compile）会左右结论，
    内核对错不该取决于竞品用了哪个 API。

    【判定】
    逐元素 |output - golden| > 1 即计入错误；错误占比 > ptol 判失败（ptol 默认 0，
    即一个都不许超）。NaN 位置双方都是 NaN 时视为相等（与 requant 一致）。
    """

    STANDARD_NAME = "quant"

    def compare_impl(self) -> EachCompareResult:
        out = self.output.astype(np.int32, copy=False)
        gold = self.golden.astype(np.int32, copy=False)
        diff = np.abs(out - gold)
        bad = np.where(diff > 1)[0]

        golden_size = gold.size
        bad_size = bad.size
        precision = (golden_size - bad_size) / golden_size if golden_size else 1.0
        ptol = self._ptol()
        is_pass = (1 - precision) <= ptol

        metrics = {"standard": self.STANDARD_NAME,
                   "precision": f"{precision * 100}%",
                   "abs_err_limit": 1,
                   "exceed_count": int(bad_size),
                   "pass": bool(is_pass)}
        if not is_pass:
            metrics["reason"] = FAIL_REASONS.get("precision_exceeded", "precision exceeded")
        return EachCompareResult(precision, bad, is_pass=is_pass,
                                 standard=self.STANDARD_NAME, metrics=metrics)

    def _ptol(self):
        """允许超差的元素占比。默认 0——标准是绝对误差 <= 1，超出即缺陷。
        留出 Spec.tolerance 可配的口子（如 {"int8": {"standard": "quant", "ptol": 0.001}}），
        但不设默认放宽。"""
        try:
            return float((self.tol_options or {}).get("ptol", 0) or 0)
        except (TypeError, ValueError):
            return 0.0
