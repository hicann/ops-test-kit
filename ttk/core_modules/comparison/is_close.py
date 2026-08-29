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
is_close comparison
"""

# Standard Packages
import numpy as np

# Third-party Packages
from ...utilities import get
from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


@register_comparison(["close", "isclose"])
class CloseComparison(ComparisonBase):
    STANDARD_NAME = "isclose"

    def __post_init__(self):
        legacy = self.tol_options.get("legacy", {})
        self.rtol = legacy.get("rtol", None)
        if not isinstance(self.rtol, (tuple, list)):
            self.rtol = [self.rtol]
        self.ptol = legacy.get("ptol", None)
        if not isinstance(self.ptol, (tuple, list)):
            self.ptol = [self.ptol]
        self.atol = legacy.get("atol", None)
        if not isinstance(self.atol, (tuple, list)):
            self.atol = [self.atol]

    @staticmethod
    def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
        if hasattr(arr, "dtype") and hasattr(arr.dtype, "name"):
            if arr.dtype.name == "bfloat16":
                return arr.astype("float32", copy=False)
            elif arr.dtype.name == "int4":
                return arr.astype("int8", copy=False)
        return arr

    def _numpy_isclose(self, output: np.ndarray, golden: np.ndarray, rtol: float, atol: float, golden_size: int):
        if rtol == 0 and atol == 0:
            normal_equal = output == golden
            both_nan = np.isnan(output) & np.isnan(golden)
            diff_results = normal_equal | both_nan
        else:
            diff_results = np.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
        diff_indices = np.where(~diff_results)[0]
        del diff_results
        precision = (golden_size - diff_indices.size) / golden_size
        return precision, diff_indices

    def compare_impl(self) -> EachCompareResult:
        ptol = self._get_ptol(self.output.dtype)
        compare_result = self._isclose()
        if isinstance(compare_result.precision, str):
            compare_result.is_pass = False
        else:
            if (1 - compare_result.precision) > ptol:
                compare_result.is_pass = False
            else:
                compare_result.is_pass = True
        compare_result.standard = "isclose"
        metrics = {
            "standard": "isclose",
            "precision": (
                f"{compare_result.precision * 100}%"
                if not isinstance(compare_result.precision, str)
                else compare_result.precision
            ),
            "pass": bool(compare_result.is_pass),
        }
        if not compare_result.is_pass:
            metrics["reason"] = FAIL_REASONS["tolerance_exceeded"]
        compare_result.metrics = metrics
        return compare_result

    def _isclose(self) -> EachCompareResult:
        # 空数组已在 base compare() 的 _check_empty 统一短路，这里只处理非空
        rtol = self._get_rtol(self.output.dtype)
        atol = self._get_atol(self.output.dtype)
        diff_indices, log = None, ""
        output_size = self.output.size
        golden_size = self.golden.size
        if output_size != golden_size:
            precision = f"{output_size} vs {golden_size}"
            log += f"Output {self.output_idx} size is different with golden size: {precision}\n"
        else:
            output = self._normalize_dtype(self.output)
            golden = self._normalize_dtype(self.golden)
            precision, diff_indices = self._numpy_isclose(output, golden, rtol, atol, golden_size)
        return EachCompareResult(precision, diff_indices, False, log)

    def _get_rtol(self, dtype):
        rtol = get(self.rtol, self.output_idx)
        if rtol is None:
            dtype_str = str(dtype).split(".")[-1]
            rtol = 0.001 if dtype_str in ("float16", "bfloat16", "complex32") else 0.0001
        return rtol

    def _get_atol(self, dtype):
        atol = get(self.atol, self.output_idx)
        if atol is None:
            atol = 1e-8
        return atol

    def _get_ptol(self, dtype):
        ptol = get(self.ptol, self.output_idx)
        if ptol is None:
            dtype_str = str(dtype).split(".")[-1]
            ptol = 0.001 if dtype_str in ("float16", "bfloat16", "complex32") else 0.0001
        return ptol
