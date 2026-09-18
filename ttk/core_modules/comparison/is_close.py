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
            if arr.dtype.name == "int4":
                return arr.astype("int8", copy=False)
        return arr

    # 大数组分块比较的分块粒度（元素数）。与 utilities/data.py 的 CHUNK_ELEMS 同数量级，
    # 取值只影响峰值内存与调用开销，不影响结果。
    CHUNK_ELEMS = 4_000_000

    @staticmethod
    def _close_mask(out_chunk: np.ndarray, gold_chunk: np.ndarray, rtol: float, atol: float) -> np.ndarray:
        """与 np.isclose(..., equal_nan=True) 逐元素等价，但把窄浮点一次性抬到 float32 再算。

        np.isclose 内部要做 isfinite(x)/isfinite(y)/abs(x-y)/rtol*abs(y)/加法/比较 等约 8 趟运算，
        **每一趟都会独立地把 fp16 转一次 float32**——numpy 对 fp16 没有原生 SIMD 循环，转换是
        逐元素标量完成的。实测同规模 fp16 比 fp32 慢 13 倍（1 亿元素 2.08s vs 0.16s），10.7 亿
        元素一次 isclose 要 110s。这里每块只转一次，后续算术全在 float32 上走 SIMD。
        """
        a = out_chunk.astype(np.float32, copy=False) if out_chunk.dtype == np.float16 else out_chunk
        b = gold_chunk.astype(np.float32, copy=False) if gold_chunk.dtype == np.float16 else gold_chunk
        finite = np.isfinite(a) & np.isfinite(b)
        # 有限位按容差判；非有限位按 np.isclose 的语义走 ==（同号 inf 相等），
        # 再叠加 equal_nan=True 的 both-nan 视为相等。
        within = np.abs(a - b) <= atol + rtol * np.abs(b)
        nonfinite_eq = (a == b) | (np.isnan(a) & np.isnan(b))
        return np.where(finite, within, nonfinite_eq)

    def _numpy_isclose(self, output: np.ndarray, golden: np.ndarray, rtol: float, atol: float, golden_size: int):
        if rtol == 0 and atol == 0:
            normal_equal = output == golden
            both_nan = np.isnan(output) & np.isnan(golden)
            diff_results = normal_equal | both_nan
            diff_indices = np.where(~diff_results)[0]
            del diff_results
        elif output.size <= 2 * self.CHUNK_ELEMS or output.dtype != np.float16:
            # 小数组、以及本就有 SIMD 循环的 dtype：保持单次全量 np.isclose，行为逐位不变。
            diff_results = np.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
            diff_indices = np.where(~diff_results)[0]
            del diff_results
        else:
            # 大 fp16 数组：分块 + 块内一次性抬 float32。峰值内存由分块粒度决定，
            # 不再出现 np.isclose 在 2GiB 输入上开 ~6GiB 临时量的尖峰。
            parts = []
            for start in range(0, output.size, self.CHUNK_ELEMS):
                end = min(start + self.CHUNK_ELEMS, output.size)
                bad = np.where(~self._close_mask(output[start:end], golden[start:end], rtol, atol))[0]
                if bad.size:
                    parts.append(bad + start)
            diff_indices = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        precision = (golden_size - diff_indices.size) / golden_size
        return precision, diff_indices

    def compare_impl(self) -> EachCompareResult:
        ptol = self._get_ptol(self.output.dtype)
        compare_result = self._isclose()
        if isinstance(compare_result.precision, str) or (1 - compare_result.precision) > ptol:
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
