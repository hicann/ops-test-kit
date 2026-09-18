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
binary comparison
"""

# Standard Packages
import hashlib

import numpy as np

from ...utilities.dtypes import is_4bit_dtype

# Third-party Packages
from .registry import FAIL_REASONS, ComparisonBase, EachCompareResult, register_comparison


@register_comparison(["bin", "binary", "binary_equal"])
class BinaryComparison(ComparisonBase):
    STANDARD_NAME = "binary_equal"

    def compare_impl(self) -> EachCompareResult:
        # 空数组已在 compare() 入口 _check_empty 统一短路，这里只处理非空
        if str(self.output.dtype) != str(self.golden.dtype):
            return self._cross_dtype_compare()
        diff_idx, golden_size, diff_idx_size = self._numpy_binary_compare(self.output, self.golden)
        return self._result(diff_idx, golden_size, diff_idx_size)

    def _result(self, diff_idx, golden_size, diff_idx_size):
        if diff_idx_size == 0:
            return EachCompareResult(
                1, is_pass=True, standard="binary_equal", metrics={"standard": "binary_equal", "pass": True}
            )
        precision = (golden_size - diff_idx_size) / golden_size
        return EachCompareResult(
            precision,
            diff_idx,
            is_pass=False,
            standard="binary_equal",
            metrics={"standard": "binary_equal", "pass": False, "reason": FAIL_REASONS["bitwise_mismatch"]},
        )

    def _cross_dtype_compare(self) -> EachCompareResult:
        od, gd = self.output.dtype, self.golden.dtype
        # int4/float4 自定义 dtype 会让 np.issubdtype/promote_types 崩，短路拒
        if is_4bit_dtype(od) or is_4bit_dtype(gd):
            return self._reject(od, gd)
        out_int = np.issubdtype(od, np.integer) or od.kind == "b"
        gold_int = np.issubdtype(gd, np.integer) or gd.kind == "b"
        if out_int and gold_int:
            promoted = np.promote_types(od, gd)
            if np.issubdtype(promoted, np.integer):
                diff_idx, golden_size, diff_idx_size = self._numpy_binary_compare(
                    self.output.astype(promoted), self.golden.astype(promoted)
                )
                return self._result(diff_idx, golden_size, diff_idx_size)
        return self._reject(od, gd)

    def _reject(self, od, gd) -> EachCompareResult:
        return EachCompareResult(
            0,
            is_pass=False,
            standard="binary_equal",
            log=f"Dtype not bitwise-comparable: {od} vs {gd}",
            metrics={"standard": "binary_equal", "pass": False, "reason": FAIL_REASONS["cross_dtype_uncomparable"]},
        )

    # 分块粒度(字节)。只影响峰值内存与短路时机, 不影响结果。
    BYTES_CHUNK = 4_000_000

    @staticmethod
    def _bytes_equal(output: np.ndarray, golden: np.ndarray) -> bool:
        """逐字节相等判定(与 SHA256 字节哈希同语义, 但走 SIMD 且首个不等块即短路)。"""
        if output.nbytes != golden.nbytes:
            return False
        ou = output.view(np.uint8).reshape(-1)
        gu = golden.view(np.uint8).reshape(-1)
        step = BinaryComparison.BYTES_CHUNK
        for start in range(0, ou.size, step):
            end = min(start + step, ou.size)
            if not np.array_equal(ou[start:end], gu[start:end]):
                return False
        return True

    @staticmethod
    def _numpy_binary_compare(output: np.ndarray, golden: np.ndarray):
        if not output.flags["C_CONTIGUOUS"]:
            output = np.ascontiguousarray(output)
        if not golden.flags["C_CONTIGUOUS"]:
            golden = np.ascontiguousarray(golden)
        if "float8_e8m0" in str(output.dtype):
            output = output.view(np.uint8)
            golden = golden.view(np.uint8)
        if "float4" in str(output.dtype):
            output = output.view(np.int8)
            golden = golden.view(np.int8)
        # 判"两块内存是否逐字节相同"不需要密码学哈希: uint8 视图 + 分块 array_equal 走 SIMD,
        # 且首个不等的块即短路。实测 2GiB 一对: SHA256 x2 = 1.97s, 本法 = 0.09s(22x)。
        # 语义与字节哈希完全一致(都是逐字节), 浮点 ±0 / NaN 的字节级差异同样判不等 ——
        # 换成裸 == 才会踩 [[binary-equal-float-zero-sign-trap]] 那个坑。
        # view(uint8) 对自定义窄 dtype(bfloat16/int4/fp8 等)不成立, 那几类仍走 tobytes 哈希。
        if output.dtype.name not in ("bfloat16", "int4", "float8_e5m2", "float8_e4m3fn", "hifloat8"):
            if BinaryComparison._bytes_equal(output, golden):
                return None, golden.size, 0
        else:
            hash_output = hashlib.sha256(output.tobytes()).hexdigest()
            hash_golden = hashlib.sha256(golden.tobytes()).hexdigest()
            if hash_output == hash_golden:
                return None, golden.size, 0

        output_int = output.view(dtype=np.uint8)
        golden_int = golden.view(dtype=np.uint8)
        same = output_int == golden_int
        diff_idx = np.floor_divide(np.where(~same)[0], output.dtype.itemsize)
        diff_idx = np.unique(diff_idx)

        npu_nan, golden_nan = np.isnan(output), np.isnan(golden)
        diff_nan = np.logical_and(npu_nan, golden_nan)
        both_nan_idx = np.where(diff_nan)

        diff_idx = np.setdiff1d(diff_idx, both_nan_idx)
        del same, npu_nan, golden_nan, diff_nan

        golden_size, diff_idx_size = golden.size, diff_idx.size
        return diff_idx, golden_size, diff_idx_size
