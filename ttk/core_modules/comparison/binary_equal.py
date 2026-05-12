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
import math

import numpy as np

# Third-party Packages
from .registry import ComparisonBase, EachCompareResult, register_comparison


@register_comparison(['bin', 'binary'])
class BinaryComparison(ComparisonBase):
    def compare_impl(self) -> EachCompareResult:
        if str(self.output.dtype) != str(self.golden.dtype):
            return EachCompareResult(0, is_pass=False,
                                     log=f"Dtype for output & golden is different: "
                                         f"{self.output.dtype} vs {self.golden.dtype}")
        if self.is_torch:
            diff_idx, golden_size, diff_idx_size = self._torch_binary_compare(self.output, self.golden)
        else:
            diff_idx, golden_size, diff_idx_size = self._numpy_binary_compare(self.output, self.golden)

        if diff_idx_size == 0:
            return EachCompareResult(1, is_pass=True)
        else:
            precision = (golden_size - diff_idx_size) / golden_size
            return EachCompareResult(precision, diff_idx, is_pass=False)

    @staticmethod
    def _numpy_binary_compare(output: np.ndarray, golden: np.ndarray):
        if not golden.flags['C_CONTIGUOUS']:
            golden = np.ascontiguousarray(golden)
        if 'float8_e8m0' in str(output.dtype):
            output = output.view(np.uint8)
            golden = golden.view(np.uint8)
        if 'float4' in str(output.dtype):
            output = output.view(np.int8)
            golden = golden.view(np.int8)
        if output.dtype.name not in ('bfloat16', 'int4', 'float8_e5m2', 'float8_e4m3fn', 'hifloat8'):
            hash_output = hashlib.sha256(output.data).hexdigest()
            hash_golden = hashlib.sha256(golden.data).hexdigest()
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

    @staticmethod
    def _torch_binary_compare(output: "torch.Tensor", golden: "torch.Tensor"):
        import torch
        dtypes = tuple([torch.int8, torch.int16, torch.int32, torch.int64])
        normalized_dtype = dtypes[int(math.log2(output.element_size()))]
        output_int = output.view([-1]).view(dtype=normalized_dtype)
        golden_int = golden.view([-1]).view(dtype=normalized_dtype)
        same = output_int == golden_int
        diff_idx = torch.floor_divide(torch.where(~same)[0], output.element_size())
        diff_idx = torch.unique(diff_idx)

        npu_nan, golden_nan = torch.isnan(output.view([-1])), torch.isnan(golden.view([-1]))
        diff_nan = torch.logical_and(npu_nan, golden_nan)
        both_nan_idx = torch.where(diff_nan)

        diff_idx = torch.where(torch.logical_not(torch.isin(diff_idx, both_nan_idx[0])))[0]
        del same, npu_nan, golden_nan, diff_nan

        golden_size, diff_idx_size = golden.numel(), diff_idx.numel()
        return diff_idx, golden_size, diff_idx_size
