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
re-quantize comparison
"""

# Standard Packages
import numpy as np

# Third-party Packages
from .registry import ComparisonBase, EachCompareResult, register_comparison


@register_comparison('requant')
class ReQuantizeComparison(ComparisonBase):
    def compare_impl(self) -> EachCompareResult:
        dtype_str = str(self.output.dtype).split('.')[-1]
        ptol = self._get_ptol(dtype_str)
        output = self.output
        golden = self.golden
        if self.is_torch:
            import torch
            if dtype_str in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8'):
                # compare ULP for FP8, so view as int8.
                output = self.output.view(torch.int8)
                golden = self.golden.view(torch.int8)
            diff_results = torch.abs(torch.subtract(output.view(-1), golden.view(-1)))
            diff_indices = torch.where(diff_results > 1)[0]

            npu_nan, golden_nan = torch.isnan(self.output.view([-1])), torch.isnan(self.golden.view([-1]))
            diff_nan = torch.logical_and(npu_nan, golden_nan)
            both_nan_idx = torch.where(diff_nan)

            diff_indices = torch.where(torch.logical_not(torch.isin(diff_indices, both_nan_idx[0])))[0]
            del diff_results, npu_nan, golden_nan, diff_nan

            golden_size, diff_size = golden.numel(), diff_indices.numel()
        else:
            if dtype_str in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8'):
                # compare ULP for FP8, so view as int8.
                output = output.view(np.int8)
                golden = golden.view(np.int8)
            diff_results = np.abs(np.subtract(output, golden))
            diff_indices = np.where(diff_results > 1)[0]

            npu_nan, golden_nan = np.isnan(self.output), np.isnan(self.golden)
            diff_nan = np.logical_and(npu_nan, golden_nan)
            both_nan_idx = np.where(diff_nan)
            diff_indices = np.setdiff1d(diff_indices, both_nan_idx)
            del diff_results, npu_nan, golden_nan, diff_nan

            golden_size, diff_size = golden.size, diff_indices.size
        precision = (golden_size - diff_size) / golden_size
        return EachCompareResult(precision, diff_indices, is_pass=(1 - precision) <= ptol)

    @staticmethod
    def _get_ptol(dtype: str):
        return 0.001 if dtype in ('float8_e5m2', 'float8_e4m3fn', 'hifloat8') else 0
