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
cosine similarity comparison
"""

# Standard Packages
from typing import Union
import numpy as np

# Third-party Packages
from ...utilities import get
from .registry import ComparisonBase, EachCompareResult, register_comparison


@register_comparison("cosine")
class CosineSimilarityComparison(ComparisonBase):
    def __post_init__(self):
        self.rtol = self.tol_options.get('rtol', None)
        if not isinstance(self.rtol, (tuple, list)):
            self.rtol = [self.rtol]

    @staticmethod
    def _normalize_dtype(arr: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        # convert int4 -> int8, bcz np.dot will be wrong with int4
        if hasattr(arr, "dtype") and hasattr(arr.dtype, "name"):
            if arr.dtype.name == 'int4':
                return arr.astype("int8", copy=False)
        return arr

    def compare_impl(self) -> EachCompareResult:
        rtol = self._get_rtol(self.output.dtype)
        output = self._normalize_dtype(self.output)
        golden = self._normalize_dtype(self.golden)
        if self.is_torch:
            from torch import cosine_similarity
            precision = cosine_similarity(output.view([-1]), golden.view([-1]), dim=0)
        else:
            common_dtype = np.promote_types(output.dtype, golden.dtype)
            _output = output if common_dtype == output.dtype else output.astype(common_dtype)
            _golden = golden if common_dtype == golden.dtype else golden.astype(common_dtype)
            output_norm = np.linalg.norm(_output)
            golden_norm = np.linalg.norm(_golden)
            precision = np.dot(_output, _golden.T) / (output_norm * golden_norm)
        return EachCompareResult(precision, is_pass=False if (1 - precision) > rtol else True)

    def _get_rtol(self, dtype):
        rtol = get(self.rtol, self.output_idx)
        if rtol is None:
            rtol = 0.01
        return rtol
