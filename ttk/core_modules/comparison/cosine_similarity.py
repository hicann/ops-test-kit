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
import numpy as np

# Third-party Packages
from ...utilities import get
from .registry import ComparisonBase, EachCompareResult, register_comparison, FAIL_REASONS


@register_comparison("cosine")
class CosineSimilarityComparison(ComparisonBase):
    STANDARD_NAME = "cosine"

    def __post_init__(self):
        legacy = self.tol_options.get("legacy", {})
        self.rtol = legacy.get("rtol", None)
        if not isinstance(self.rtol, (tuple, list)):
            self.rtol = [self.rtol]

    @staticmethod
    def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
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
            # np.dot accumulates in the operand dtype; fp16/bf16 overflow to inf
            # for large vectors (e.g. 1M elements * magnitude ~1 -> sum ~1e6 >
            # fp16 max 65504). Promote to float32 for the dot product and norms
            # so the cosine similarity is computed in fp32 precision regardless
            # of the input dtype.
            compute_dtype = np.float32 if common_dtype in (np.float16,) or \
                (hasattr(common_dtype, 'name') and common_dtype.name in ('float16', 'bfloat16')) \
                else common_dtype
            _output = _output.astype(compute_dtype, copy=False)
            _golden = _golden.astype(compute_dtype, copy=False)
            output_norm = np.linalg.norm(_output)
            golden_norm = np.linalg.norm(_golden)
            precision = np.dot(_output, _golden.T) / (output_norm * golden_norm)
        is_pass = (1 - precision) <= rtol
        metrics = {"standard": "cosine", "precision": float(precision),
                   "pass": bool(is_pass)}
        if not is_pass:
            metrics["reason"] = FAIL_REASONS["similarity_below_threshold"]
        return EachCompareResult(precision, is_pass=is_pass, standard="cosine",
                                 metrics=metrics)

    def _get_rtol(self, dtype):
        rtol = get(self.rtol, self.output_idx)
        if rtol is None:
            rtol = 0.01
        return rtol
