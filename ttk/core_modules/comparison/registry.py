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
comparison method register
"""

# Standard Packages
import gc
import logging
import numpy as np
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union


@dataclass
class EachCompareResult:
    precision: Union[str, float, int]
    diff_index: Optional[np.ndarray] = None
    is_pass: bool = False
    log: str = ""


class ComparisonBase(metaclass=ABCMeta):
    def __init__(self,
                 output: Union[np.ndarray, np.ndarray],
                 golden: Union[np.ndarray, np.ndarray],
                 output_idx: int,
                 output_dtype: str,
                 options: dict = None):
        if not options:
            options = {}
        self.tol_options = options
        self.output = output.reshape([-1])
        self.golden = golden.reshape([-1])
        self._pad_int4_if_need()
        self.output_idx = output_idx
        self.output_dtype = output_dtype
        self.__post_init__()
        self.is_torch: bool = self._is_torch(self.output)
        self.is_torch_output: bool = self._is_torch(self.output)
        self.is_torch_golden: bool = self._is_torch(self.golden)

    def __post_init__(self):
        pass

    @staticmethod
    def _is_torch(obj):
        return hasattr(obj, "numel")

    @abstractmethod
    def compare_impl(self) -> EachCompareResult:
        pass

    def compare(self):
        compare_result = self.compare_impl()
        if isinstance(compare_result.precision, Sequence):
            precision = compare_result.precision
        else:
            precision = f"{compare_result.precision * 100}%"
        compare_result.log += "Output %d Precision is %s\n" % (self.output_idx, precision)
        compare_result.log += self._log_diff_output(compare_result.diff_index)
        gc.collect()
        return precision, compare_result.log, compare_result.is_pass

    def _pad_int4_if_need(self):
        # only numpy dtype has `name` attribute. Op api output will always be torch.Tensor.
        if (hasattr(self.output.dtype, "name") and
                ('int4' in str(self.output.dtype) or 'float4' in str(self.output.dtype))):
            # output will always be 2*n
            if self.output.size >= 2 and self.golden.size == self.output.size - 1:
                self.golden = np.pad(self.golden, (0, 1), mode='constant')

    def _log_diff_output(self, diff_index: Union[np.ndarray, "torch.Tensor"]) -> str:
        log = ""
        if diff_index is not None:
            diff_index_is_torch = hasattr(diff_index, "numel")
            golden_size = self.golden.numel() if self.is_torch_golden else self.golden.size
            diff_index_size = diff_index.numel() if diff_index_is_torch else diff_index.size
            log += "Output %d Compare Difference length %d\n" % (self.output_idx, diff_index_size)
            for idx in range(diff_index_size):
                real_index = diff_index[idx]
                golden_v = self.golden[real_index] if golden_size > 0 else 0
                actual_v = self.output[real_index]
                if golden_size <= 0:
                    log += "Index: %03d RealIndex: %06d Expected: 0 Actual: %-14.18f, Diff: inf\n" \
                            % (idx, real_index, actual_v)
                elif any([s in str(self.golden.dtype) for s in ('float8', 'float4', 'hifloat8')]):
                    # out-of-range number will be nan/inf in
                    # `- (golden_v - actual_v) / golden_v`. convert to float.
                    tmp_array = np.array([float(golden_v), float(actual_v)], dtype=np.float32)
                    log += "Index: %03d RealIndex: %06d Expected: %-14.18f Actual: %-14.18f Diff: %-14.06f\n" \
                           % (idx, real_index, golden_v, actual_v, - (tmp_array[0] - tmp_array[1]) / tmp_array[0])
                elif "bool" not in str(self.golden.dtype):
                    # (golden_v - actual_v) is different with (actual_v - golden_v) in BFP16 vs FP32-Golden
                    log += "Index: %03d RealIndex: %06d Expected: %-14.18f Actual: %-14.18f Diff: %-14.06f\n" \
                            % (idx, real_index, golden_v, actual_v, - (golden_v - actual_v) / golden_v)
                else:
                    log += "Index: %03d RealIndex: %06d Expected: %s Actual: %s Diff: FAIL\n" \
                            % (idx, real_index, str(golden_v), str(actual_v))
                if idx == 100:
                    break
        return log


class ComparisonRegister(type):
    registry: Dict[str, ComparisonBase] = {}

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_class
        return new_class


def register_comparison(methods: Sequence):
    if not isinstance(methods, (list, tuple, str)):
        raise TypeError(f"Register class for comparison must receive a list / tuple / str, not {methods}")
    if isinstance(methods, str):
        methods = [methods]

    def decorator(cls: ComparisonBase):
        if isinstance(cls, ComparisonBase):
            raise TypeError(f"Implement of comparison {methods} is not a subclass of ComparisonBase !!")
        for method in methods:
            if method in ComparisonRegister.registry:
                logging.warning(f"comparison class of [{method}] has already been registered!")
            ComparisonRegister.registry[method.lower()] = cls
        return cls

    return decorator
