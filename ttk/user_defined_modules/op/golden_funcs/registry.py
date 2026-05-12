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
Golden data generator functions
"""

# Standard Packages
import logging
from typing import Sequence
from functools import wraps

# Third-party Packages
import numpy


golden_funcs = {
    "floor_div": numpy.floor_divide,
    #"real_div": numpy.true_divide,
    "neg": numpy.negative,
    "acos": numpy.arccos,
    "acosh": numpy.arccosh,
    "asin": numpy.arcsin,
    "asinh": numpy.arcsinh,
    "atan": numpy.arctan,
    "atan2": numpy.arctan2,
    "atanh": numpy.arctanh,
    "assign_sub": numpy.subtract,
    "assign_add": numpy.add,
    "mod": numpy.fmod,
    #"sub": numpy.subtract,
    #"mul": numpy.multiply,  # comment for complex32
    "add_ascendc": numpy.add,
    "is_finite": numpy.isfinite,
    "is_nan": numpy.isnan,
    "is_inf": numpy.isinf,
    "is_pos_inf": numpy.isposinf,
    "is_neg_inf": numpy.isneginf,
    "is_close": numpy.isclose,
}


golden_needs_ori_inputs = set()
# dma_copy ops don't need to transfer INF/NAN in overflow_mode=0 scenario.
user_specified_dma_copy_ops = set()


def register_golden(operator_names: Sequence[str],
                    need_original_input_arrays: bool = False,
                    dma_copy_op: bool = False):
    """Register golden function"""
    if not isinstance(operator_names, (list, tuple)):
        raise TypeError("Register function for golden funcs must receive a list or tuple, not %s"
                        % str(operator_names))

    def __inner_golden_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in golden_funcs:
                logging.warning("golden function of %s has already been registered!" % operator_name)
            golden_funcs[operator_name] = __wrapper
            if need_original_input_arrays:
                golden_needs_ori_inputs.add(operator_name)
            if dma_copy_op:
                user_specified_dma_copy_ops.add(operator_name)
        return __wrapper

    return __inner_golden_registry

