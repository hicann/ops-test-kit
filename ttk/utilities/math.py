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
math utils
"""


__all__ = ["ceil_div", "align", "lcm",
           "is_negative_zero", "is_positive_zero",
           "positive_zero_in_array", "negative_zero_in_array"]


# Standard Packages
import numpy


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def lcm(a, b):
    """least common multiple of a and b"""
    return numpy.lcm(a, b)


def is_negative_zero(number: float) -> bool:
    # warning: return value of torch.signbit is different within torch version ...
    return bool(number == 0 and numpy.signbit(number))


def is_positive_zero(number: float) -> bool:
    return bool(number == 0 and not numpy.signbit(number))


def positive_zero_in_array(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.logical_and(array == 0, ~numpy.signbit(array))


def negative_zero_in_array(array: numpy.ndarray) -> numpy.ndarray:
    return numpy.logical_and(array == 0, numpy.signbit(array))
