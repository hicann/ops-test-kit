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
Comparison utilities for dynamic shape
"""
# Standard Packages
import logging
from typing import Tuple, Union

import numpy

# Third-party Packages
from ....utilities import get, resolve_custom_numpy_dtypes, unpack_4bits
from ...comparison.custom import compare_with_hooks
from .profiling_structure import ComparisonResult


def comparing(dyn_kernel_name: str, cst_kernel_name: str, bin_kernel_name: str,
              dyn_outputs: Tuple[Union[str, numpy.ndarray]],
              cst_outputs: Tuple[Union[str, numpy.ndarray]],
              bin_outputs: Tuple[Union[str, numpy.ndarray]],
              goldens: Tuple[Union[str, numpy.ndarray]],
              output_dtypes: tuple,
              *, standards, third_parties=None, testcase=None,
              pre_compare=None, custom_compare=None) -> ComparisonResult:
    # outputs may hold sentinels (DYN_OFF/None) of different length than dtypes;
    # __outputs_to_numpy_arrays + compare() skip sentinels — no length assert.
    __outputs_to_numpy_arrays(dyn_outputs, output_dtypes)
    __outputs_to_numpy_arrays(cst_outputs, output_dtypes)
    __outputs_to_numpy_arrays(bin_outputs, output_dtypes)
    try:
        logging_data = "\n"
        _std_tokens = sorted({str(s.token) for s in standards}) if standards else []
        _std = ",".join(_std_tokens) if _std_tokens else "unknown"
        logging_data += f"Comparing {dyn_kernel_name} with {_std}\n"
        dyn_precision, _logging_data, d_passed, dyn_m = _compare_mode(
            testcase, dyn_outputs, goldens, output_dtypes, standards,
            third_parties, pre_compare, custom_compare)
        logging_data += _logging_data
        logging_data += f"Comparing {cst_kernel_name} with {_std}\n"
        cst_precision, _logging_data, c_passed, cst_m = _compare_mode(
            testcase, cst_outputs, goldens, output_dtypes, standards,
            third_parties, pre_compare, custom_compare)
        logging_data += _logging_data
        logging_data += f"Comparing {bin_kernel_name} with {_std}\n"
        bin_precision, _logging_data, b_passed, bin_m = _compare_mode(
            testcase, bin_outputs, goldens, output_dtypes, standards,
            third_parties, pre_compare, custom_compare)
        logging_data += _logging_data
        logging.debugc(logging_data)

        passed = "PASS" if all([d_passed, c_passed, b_passed]) else "FAIL"
        metrics = {"dyn": dyn_m, "cst": cst_m, "bin": bin_m}
    except Exception:
        logging.exception("Comparison failed")
        return ComparisonResult("COMPARE_FAILURE")
    return ComparisonResult(None).set(dyn_precision, cst_precision, bin_precision, passed, metrics)


def _compare_mode(testcase, outputs, goldens, output_dtypes, standards,
                  third_parties, pre_compare, custom_compare):
    # 钩子比对逻辑已上提到 comparison/custom.py::compare_with_hooks 供各通路共用
    # （GEIR 原先没有这套逻辑，自实现 compare 在该通路静默失效）。此处保留同名薄封装，
    # kernel 通路调用点与行为均不变。
    return compare_with_hooks(testcase, outputs, goldens, output_dtypes,
                              standards, third_parties, pre_compare, custom_compare)


def __outputs_to_numpy_arrays(outputs, output_dtypes):
    output_dtypes = resolve_custom_numpy_dtypes(output_dtypes)
    for idx, output in enumerate(outputs):
        if isinstance(output, (str, type(None))):
            continue
        else:
            output_dtype = get(output_dtypes, idx)
            # Probably ctypes char array
            if "complex32" in str(output_dtype):
                outputs[idx] = numpy.frombuffer(output, "float16")
            elif "int4" in str(output_dtype) or "float4" in str(output_dtype):
                output_uint8 = numpy.frombuffer(output, "uint8")
                outputs[idx] = unpack_4bits(output_uint8, output_dtype)
            elif "uint1" == str(output_dtype):
                outputs[idx] = numpy.frombuffer(output, "uint8")
            else:
                outputs[idx] = numpy.frombuffer(output, output_dtype)
