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
import numpy
from typing import Tuple, Union

# Third-party Packages
from .profiling_structure import ComparisonResult
from ...comparison import compare
from ....utilities import resolve_custom_numpy_dtypes, get, unpack_4bits


def comparing(dyn_kernel_name: str, cst_kernel_name: str, bin_kernel_name: str,
              dyn_outputs: Tuple[Union[str, numpy.ndarray]],
              cst_outputs: Tuple[Union[str, numpy.ndarray]],
              bin_outputs: Tuple[Union[str, numpy.ndarray]],
              goldens: Tuple[Union[str, numpy.ndarray]],
              output_dtypes: tuple,
              *, standards, third_parties=None) -> ComparisonResult:
    # outputs may hold sentinels (DYN_OFF/None) of different length than dtypes;
    # __outputs_to_numpy_arrays + compare() skip sentinels — no length assert.
    __outputs_to_numpy_arrays(dyn_outputs, output_dtypes)
    __outputs_to_numpy_arrays(cst_outputs, output_dtypes)
    __outputs_to_numpy_arrays(bin_outputs, output_dtypes)
    try:
        logging_data = "\n"
        _std_tokens = sorted({str(s.token) for s in standards}) if standards else []
        _std = ",".join(_std_tokens) if _std_tokens else "unknown"
        logging_data += "Comparing %s with %s\n" % (dyn_kernel_name, _std)
        dyn_precision, _logging_data, d_passed, dyn_m = compare(
            dyn_outputs, goldens, output_dtypes, standards=standards, third_parties=third_parties)
        logging_data += _logging_data
        logging_data += "Comparing %s with %s\n" % (cst_kernel_name, _std)
        cst_precision, _logging_data, c_passed, cst_m = compare(
            cst_outputs, goldens, output_dtypes, standards=standards, third_parties=third_parties)
        logging_data += _logging_data
        logging_data += "Comparing %s with %s\n" % (bin_kernel_name, _std)
        bin_precision, _logging_data, b_passed, bin_m = compare(
            bin_outputs, goldens, output_dtypes, standards=standards, third_parties=third_parties)
        logging_data += _logging_data
        logging.debugc(logging_data)

        passed = "PASS" if all([d_passed, c_passed, b_passed]) else "FAIL"
        metrics = {"dyn": dyn_m, "cst": cst_m, "bin": bin_m}
    except:
        logging.exception("Comparison failed")
        return ComparisonResult("COMPARE_FAILURE")
    return ComparisonResult(None).set(dyn_precision, cst_precision, bin_precision, passed, metrics)


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
