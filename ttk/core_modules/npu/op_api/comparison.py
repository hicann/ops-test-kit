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
Comparison for op api
"""


__all__ = ["Comparator"]


# Standard Packages
import logging
import numpy

# Third-party Packages
from .profiling_structure import ApiComparisonResult
from ...testcase_manager import TestcaseAclnn
from ...comparison import compare
from ...comparison.resolve import resolve_tolerance
from ....test_spec import get_spec_attr
from ....utilities import get, get_global_storage
from ....utilities import numpy_to_torch_tensor, resolve_custom_numpy_dtypes


class Comparator:
    def __init__(self, context: TestcaseAclnn):
        self._ctx = context
        self._tolerance = get_spec_attr(
            context.api_name, "tolerance",
            getattr(get_global_storage(), "plugin_path", None))

    def compare(self):
        compare_method = get_global_storage().compare_method
        output_dtypes = resolve_custom_numpy_dtypes(self._ctx.flat_output_dtypes)
        standards = resolve_tolerance(self._tolerance, self._ctx.flat_precision_tolerances,
                                      self._ctx.flat_absolute_precision, output_dtypes, compare_method)
        self._output_bytes_to_tensors()
        try:
            logging_data = "\n"
            logging_data += "Comparing %s with golden\n" % self._ctx.testcase_name
            precision, _logging_data, passed, metrics = compare(
                self._ctx.prof_result.output_bytes,
                self._ctx.golden_tensors,
                output_dtypes,
                standards=standards,
                third_parties=None)
            logging_data += _logging_data
            logging.debugc(logging_data)
            passed = "PASS" if passed else "FAIL"
        except:
            logging.exception("Comparison failed")
            return ApiComparisonResult("COMPARE_FAILURE")
        return ApiComparisonResult(None).set(precision, passed, metrics)

    def _output_bytes_to_tensors(self):
        outputs = self._ctx.prof_result.output_bytes
        dtypes = resolve_custom_numpy_dtypes(self._ctx.flat_output_dtypes)
        for idx, o in enumerate(outputs):
            if isinstance(o, str):
                continue
            if o is None:
                continue
            complex32 = False
            dtype = get(dtypes, idx)
            # Probably ctypes char array
            if "complex32" in str(dtype):
                complex32 = True
                np_array = numpy.frombuffer(o, "float16")
            else:
                np_array = numpy.frombuffer(o, dtype)

            if self._ctx.is_torch_dtype_support():
                # torch 路径
                import torch
                t_storage = numpy_to_torch_tensor(np_array,
                                                  is_complex32=complex32)
                s_shape = self._ctx.flat_output_storage_shapes[idx]
                t_storage = t_storage.reshape(s_shape)
                v_shape = self._ctx.flat_output_view_shapes[idx]
                v_offset = self._ctx.flat_output_view_offsets[idx]
                v_stride = self._ctx.flat_output_view_strides[idx]
                try:
                    if t_storage.dim() == 0 and v_shape == ():
                        outputs[idx] = t_storage
                    else:
                        outputs[idx] = torch.as_strided(t_storage, v_shape, v_stride, v_offset)
                    ret_v_shape = tuple(self._ctx.prof_result.output_view_shapes[idx])
                    if ret_v_shape != tuple(v_shape):
                        outputs[idx] = outputs[idx].resize_(ret_v_shape)
                except RuntimeError:
                    logging.error(f"torch.as_strided failed. storage_shape={t_storage.shape} "
                                  f"view_shape={v_shape}, view_stride={v_shape}, view_offset={v_offset}")
                    raise
            else:
                # numpy 路径
                from ttk.utilities.dtypes import np_as_strided_safe
                s_shape = self._ctx.flat_output_storage_shapes[idx]
                np_storage = np_array.reshape(s_shape)
                v_shape = self._ctx.flat_output_view_shapes[idx]
                v_offset = self._ctx.flat_output_view_offsets[idx]
                v_stride = self._ctx.flat_output_view_strides[idx]
                try:
                    if np_storage.ndim == 0 and v_shape == ():
                        outputs[idx] = np_storage
                    else:
                        byte_strides = tuple(s * np_storage.itemsize for s in v_stride)
                        if v_offset and v_offset > 0:
                            base = np_storage.ravel()[v_offset:]
                        else:
                            base = np_storage.ravel()
                        outputs[idx] = np_as_strided_safe(base, shape=v_shape, strides=byte_strides)
                    ret_v_shape = tuple(self._ctx.prof_result.output_view_shapes[idx])
                    if ret_v_shape != tuple(v_shape):
                        outputs[idx] = outputs[idx].reshape(ret_v_shape)
                except Exception:
                    logging.error(f"numpy.as_strided failed. storage_shape={np_storage.shape} "
                                  f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}")
                    raise
