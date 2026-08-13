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
Input generation method for Universal testcases
"""


__all__ = ["InputGenerator"]


# Standard Packages
import logging
import numpy
import os
from typing import Optional, List

# Third-party Packages
from ...testcase_manager import TestcaseAclnn
from ...aclnn import OpApiInfoKeeper, OpApiInfo
from ....utilities import apply_as_list, resolve_custom_numpy_dtypes, numpy_to_torch_tensor, tuple_flatten
from ....utilities import get, get_global_storage, RandomData
from ...plugin_loader import get_plugin_function
from ...pre_npu import add_context_if_declared, refresh_ttk_context

class InputGenerator:
    def __init__(self, context: TestcaseAclnn, ttk_context=None):
        self._ctx = context
        self._switch = get_global_storage()
        self._ttk_context = ttk_context

    def gen(self, stored_inputs=None, stored_scalars=None):
        """generate tensor & scalar."""
        if stored_inputs is not None:
            self._restore(stored_inputs, stored_scalars or ())
            if self._ttk_context is not None:
                refresh_ttk_context(self._ttk_context, self._ctx)
            return
        # Manual inputs
        if self._ctx.manual_tensor_binaries:
            self._use_manual_input()
        else:
            # Automatic random tensors
            self._realtime_random_tensors()
            # Automatic random input scalars
            self._realtime_random_scalars()

            if self._ctx.is_torch_dtype_support():
                # torch 原生 dtype：numpy.ndarray -> torch.Tensor
                self._convert_np_to_torch_tensor()
                self._package_scalars()
            else:
                # 非 torch 原生 dtype：保持 numpy.ndarray，使用 numpy as_strided
                self._convert_np_to_numpy_view()
                self._package_scalars_numpy()

            # Check special input operators
            input_func = get_plugin_function(self._ctx.api_name,
                "input", "aclnn", self._switch.plugin_path)
            if input_func:
                self._call_custom_input(input_func)
        if self._ttk_context is not None:
            refresh_ttk_context(self._ttk_context, self._ctx)

    def _restore(self, stored_inputs, stored_scalars):
        """Restore final generated state without rerunning random/input plugins."""
        self._ctx.np_storages = list(stored_inputs)
        use_torch = self._ctx.is_torch_dtype_support()
        if use_torch:
            self._convert_np_to_torch_tensor()
        else:
            self._convert_np_to_numpy_view()

        scalar_values = []
        flat_scalar_dtypes = self._ctx.flat_scalar_dtypes or ()
        for index, value in enumerate(stored_scalars):
            if value is None or not use_torch:
                scalar_values.append(value)
                continue
            dtype = get(flat_scalar_dtypes, index)
            scalar_values.append(
                numpy_to_torch_tensor(value, is_complex32="complex32" in str(dtype)).squeeze()
            )
        self._ctx.scalars = tuple(apply_as_list(
            scalar_values, self._ctx.scalar_list_dist
        ))

    def _call_custom_input(self, input_func):
        plan = self._ctx.get_param_plan()
        args, extra_attrs = plan.build_args(self._ctx.tensors, self._ctx.scalars,
                               self._ctx.attributes)
        kwargs = {
            'short_soc_version': self._switch.short_soc_version,
            'testcase_name': self._ctx.testcase_name,
            'tensor_dtypes': self._ctx.tensor_dtypes,
            'tensor_formats': self._ctx.tensor_formats,
            'scalar_dtypes': self._ctx.scalar_dtypes,
            'input_ranges': self._ctx.input_data_ranges,
            'use_torch': self._ctx.is_torch_dtype_support(),
        }
        if hasattr(self._ctx, 'batch_axis') and self._ctx.batch_axis is not None:
            kwargs['batch_axis'] = self._ctx.batch_axis
        if hasattr(self._ctx, 'batch_slice_info') and self._ctx.batch_slice_info is not None:
            kwargs['batch_slice_info'] = self._ctx.batch_slice_info
        if hasattr(self._ctx, 'batch_seed') and self._ctx.batch_seed is not None:
            kwargs['batch_seed'] = self._ctx.batch_seed
        kwargs.update(extra_attrs)
        if self._ttk_context is not None:
            add_context_if_declared(input_func, kwargs, self._ttk_context)
        input_func(*args, **kwargs)

    def _use_manual_input(self):
        # TODO
        pass

    def _realtime_random_tensors(self):
        dtypes = resolve_custom_numpy_dtypes(self._ctx.flat_tensor_dtypes)
        arrays: List[Optional[numpy.ndarray]] = []
        actual_data_ranges = []
        flat_shapes = self._ctx.flat_tensor_view_shapes

        ranges = self._ctx.flat_input_data_ranges or ()
        base_seed = getattr(self._switch, 'random_seed', None)
        batch_seed = getattr(self._ctx, 'batch_seed', None)
        for idx, vs in enumerate(flat_shapes):
            if vs is None:
                arrays.append(None)
                actual_data_ranges.append((None, None))
                continue
            data_range = ranges[idx] if idx < len(ranges) else (None, None)
            ss = self._ctx.flat_storage_shape(idx)
            dtype = get(dtypes, idx)
            if idx not in self._ctx.pure_output_indexes:
                # pure input & inplace output
                if base_seed and batch_seed is not None:
                    # batch consistency compare different case support same shape tensor has same value
                    numpy.random.seed(base_seed + idx)  
                rd = RandomData(dtype, ss, data_range)
                arrays.append(rd.generate(self._switch.input_distribution))
                actual_data_ranges.append(tuple(rd.data_range))
            else:
                # pure output. initial it as dtype(1)
                from ttk.utilities.data import fixed_np_array
                init_val = 0 if self._ctx.api_name in ("aclnnInplaceOne",) else 1
                arrays.append(fixed_np_array(dtype, ss, init_value=init_val))
                actual_data_ranges.append(data_range)
        self._ctx.np_storages = arrays
        self._ctx.actual_input_data_ranges = actual_data_ranges

    def _realtime_random_scalars(self):
        flat_scalar_dtypes = self._ctx.flat_scalar_dtypes
        dtypes = resolve_custom_numpy_dtypes(flat_scalar_dtypes)
        scalars = []

        for idx, sd in enumerate(dtypes):
            if sd is None:
                scalars.append(None)
                continue
            data_range = get(self._ctx.flat_scalar_data_ranges or (), idx, out_of_range=(None, None))
            dtype = get(dtypes, idx)
            rd = RandomData(dtype, [1], data_range)
            np_arr = rd.generate(self._switch.input_distribution)
            t_scalar = numpy_to_torch_tensor(np_arr, is_complex32="complex32" in str(dtype))
            scalars.append(t_scalar.squeeze())
        self._ctx.scalars = apply_as_list(scalars,
                                          self._ctx.scalar_list_dist)

    def _convert_np_to_torch_tensor(self):
        import torch
        torch_tensors = []
        flat_shapes = self._ctx.flat_tensor_view_shapes
        for idx, np_arr in enumerate(self._ctx.np_storages):
            if np_arr is None:
                torch_tensors.append(None)
            else:
                dtype = get(self._ctx.flat_tensor_dtypes, idx)
                v_shape = flat_shapes[idx]
                v_stride = self._ctx.flat_view_stride(idx)
                v_offset = self._ctx.flat_view_offset(idx)
                t_storage = numpy_to_torch_tensor(np_arr, is_complex32="complex32" in str(dtype))
                try:
                    if t_storage.dim() == 0 and v_shape == ():
                        t_view = t_storage
                    else:
                        t_view = torch.as_strided(t_storage, v_shape, v_stride, v_offset)
                except RuntimeError:
                    logging.error(f"torch.as_strided failed. storage_shape={t_storage.shape} "
                                  f"view_shape={v_shape}, view_stride={v_shape}, view_offset={v_offset}")
                    raise
                torch_tensors.append(t_view)
        self._ctx.tensors = apply_as_list(torch_tensors,
                                          self._ctx.tensor_list_dist)

    def _package_scalars(self):
        """
        1.package scalar-list.
        2.pick scalars defined in context.attributes.
        """
        import torch

        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        scalars = self._ctx.scalars
        scalars = scalars[:len(op_api_info.scalars)]
        for idx, s_name in enumerate(op_api_info.scalars):
            if idx >= len(scalars):
                raise RuntimeError(f"Some Scalar/ScalarList is not configured: {op_api_info.scalars[idx:]}")
            if s_name in self._ctx.attributes:
                val = self._ctx.attributes[s_name]
                if isinstance(val, (list, tuple)):
                    if not isinstance(scalars[idx], list):
                        raise RuntimeError(f"[{s_name}] is a list/tuple configured in attributes. "
                                           f"But got a scalar rather than ScalarList [{scalars[idx]}]. "
                                           f"Check scalar_dtypes nesting.")
                    if len(val) != len(scalars[idx]):
                        raise RuntimeError(f"Value count of [{s_name}] mismatch: "
                                           f"expected [{len(scalars[idx])}].")
                    for j in range(len(scalars[idx])):
                        scalars[idx][j] = torch.tensor(val[j], dtype=scalars[idx][j].dtype)
                else:
                    scalars[idx] = torch.tensor(val, dtype=scalars[idx].dtype)
        self._ctx.scalars = tuple(scalars)

    def _convert_np_to_numpy_view(self):
        """非 torch 原生 dtype 时，使用 numpy as_strided 创建 view（替代 torch.as_strided）"""
        from ttk.utilities.dtypes import np_as_strided_safe
        np_views = []
        flat_shapes = self._ctx.flat_tensor_view_shapes
        for idx, np_arr in enumerate(self._ctx.np_storages):
            if np_arr is None:
                np_views.append(None)
            else:
                v_shape = flat_shapes[idx]
                v_stride = self._ctx.flat_view_stride(idx)
                v_offset = self._ctx.flat_view_offset(idx)
                try:
                    if np_arr.ndim == 0 and v_shape == ():
                        np_views.append(np_arr)
                    else:
                        # numpy as_strided 的 strides 单位是字节，需要将元素 stride 转换
                        byte_strides = tuple(s * np_arr.itemsize for s in v_stride)
                        # 处理 storage_offset：偏移到起始位置
                        if v_offset and v_offset > 0:
                            base = np_arr.ravel()[v_offset:]
                        else:
                            base = np_arr.ravel()
                        view = np_as_strided_safe(base, shape=v_shape, strides=byte_strides)
                        np_views.append(view)
                except Exception:
                    logging.error(f"numpy.as_strided failed. storage_shape={np_arr.shape} "
                                  f"view_shape={v_shape}, view_stride={v_stride}, view_offset={v_offset}")
                    raise
        self._ctx.tensors = apply_as_list(np_views,
                                          self._ctx.tensor_list_dist)

    def _package_scalars_numpy(self):
        """非 torch 原生 dtype 时，scalar 保持为 numpy scalar/ndarray"""
        scalars = self._ctx.scalars
        op_api_info: OpApiInfo = OpApiInfoKeeper().info_of(self._ctx.api_name)
        scalars = scalars[:len(op_api_info.scalars)]
        for idx, s_name in enumerate(op_api_info.scalars):
            if idx >= len(scalars):
                raise RuntimeError(f"Some Scalar/ScalarList is not configured: {op_api_info.scalars[idx:]}")
            if s_name in self._ctx.attributes:
                val = self._ctx.attributes[s_name]
                if isinstance(val, (list, tuple)):
                    if not isinstance(scalars[idx], list):
                        raise RuntimeError(f"[{s_name}] is a list/tuple configured in attributes. "
                                           f"But got a scalar rather than ScalarList [{scalars[idx]}]. "
                                           f"Check scalar_dtypes nesting.")
                    if len(val) != len(scalars[idx]):
                        raise RuntimeError(f"Value count of [{s_name}] mismatch: "
                                           f"expected [{len(scalars[idx])}].")
                    for j in range(len(scalars[idx])):
                        scalars[idx][j] = numpy.array(val[j], dtype=scalars[idx][j].dtype)
                else:
                    scalars[idx] = numpy.array(val, dtype=scalars[idx].dtype)
        self._ctx.scalars = tuple(scalars)

    @staticmethod
    def is_broadcast(tensor) -> bool:
        """
        判断 tensor 是否是广播视图（stride 中有 0 且对应 size > 1）

        Args:
            tensor: torch.Tensor 或 None

        Returns:
            bool: True 表示是广播视图，False 表示非广播
        """
        if tensor is None:
            return False
        for size, stride in zip(tensor.shape, tensor.stride()):
            if size > 1 and stride == 0:
                return True
        return False
