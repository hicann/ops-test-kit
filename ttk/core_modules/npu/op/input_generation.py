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
# Standard Packages
import logging
import numpy
import os

# Third-party Packages
from ...plugin_loader import get_plugin_function
from ...operator.op_info_keeper import OpInfoKeeper
from ...infershape import format_transformation
from ...testcase_manager import TestcaseOp
from ....utilities import resolve_custom_numpy_dtypes, eliminate_scalar_shapes, load_numpy_data
from ....utilities import get, get_global_storage, param_transformation, RandomData, input_apply_as_list, deep_flatten


def __use_manual_input(context: TestcaseOp):
    logging.info("Using manually configured input data")

    flat_binaries = context.flat_manual_input_binaries
    flat_shapes = eliminate_scalar_shapes(context.flat_input_shapes)
    flat_dtypes = resolve_custom_numpy_dtypes(context.flat_input_dtypes)

    input_arrays = []
    for idx, fp in enumerate(flat_binaries):
        shape = get(flat_shapes, idx)
        if shape is None or fp is None:
            input_arrays.append(None)
            continue
        logging.info(f"Reading manual input file: {fp} ...")
        input_arrays.append(load_numpy_data(fp, get(flat_dtypes, idx), shape))

    # flat → nested (与 input_shapes 同构)
    nested_arrays = input_apply_as_list(input_arrays, context.input_distribution)
    context.input_arrays = tuple(nested_arrays)
    context.actual_input_data_ranges = ((None, None),)


def __assign_tensor_value(arr, val, label):
    """Inplace: arr[:] = numpy.array(val, dtype=arr.dtype)"""
    if arr is None or arr.size == 0:
        return
    spec = numpy.array(val, dtype=arr.dtype)
    if arr.ndim == 0:
        arr[...] = spec.item()
    elif spec.shape == arr.shape:
        arr[:] = spec
    elif spec.size == arr.size:
        arr[:] = spec.reshape(arr.shape)
    else:
        arr[:] = spec


def __need_transform(ori_format, format_, ori_shape, shape):
    if ori_format == format_:
        return False
    if ori_shape is None or shape is None:
        if ori_shape is not None or shape is not None:
            raise ValueError(f"ori_shape and shape must be both None or both not None, "
                             f"got ori_shape={ori_shape}, shape={shape}")
        return False
    if tuple(ori_shape) == tuple(shape) and (ori_format == 'ND' or format_ == 'ND'):
        return False
    return True


def __transform_single_to_ori(context, arr, nested_ori, group_idx, sub_idx):
    """Transform array from current format to ori_format, update nested_ori inplace."""
    dist = context.input_distribution or ()
    flat_idx = sum(max(d, 1) for d in dist[:group_idx])
    if sub_idx is not None:
        flat_idx += sub_idx

    ori_format = get(context.flat_input_ori_formats, flat_idx)
    format_ = get(context.flat_input_formats, flat_idx)
    flat_shapes = eliminate_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = eliminate_scalar_shapes(context.flat_input_ori_shapes)
    shape_ = get(flat_shapes, flat_idx)
    ori_shape = get(flat_ori_shapes, flat_idx)

    if not __need_transform(ori_format, format_, ori_shape, shape_):
        if sub_idx is not None:
            nested_ori[group_idx][sub_idx][:] = arr
        else:
            nested_ori[group_idx][:] = arr
        return

    if not format_transformation.is_transformable(format_, ori_format):
        raise RuntimeError(f"Can not transform from [{format_}] to [{ori_format}].")

    transformed = format_transformation.transform(arr, format_, ori_format, ori_shape,
                                                  groups=context.attributes.get('groups'))
    if sub_idx is not None:
        nested_ori[group_idx][sub_idx][:] = transformed
    else:
        nested_ori[group_idx][:] = transformed


def __override_inputs_from_attributes(context: TestcaseOp):
    if not context.attributes:
        return
    op_info = OpInfoKeeper().info_of(context.op_name)
    if not op_info:
        return
    input_names = [inp["name"] for inp in op_info["inputs"]]
    attr_inputs = param_transformation(context.spec_tensors, context.dyn_func_params)
    dist = context.input_distribution or ()

    nested_arrays = list(context.input_arrays)
    nested_ori_arrays = list(context.original_input_arrays) if context.original_input_arrays else None

    for group_idx, name in enumerate(input_names):
        if group_idx >= len(nested_arrays):
            break
        if name not in attr_inputs:
            continue
        val = attr_inputs[name]
        item = nested_arrays[group_idx]
        if item is None:
            continue

        d = dist[group_idx] if group_idx < len(dist) else 0
        if d > 0:
            # TensorList
            num_sub = len(item)
            if not isinstance(val, (list, tuple)):
                per_tensor_val = [val] * num_sub
            elif len(val) == 1 and not isinstance(val[0], (list, tuple)):
                per_tensor_val = [val[0]] * num_sub
            else:
                if len(val) != num_sub:
                    raise ValueError(
                        f"Attribute [{name}] TensorList length mismatch: "
                        f"got {len(val)}, expected {num_sub}")
                per_tensor_val = list(val)
            for j in range(num_sub):
                __assign_tensor_value(item[j], per_tensor_val[j], f"{name}[{j}]")
                if nested_ori_arrays is not None and nested_ori_arrays[group_idx] is not None:
                    __transform_single_to_ori(context, item[j], nested_ori_arrays, group_idx, j)
        else:
            # Single tensor
            __assign_tensor_value(item, val, name)
            if nested_ori_arrays is not None:
                __transform_single_to_ori(context, item, nested_ori_arrays, group_idx, None)

    context.input_arrays = tuple(nested_arrays)
    if nested_ori_arrays is not None:
        context.original_input_arrays = tuple(nested_ori_arrays)


def __transform_to_original_format(context: TestcaseOp):
    switches = get_global_storage()
    if switches.golden_mode == "Disable" or context.manual_golden_binaries:
        return

    from ...utilities.container_utils import shape_like_flatten

    flat_shapes = eliminate_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = eliminate_scalar_shapes(context.flat_input_ori_shapes)
    flat_arrays = shape_like_flatten(context.input_arrays)

    ori_arrays = list(flat_arrays)
    for idx, input_array in enumerate(flat_arrays):
        if input_array is None:
            continue
        ori_format = get(context.flat_input_ori_formats, idx)
        format_ = get(context.flat_input_formats, idx)
        ori_shape = get(flat_ori_shapes, idx)
        shape_ = get(flat_shapes, idx)
        if not __need_transform(ori_format, format_, ori_shape, shape_):
            continue
        if not format_transformation.is_transformable(format_, ori_format):
            raise RuntimeError(f"Can not transform from [{format_}] to [{ori_format}].")
        transformed = format_transformation.transform(input_array, format_, ori_format, ori_shape,
                                                      groups=context.attributes.get('groups'))
        if list(transformed.shape) != list(ori_shape):
            raise RuntimeError(f"Try to transform from [{format_}] to original format [{ori_format}] failed: "
                               f"Transformed shape: {transformed.shape}, but expected shape: {ori_shape}. "
                               f"From shape is: {shape_}.")
        ori_arrays[idx] = transformed

    # flat → nested
    context.original_input_arrays = tuple(input_apply_as_list(ori_arrays, context.input_distribution))


def __realtime_random_input(context: TestcaseOp):
    """Realtime Input Data Generation (Default)"""
    switches = get_global_storage()
    flat_shapes = eliminate_scalar_shapes(context.flat_input_shapes)
    flat_ori_shapes = eliminate_scalar_shapes(context.flat_input_ori_shapes)
    flat_dtypes = resolve_custom_numpy_dtypes(context.flat_input_dtypes)

    input_arrays = []
    ori_input_arrays = []
    actual_input_data_ranges = []

    for idx, shape in enumerate(flat_shapes):
        dtype = get(flat_dtypes, idx)
        ori_format = get(context.flat_input_ori_formats, idx)
        format_ = get(context.flat_input_formats, idx)
        ori_shape = get(flat_ori_shapes, idx)
        data_range = get(context.flat_input_data_ranges, idx)

        if shape is None:
            input_arrays.append(None)
            ori_input_arrays.append(None)
            actual_input_data_ranges.append((None, None))
            continue

        if __need_transform(ori_format, format_, ori_shape, shape):
            if not format_transformation.is_transformable(ori_format, format_):
                raise RuntimeError(f"Can not transform from [{ori_format}] to [{format_}]. "
                                   f"Please check `input_ori_formats` and `input_formats`.")
            rd = RandomData(dtype, ori_shape, data_range)
            ori_arr = rd.generate(switches.input_distribution)
            transformed = format_transformation.transform(ori_arr, ori_format, format_, shape,
                                                          groups=context.attributes.get('groups'))
            if list(transformed.shape) != list(shape):
                raise RuntimeError(f"Try to transform from original format [{ori_format}] to [{format_}] failed: "
                                   f"Transformed shape: {transformed.shape}, but expected shape: {shape}. "
                                   f"From original shape is: {ori_shape}.")
            input_arrays.append(transformed)
            ori_input_arrays.append(ori_arr)
        else:
            rd = RandomData(dtype, shape, data_range)
            arr = rd.generate(switches.input_distribution)
            input_arrays.append(arr)
            ori_input_arrays.append(arr)
        actual_input_data_ranges.append(tuple(rd.data_range))

    context.input_arrays = tuple(input_apply_as_list(input_arrays, context.input_distribution))
    context.original_input_arrays = tuple(input_apply_as_list(ori_input_arrays, context.input_distribution))
    context.actual_input_data_ranges = tuple(actual_input_data_ranges)


def __gen_input(context: TestcaseOp):
    switches = get_global_storage()

    if context.manual_input_binaries:
        # Manual inputs
        __use_manual_input(context)
        # Try to convert to original format if golden needs original input arrays
        __transform_to_original_format(context)
    else:
        # Automatic random input
        __realtime_random_input(context)
        __override_inputs_from_attributes(context)

        input_func, src = get_plugin_function(context.op_name, "input", "kernel", switches.plugin_path)
        if input_func:
            if src == 'builtin':  # disabled for decouple.
                disable_builtin = int(os.getenv("TTK_DISABLE_BUILTIN", "1"))
                if disable_builtin:
                    raise RuntimeError('Special input function is not moved out.')
                else:
                    # Check special input operators
                    special_return = input_func(context)
                    if len(special_return) == 2:
                        input_arrays, _ = special_return
                    else:
                        raise RuntimeError("Special input function of operator %s returns invalid number of input %d"
                                        % (context.op_name, len(special_return)))
            else:
                input_arrays = list(context.input_arrays)
                kwargs = __collect_dynamic_kwargs(context)
                input_arrays = input_func(*input_arrays, **kwargs)
            # flatten (flat or nested) → validate → nest
            flat = deep_flatten(input_arrays)
            if len(flat) != len(context.flat_input_shapes):
                raise RuntimeError(
                    f"Input plugin returned {len(flat)} arrays, expected {len(context.flat_input_shapes)}")
            context.input_arrays = tuple(input_apply_as_list(flat, context.input_distribution))


def __collect_dynamic_kwargs(context: TestcaseOp):
    switches = get_global_storage()
    kwargs = context.attributes.copy()
    # delete internal attributes
    keys = list(kwargs.keys())
    for k in keys:
        if str(k)[0] in ('!', '#', '@'):
            del kwargs[k]
    # delete const inputs in attributes
    op_info = OpInfoKeeper().info_of(context.op_name)
    inputs = [ipt["name"] for ipt in op_info["inputs"]]
    keys = list(kwargs.keys())
    for ipt in inputs:
        if ipt in keys:
            del kwargs[ipt]
    # add some additional information.
    kwargs.update({
        'full_soc_version': switches.dev_plat,
        'short_soc_version': switches.short_soc_version,
        'testcase_name': context.testcase_name,
        'input_ori_shapes': context.input_ori_shapes,
        'output_ori_shapes': context.output_ori_shapes,
        'input_formats': context.input_formats,
        'output_formats': context.output_formats,
        'input_ori_formats': context.input_ori_formats,
        'output_ori_formats': context.output_ori_formats,
        'input_dtypes': context.input_dtypes,
        'output_dtypes': context.output_dtypes,
        'input_ranges': context.input_data_ranges
    })
    return kwargs
