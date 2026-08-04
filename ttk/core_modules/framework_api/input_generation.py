#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""
Input data generation for framework_api tests.

Generates numpy arrays from testcase metadata, applies custom plugin overrides,
and converts to framework tensors (torch).
"""
import numpy as np

from ttk.core_modules.plugin_loader import get_plugin_function
from ttk.utilities import get
from ttk.utilities.container_utils import apply_as_list
from ttk.utilities.data import RandomData, resolve_custom_numpy_dtypes
from ttk.utilities.torch_ops_package_loader import TorchOpsPackageLoader


def generate_inputs(testcase, switches, backend, plan, stored_inputs=None):
    """Generate numpy input arrays. Default generation first, then optional custom plugin.

    Custom input plugin interface (signature matches API definition, same as golden plugin):
        def my_input_torch_gather(input, dim, index, sparse_grad=False, out=None):
            input[:] = np.zeros_like(input)
            index[:] = np.random.randint(0, input.shape[1], size=index.shape, dtype=index.dtype)

    Plugin has no return value, modifies tensor arrays in-place via x[:] = value.
    Called with same ParamPlan arg order as profiling execution and golden generation.
    """
    # Custom input runs before API resolution in worker processes. Register the
    # installed torch.ops package before a callback calls a companion metadata op.
    TorchOpsPackageLoader.ensure_registered(testcase.api_name)
    if stored_inputs is not None:
        testcase.np_storages = list(stored_inputs)
        raw_inputs = build_views_from_storages(testcase)
        _set_runtime_tensors(testcase, raw_inputs)
        return raw_inputs

    raw_inputs = default_generate_inputs(testcase, switches)
    override_tensors_from_attributes(testcase, raw_inputs)

    plugin_path = switches.plugin_path
    input_func = get_plugin_function(
        testcase.api_name, "input", "e2e", plugin_path
    )
    if input_func is not None:
        use_torch = testcase.is_torch_dtype_support()
        dist = testcase.tensor_list_dist
        if use_torch:
            plugin_inputs = np_to_torch_inputs(testcase, raw_inputs)
        else:
            plugin_inputs = raw_inputs
        if dist:
            nested_for_plugin = apply_as_list(plugin_inputs, dist)
        else:
            nested_for_plugin = plugin_inputs
        args, kwargs, extra_attrs = plan.build_args(nested_for_plugin)
        extra = {
            'backend': backend.alias(),
            'tensor_formats': testcase.tensor_formats,
            'tensor_dtypes': testcase.tensor_dtypes,
            'use_torch': use_torch,
            'short_soc_version': switches.short_soc_version,
            'testcase_name': testcase.testcase_name,
            'input_ranges': testcase.input_data_ranges,
        }
        extra.update(extra_attrs)

        if hasattr(testcase, 'batch_axis') and testcase.batch_axis is not None:
            extra['batch_axis'] = testcase.batch_axis
        if hasattr(testcase, 'batch_slice_info') and testcase.batch_slice_info is not None:
            extra['batch_slice_info'] = testcase.batch_slice_info
        if hasattr(testcase, 'batch_seed') and testcase.batch_seed is not None:
            extra['batch_seed'] = testcase.batch_seed

        import inspect
        sig = inspect.signature(input_func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            kwargs.update(extra)
        else:
            kwargs.update({k: v for k, v in extra.items() if k in sig.parameters})
        input_func(*args, **kwargs)

    _set_runtime_tensors(testcase, raw_inputs)
    return raw_inputs


def _set_runtime_tensors(testcase, raw_inputs):
    """Rebuild framework tensors and TensorList nesting from backing storages."""
    use_torch = testcase.is_torch_dtype_support()
    if use_torch:
        flat_tensors = np_to_torch_inputs(testcase, raw_inputs)
    else:
        flat_tensors = list(raw_inputs)
    dist = testcase.tensor_list_dist
    if dist:
        testcase.tensors = apply_as_list(flat_tensors, dist)
    else:
        testcase.tensors = flat_tensors


def np_to_torch_inputs(testcase, raw_inputs):
    """Convert to torch tensors preserving non-contiguous views via torch.as_strided.

    Uses testcase.np_storages (contiguous) for from_numpy, then rebuilds
    non-contiguous views from testcase metadata (view_shape/view_stride/view_offset),
    same approach as aclnn input_generation.
    """
    import torch
    from ttk.utilities.dtypes import numpy_to_torch_tensor
    np_storages = getattr(testcase, 'np_storages', None)
    if np_storages is None:
        return [torch.from_numpy(np.ascontiguousarray(arr)) if arr is not None else None
                for arr in raw_inputs]
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = testcase.flat_tensor_dtypes
    result = []
    for idx, raw in enumerate(raw_inputs):
        if raw is None:
            result.append(None)
            continue
        storage = np_storages[idx]
        dtype = get(flat_dtypes, idx) if flat_dtypes else None
        t = numpy_to_torch_tensor(storage, is_complex32="complex32" in str(dtype or ""))
        v_shape = flat_shapes[idx]
        v_stride = testcase.flat_view_stride(idx)
        v_offset = testcase.flat_view_offset(idx)
        if t.dim() == 0 and v_shape == ():
            result.append(t)
        else:
            result.append(torch.as_strided(t, v_shape, v_stride, v_offset))
    return result


def assign_tensor_value(arr, val, label):
    """Assign a Python value to a numpy array (possibly scalar/non-contiguous/empty)."""
    if arr.size == 0:
        return
    spec = np.array(val, dtype=arr.dtype)
    try:
        if arr.ndim == 0:
            arr[...] = spec.item()
        else:
            arr[:] = spec
    except ValueError as e:
        raise ValueError(f"Specify tensor [{label}] from `attributes` fail: {e}")


def override_tensors_from_attributes(testcase, raw_inputs):
    """Override tensor values from testcase attributes dict by parameter name.

    For regular tensor: val is scalar or array broadcastable to target shape.
    For TensorList, two modes:
      1. Scalar broadcast: val is a single number or [scalar] -> all sub-tensors get same value
      2. Per-tensor: val is a list, len must match sub-tensor count -> applied one-by-one
    """
    info = testcase.get_api_info()
    if not info or not testcase.attributes:
        return
    dist = testcase.tensor_list_dist
    if dist:
        nested_np = apply_as_list(raw_inputs, dist)
    else:
        nested_np = list(raw_inputs)
    tensor_params = info.tensors
    for idx, param in enumerate(tensor_params):
        if idx >= len(nested_np):
            break
        if nested_np[idx] is None:
            continue
        if param.name not in testcase.attributes:
            continue
        val = testcase.attributes[param.name]
        if param.is_tensor_like and val in (None, 'None', ''):
            raise ValueError(
                f"[{testcase.testcase_name}] Invalid testcase: param '{param.name}' "
                f"has a tensor in view_shapes but attributes specifies "
                f"{repr(val)}.  Tensor params should be provided via "
                f"view_shapes or with a concrete value in attributes, not None.")
        if param.is_tensor_list:
            sub_tensors = nested_np[idx]
            num_sub = len(sub_tensors)
            if not isinstance(val, (list, tuple)):
                per_tensor_val = [val] * num_sub
            elif len(val) == 1 and not isinstance(val[0], (list, tuple)):
                per_tensor_val = [val[0]] * num_sub
            else:
                if len(val) != num_sub:
                    raise ValueError(
                        f"Specify TensorList [{param.name}] for case [{testcase.testcase_name}] "
                        f"from `attributes` length mismatch: got {len(val)}, expected {num_sub}.")
                per_tensor_val = list(val)
            for j in range(num_sub):
                assign_tensor_value(sub_tensors[j], per_tensor_val[j],
                                    f"{param.name}[{j}]")
        else:
            assign_tensor_value(nested_np[idx], val, param.name)


def default_generate_inputs(testcase, switches):
    """Default input generation -- flat iteration over all tensors.

    Iterates flat_tensor_view_shapes so nested TensorList structures
    are handled correctly (each sub-tensor generated independently).
    Returns flat list of numpy arrays (possibly non-contiguous views).
    """
    generate_np_storages(testcase, switches)
    return build_views_from_storages(testcase)


def generate_np_storages(testcase, switches):
    """Generate contiguous numpy storage arrays for all tensors.

    Input tensors get random data; pure output tensors get fixed values.
    Stores result in testcase.np_storages.
    """
    np_storages = []
    distribution = switches.input_distribution
    pure_output_indexes = set(testcase.pure_output_indexes)
    flat_shapes = testcase.flat_tensor_view_shapes
    flat_dtypes = resolve_custom_numpy_dtypes(testcase.flat_tensor_dtypes)
    ranges = testcase.flat_input_data_ranges or ()

    for idx, view_shape in enumerate(flat_shapes):
        if view_shape is None:
            np_storages.append(None)
            continue

        dtype = flat_dtypes[idx]
        s_shape = testcase.flat_storage_shape(idx)
        data_range = ranges[idx] if idx < len(ranges) else (None, None)

        if idx not in pure_output_indexes:
            rd = RandomData(dtype, s_shape, data_range)
            np_storages.append(rd.generate(distribution))
        else:
            from ttk.utilities.data import fixed_np_array
            init_val = 0 if testcase.api_name in ("torch.ones",) else 1
            np_storages.append(fixed_np_array(dtype, s_shape, init_value=init_val))

    testcase.np_storages = np_storages


def build_views_from_storages(testcase):
    """Build (possibly non-contiguous) numpy views from contiguous storages."""
    flat_shapes = testcase.flat_tensor_view_shapes
    inputs = []
    for idx, storage in enumerate(testcase.np_storages):
        if storage is None:
            inputs.append(None)
            continue
        view_shape = flat_shapes[idx]
        v_stride = testcase.flat_view_stride(idx)
        v_offset = testcase.flat_view_offset(idx)

        if v_stride is not None or v_offset != 0:
            inputs.append(to_non_contiguous_view(storage, view_shape, v_stride, v_offset))
        else:
            inputs.append(storage)
    return inputs


def to_non_contiguous_view(storage, view_shape, view_stride, view_offset):
    """Create non-contiguous view from contiguous storage using numpy as_strided."""
    from ttk.utilities.dtypes import np_as_strided_safe
    dtype = storage.dtype
    byte_strides = tuple(s * dtype.itemsize for s in view_stride)
    if view_offset and view_offset > 0:
        base = storage.ravel()[view_offset:]
    else:
        base = storage.ravel()
    return np_as_strided_safe(base, shape=view_shape, strides=byte_strides)
