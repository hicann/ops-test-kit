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
output generation method for Universal testcases
"""
# Standard Packages
import contextlib
import gc
import inspect
import logging
import numpy
import os
from typing import Sequence, Tuple, Union, List, Optional
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable

# Third-party Packages
from ...plugin_loader import get_plugin_function
from ...operator.op_info_keeper import OpInfoKeeper
from ...testcase_manager import TestcaseOp
from ....utilities import resolve_custom_numpy_dtypes, get, get_global_storage, get_dtype_range, numpy_bfloat16
from ....utilities import load_numpy_data, shape_product, ceil_div, input_apply_as_list


DTYPE_PROMOTE_MAP: dict = {
    "float16": "float32",
    "bfloat16": "float32",
    "float32": "float64",
    "complex32": "complex64",
    "complex64": "complex128",
}


def __promote_dtype(context: TestcaseOp):
    from ....utilities.container_utils import deep_flatten

    need_promote = [d in DTYPE_PROMOTE_MAP for d in context.flat_input_dtypes]
    if not any(need_promote):
        yield
        return

    # Backup
    bak_input_arrays = context.input_arrays
    bak_ori_input_arrays = context.original_input_arrays
    bak_input_dtypes = context.input_dtypes
    bak_output_dtypes = context.output_dtypes

    # --- Promote inputs: flat loop → astype → nested ---
    dist = context.input_distribution
    flat_arrays = list(deep_flatten(context.input_arrays))
    flat_ori_arrays = list(deep_flatten(context.original_input_arrays)) \
        if context.original_input_arrays else None
    flat_dtypes = context.flat_input_dtypes
    new_flat_dtypes = list(flat_dtypes)

    for idx, array in enumerate(flat_arrays):
        if get(need_promote, idx) and array is not None:
            original_dtype = get(flat_dtypes, idx)
            promoted = DTYPE_PROMOTE_MAP[original_dtype]
            if original_dtype == "complex32":
                real, imag = numpy.split(array, 2, axis=-1)
                flat_arrays[idx] = (real + imag * 1j).reshape(array.shape[:-1])
            else:
                flat_arrays[idx] = array.astype(promoted)
            new_flat_dtypes[idx] = promoted
            # ori_arrays promote in sync
            if flat_ori_arrays is not None and idx < len(flat_ori_arrays) and flat_ori_arrays[idx] is not None:
                flat_ori_arrays[idx] = flat_ori_arrays[idx].astype(promoted)

    context.input_arrays = tuple(input_apply_as_list(flat_arrays, dist))
    if flat_ori_arrays is not None:
        context.original_input_arrays = tuple(input_apply_as_list(flat_ori_arrays, dist))
    context.input_dtypes = tuple(input_apply_as_list(new_flat_dtypes, dist))
    context.invalidate_flat_cache("input_arrays", "input_dtypes")

    # --- Promote output dtypes: flat → promote → nested ---
    flat_out_dtypes = context.flat_output_dtypes
    new_flat_out_dtypes = [DTYPE_PROMOTE_MAP[d] if d in DTYPE_PROMOTE_MAP else d
                           for d in flat_out_dtypes]
    context.output_dtypes = tuple(input_apply_as_list(new_flat_out_dtypes, context.output_distribution))
    context.invalidate_flat_cache("output_dtypes")

    yield

    # Restore
    context.input_arrays = bak_input_arrays
    context.original_input_arrays = bak_ori_input_arrays
    context.input_dtypes = bak_input_dtypes
    context.output_dtypes = bak_output_dtypes
    context.invalidate_flat_cache("input_arrays", "input_dtypes", "output_dtypes")

    del flat_arrays, flat_ori_arrays
    gc.collect()


@contextlib.contextmanager
def __golden_mode(mode: str, context: TestcaseOp):
    if mode != "Promote":
        yield
    else:
        yield from __promote_dtype(context)


def __call_builtin_golden_func(context: TestcaseOp, golden_func, golden_parameters,
                               output_dtypes: list):
    switches = get_global_storage()

    with __golden_mode(switches.golden_mode, context):
        if "context" in golden_parameters:
            results = golden_func(context)
        else:
            kwargs = __correct_kwargs_for_golden(context, golden_parameters)
            # auto convert to fp32 and invoke numpy function for bfp16
            # otherwise numpy function will consider data as fp16 to compute which is wrong.
            if ((isinstance(golden_func, numpy.ufunc)
                    or inspect.getmodule(golden_func) == numpy
                    or "numpy" in str(type(golden_func)))
                and any("bfloat16" in str(arr.dtype) for arr in context.flat_input_arrays)):
                input_arrays = [arr if "bfloat16" not in str(arr.dtype) else arr.astype("float32")
                                for arr in context.flat_input_arrays]
                input_arrays = input_apply_as_list(input_arrays, context.input_distribution)
                results = golden_func(*input_arrays, **kwargs)
                results = __golden_flatten(results)
                results = [arr.astype(get(output_dtypes, i), copy=False)
                            if (isinstance(arr, numpy.ndarray) and "bfloat16" in str(get(output_dtypes, i)))
                            else arr for i, arr in enumerate(results)]
                del input_arrays
            else:
                results = golden_func(*context.input_arrays, **kwargs)
    return results


def __call_custom_golden_func(context: TestcaseOp, golden_func, golden_parameters):
    '''
    customized & decouple function like:
    def xx_xx(input0, input1, *, attr0, attr1, **kargs):
        return output0, output1
    '''
    kwargs = __collect_dynamic_golden_kwargs(context)
    results = golden_func(*context.input_arrays, **kwargs)
    return results


def __generate_golden(context: TestcaseOp, output_dtypes: list) -> list:
    switches = get_global_storage()

    golden_func, src = get_plugin_function(context.op_name, "golden", "kernel", switches.plugin_path)
    if golden_func is None:
        if context.op_name in numpy.__dir__() and isinstance(getattr(numpy, context.op_name), Callable):
            golden_func, src = getattr(numpy, context.op_name), 'builtin'
    disable_builtin = int(os.getenv("TTK_DISABLE_BUILTIN", "1"))
    if golden_func and (src != 'builtin' or not disable_builtin):
        # noinspection PyBroadException
        try:
            golden_parameters = inspect.signature(golden_func).parameters
        except:
            golden_parameters = []
        # Call golden function
        # noinspection PyBroadException
        try:
            if src == 'builtin':
                golden_results = __call_builtin_golden_func(context, golden_func, golden_parameters,
                                                            output_dtypes)
            else:  # customized & decoupled
                golden_results = __call_custom_golden_func(context, golden_func, golden_parameters)
        except TimeoutError:
            logging.exception("Golden generation timeout")
            golden_results = ["GOLDEN_TIMEOUT"]
        except Exception as e:
            logging.exception(f"Golden generation failure: {e}")
            golden_results = ["GOLDEN_FAILURE"]
        finally:
            gc.collect()
        golden_arrays = __golden_flatten(golden_results)
    else:
        logging.warning(f"Golden function for operator {context.op_name} is not provided!")
        golden_arrays = ["UNSUPPORTED"]
    return golden_arrays


def __load_golden_from_file(fp: str, dtype: str, shape: Optional[Union[list, tuple]]):
    if shape is None:
        return None
    return load_numpy_data(fp, dtype, shape)


def __gen_output(context: TestcaseOp):
    # Enable tensorflow numpy bfloat16 support
    output_dtypes = resolve_custom_numpy_dtypes(context.flat_output_dtypes)
    switches = get_global_storage()
    output_arrays = []

    if switches.golden_mode == "Disable":
        golden_arrays = ["SUPPRESSED"]
    elif context.manual_golden_binaries:
        logging.info("Using manually configured output data")
        flat_binaries = context.flat_manual_golden_binaries
        flat_shapes = context.flat_output_shapes
        golden_arrays = [__load_golden_from_file(get(flat_binaries, i), get(output_dtypes, i), get(flat_shapes, i))
                         for i in range(len(flat_shapes))]
    else:
        golden_arrays = __generate_golden(context, output_dtypes)

    # Remember: golden_arrays is always flatten.

    if len(golden_arrays) > len(context.flat_output_shapes):
        golden_arrays = golden_arrays[:len(context.flat_output_shapes)]
    __normalize_goldens(golden_arrays, output_dtypes)
    for idx, output_shape in enumerate(context.flat_output_shapes):
        out_dtype = get(output_dtypes, idx)
        if "complex32" in str(out_dtype):
            out_dtype = "float16"
            output_shape = list(output_shape) + [2]
        elif "uint1" == str(out_dtype):
            out_dtype = "uint8"
            u8_size = ceil_div(shape_product(output_shape), 8)
            output_shape = [u8_size]
        golden_shape = None
        if idx < len(golden_arrays) and isinstance(golden_arrays[idx], numpy.ndarray):
            golden_shape = golden_arrays[idx].shape
            golden_arrays[idx] = __nan_to_num(golden_arrays[idx], context.op_name, out_dtype)
        # Enable inplace
        if idx < len(context.output_inplace_indexes) and context.output_inplace_indexes[idx] is not None:
            output_arrays.append(context.output_inplace_indexes[idx])
        else:
            if golden_shape is not None:
                if output_shape and tuple(output_shape) != tuple(golden_shape):
                    logging.warning(f"Golden shape {golden_shape} not match with testcase shape "
                                    f"{output_shape}, replacing...")
                output_shape = golden_shape
            else:
                __output_shape_needs_golden(context, idx)
            output_arrays.append(numpy.ones(output_shape, dtype=out_dtype))
    __append_out_shape_unknown_golden(context, golden_arrays, output_arrays)
    context.golden_arrays = golden_arrays
    print("--->", [x.shape for x in golden_arrays])
    context.output_arrays = tuple(output_arrays)


def __normalize_goldens(golden_arrays: List[Union[str, numpy.ndarray]], output_dtypes: list):
    for idx, array in enumerate(golden_arrays):
        if not isinstance(array, numpy.ndarray):
            continue
        npu_dtype = get(output_dtypes, idx)
        if "complex32" in str(npu_dtype):
            npu_dtype = "float16"
        if "uint1" == str(npu_dtype):
            npu_dtype = "uint8"
        npu_dtype = numpy.dtype(npu_dtype).name
        if array.dtype.name == "bfloat16":
            ttk_bf16 = numpy_bfloat16()
            if array.dtype.type != ttk_bf16.dtype.type:
                golden_arrays[idx] = array = array.view(ttk_bf16)
        if array.dtype.name == npu_dtype or npu_dtype not in DTYPE_PROMOTE_MAP:
            continue
        if "complex32" in str(get(output_dtypes, idx)):
            # complex32 is a bit complicated:
            # real/imag part needs to be extracted when dtype not equal (to float16)
            # it will be complex64 or complex128
            real, imag = array.real.reshape(array.real.shape+(1,)), array.imag.reshape(array.imag.shape+(1,))
            golden_arrays[idx] = numpy.concatenate((real, imag), axis=-1)
            array = golden_arrays[idx]
        dr = get_dtype_range(npu_dtype)
        if get_global_storage().overflow_mode == 0:
            numpy.clip(array, a_min=dr[0], a_max=dr[1], out=array)
        else:
            array[array < dr[0]] = -numpy.inf
            array[array > dr[1]] = numpy.inf


def __nan_to_num(golden_array: numpy.ndarray, op: str, npu_dtype):
    from ....user_defined_modules.op.golden_funcs.registry import user_specified_dma_copy_ops
    DMA_COPY_OPS = ("as_strided", "batch_to_space", "batch_to_space_nd", "broadcast_to",
                    "concat", "depth_to_space", "diag_v2",
                    "diag_flat", "flatten",
                    "gather", "gather_nd", "gather_v2", "inv", "im2col", "masked_scatter", "mirror_pad",
                    "moe_init_routing", "moe_init_routing_v2",
                    "pack", "reverse_v2", "resize_nearest_neighbor_v2",
                    "scatter", "scatter_elements", "scatter_nd", "scatter_update", "scatter_nd_update",
                    "sign", "sign_bits_pack", "sign_bits_unpack", "slice",
                    "space_to_batch", "space_to_batch_nd", "space_to_depth", "split", "split_v",
                    "strided_slice", "strided_slice_v3", "transpose", "trans_data", "tril", "triu",
                    "view_copy", "unpack")
    if get_global_storage().overflow_mode != 0:  # 1: INF/NAN mode
        return golden_array
    if "float" not in str(npu_dtype):
        return golden_array
    if (op in DMA_COPY_OPS or
            (op.endswith("_d") and op[:-2] in DMA_COPY_OPS) or
        op in user_specified_dma_copy_ops or
            (op.endswith("_d") and op[:-2] in user_specified_dma_copy_ops)):
        return golden_array
    if get_global_storage().short_soc_version in ("Ascend310P",):  # result of this soc acts like INF/NAN mode.
        return golden_array
    npu_dtype = numpy.dtype(npu_dtype).name
    if npu_dtype in ("float32",) and \
            get_global_storage().short_soc_version in (
            "Ascend910B", "Ascend910_93", "Ascend031",
            "MC62CM12A", "Ascend950"):
        # always INF/NAN, switch not work on such dtypes + soc.
        return golden_array
    if npu_dtype == "bfloat16":
        dr = get_dtype_range("bfloat16")
        tmp = numpy.array([0, dr[1], dr[0], numpy.inf, -numpy.inf], dtype=numpy_bfloat16())
        nan = numpy.isnan(golden_array)
        if numpy.any(nan):
            golden_array = numpy.where(nan, tmp[0], golden_array)
        golden_array[golden_array == tmp[3]] = tmp[1]
        golden_array[golden_array == tmp[4]] = tmp[2]
    else:
        if numpy.any(numpy.isnan(golden_array)) or numpy.any(numpy.isinf(golden_array)):
            numpy.nan_to_num(golden_array, copy=False)
    return golden_array


def __correct_kwargs_for_golden(context: TestcaseOp, golden_parameters: list):
    attributes = context.attributes.copy()
    # Remove input names — already passed positionally via *context.input_arrays
    op_info = OpInfoKeeper().info_of(context.op_name)
    if op_info:
        for inp in op_info["inputs"]:
            attributes.pop(inp["name"], None)
    for key in tuple(attributes.keys()):
        if key not in golden_parameters:
            del attributes[key]
    return attributes


def __collect_dynamic_golden_kwargs(context: TestcaseOp):
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
        'output_dtypes': context.output_dtypes
    })
    return kwargs


def __golden_flatten(golden_results):
    from ....utilities.container_utils import deep_flatten
    if not isinstance(golden_results, Sequence):
        golden_results = [golden_results]
    return [numpy.array((x,)) if isinstance(x, numpy.generic) else x
            for x in deep_flatten(golden_results)]


def __output_shape_needs_golden(context: TestcaseOp, output_idx: int):
    if context.output_shape_unknown_indexes:
        raise RuntimeError(f"Golden must be supplied for {output_idx}th (count from 0) "
                           f"output of {context.op_name} "
                           f"since its shape depends on the value of input tensors.")


def __append_out_shape_unknown_golden(context: TestcaseOp,
                                      golden_arrays: list, output_arrays: list):
    """append output shape tensor to golden & output_arrays"""
    if not context.output_shape_unknown_indexes:
        return
    output_shape_golden = []
    for idx in sorted(context.output_shape_unknown_indexes):
        golden_shape = golden_arrays[idx].shape
        shape_len = len(golden_shape)
        if shape_len > 8:
            raise RuntimeError(f"At most 8 axes are supported for un-inferable output shape. "
                               f"But got {golden_shape}")
        # `1` is the initial value for output.
        tmp_array = numpy.ones([9], dtype=numpy.uint64)
        # set the effective number count.
        tmp_array[0] = shape_len
        tmp_array[1:shape_len + 1] = golden_shape
        output_shape_golden.extend(list(tmp_array))
    output_shape_golden_array = numpy.array(output_shape_golden, dtype="uint64")
    # mark it as uint64 encoding. set uint64 low32 bit highest bit as 1.
    # SE says only the first one needs to be set.
    output_shape_golden_array.view(numpy.uint8)[3] = 128
    output_arrays.append(numpy.ones(output_shape_golden_array.shape, dtype="uint64"))
    golden_arrays.append(output_shape_golden_array)
    context.append_output_metadata("uint64", "ND", output_shape_golden_array.shape)
