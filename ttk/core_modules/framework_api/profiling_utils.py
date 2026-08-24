#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under
# the terms of conditions of CANN Open Software License Agreement Version 2.0
# (the "License").
# See LICENSE in the root of the software repository for the full text of the License.


"""
Profiling utility functions shared between profiling.py and graph_execution.py.

Framework-neutral — all framework-specific logic is delegated to backend methods.
"""

import numpy as np


def apply_format_cast(tensors, formats):
    """Apply NPU format cast to tensors that require non-default formats."""
    from ttk.utilities import FORMAT_DICT, PRIVATE_FORMATS

    if not formats or all(f not in PRIVATE_FORMATS for f in formats):
        return tensors

    import torch_npu

    result = []
    for tensor, fmt in zip(tensors, formats):
        if fmt and fmt in PRIVATE_FORMATS:
            tensor = torch_npu.npu_format_cast(tensor, FORMAT_DICT[fmt])
        result.append(tensor)
    return result


def prepare_device_args(testcase, backend, dev_id, plan, raw_inputs):
    """Prepare device tensors and build args/kwargs for API execution.

    Shared logic between eager and graph execution modes:
    1. Convert raw_inputs to device tensors (preserving stride for torch)
    2. Apply NPU format cast if needed
    3. Apply tensor list distribution if needed
    4. Build args/kwargs using plan

    Args:
        testcase: TestcaseE2e
        backend: Backend instance
        dev_id: device ID
        plan: ParamPlan
        raw_inputs: numpy input arrays

    Returns:
        tuple: (args, kwargs) ready for API call
    """
    from ttk.utilities.container_utils import apply_as_list

    use_framework_tensors = getattr(testcase, "tensors", None) is not None
    if use_framework_tensors:
        flat_tensors = testcase.flatten_tensors
        preserve_stride = not backend.needs_numpy_fallback(testcase)
        dev_tensors = [
            backend.to_device(t, dev_id, preserve_stride=preserve_stride) if t is not None else None
            for t in flat_tensors
        ]
    else:
        dev_tensors = [backend.to_device(x, dev_id) if x is not None else None for x in raw_inputs]
    if testcase.tensor_formats and backend.supports_format_cast():
        dev_tensors = apply_format_cast(dev_tensors, testcase.flat_tensor_formats)
    dist = testcase.tensor_list_dist
    if dist:
        nested_tensors = apply_as_list(dev_tensors, dist)
    else:
        nested_tensors = dev_tensors
    args, kwargs, _ = plan.build_args(nested_tensors)
    return args, kwargs


def unpack_4bit_outputs(testcase, result_nps):
    """Unpack uint8 packed 4-bit outputs to float4/int4 numpy arrays.

    When torch has no native float4 dtype, the NPU op may return uint8
    (packed) data. This post-processing step unpacks it based on the
    testcase's declared output dtypes, mirroring the _decode_output_bytes
    logic in ACLNN comparison.py and Kernel comparison.py.
    """
    if not result_nps:
        return result_nps
    output_dtypes = getattr(testcase, "flat_output_dtypes", None)
    if not output_dtypes:
        return result_nps

    from ttk.utilities import unpack_4bits

    for idx, dtype_str in enumerate(output_dtypes):
        if idx >= len(result_nps) or result_nps[idx] is None:
            continue
        if not isinstance(result_nps[idx], np.ndarray):
            continue
        ds = str(dtype_str)
        if ("float4" in ds or "int4" in ds) and result_nps[idx].dtype == np.uint8:
            try:
                result_nps[idx] = unpack_4bits(result_nps[idx], ds)
            except Exception:
                pass
    return result_nps
