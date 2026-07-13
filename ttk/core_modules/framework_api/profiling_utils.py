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
Profiling utility functions shared between profiling.py and graph_execution.py.
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


def result_to_numpy(result, backend, copy=False):
    """Convert API result to numpy array.

    Handles: Tensor, tuple/list of results, and scalar returns (bool/int/float/dtype).
    Scalar results are wrapped in a 0-d numpy array.
    Returns list of numpy arrays or None.
    """
    import torch
    if result is None:
        return None
    if isinstance(result, (tuple, list)):
        nps = []
        for r in result:
            if r is None:
                nps.append(None)
            elif isinstance(r, torch.Tensor):
                arr = backend.to_numpy(r)
                nps.append(arr.copy() if copy else arr)
            else:
                nps.append(np.array(r))
        return nps
    if isinstance(result, torch.Tensor):
        arr = backend.to_numpy(result)
        return [arr.copy() if copy else arr]
    return [np.array(result)]


def prepare_device_args(testcase, backend, dev_id, plan, raw_inputs):
    """Prepare device tensors and build args/kwargs for API execution.
    
    Shared logic between eager and graph execution modes:
    1. Convert raw_inputs to device tensors
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
    
    dev_tensors = [backend.to_device(x, dev_id) if x is not None else None for x in raw_inputs]
    if testcase.tensor_formats and backend.is_npu():
        dev_tensors = apply_format_cast(dev_tensors, testcase.flat_tensor_formats)
    dist = testcase.tensor_list_dist
    if dist:
        nested_tensors = apply_as_list(dev_tensors, dist)
    else:
        nested_tensors = dev_tensors
    args, kwargs, _ = plan.build_args(nested_tensors)
    return args, kwargs
