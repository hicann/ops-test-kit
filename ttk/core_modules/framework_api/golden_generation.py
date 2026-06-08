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
Golden data generation with three-level fallback:
1. CSV golden_api column (per-testcase override)
2. Custom plugin via plugin_loader (level_type="e2e")
3. Default: same API on CPU

All paths use testcase.get_param_plan() to get consistent arg ordering
with profiling.py — same plan, same (*args, **kwargs) layout.
"""
import logging

from ttk.core_modules.testcase_manager.param_plan import build_positional_args
from ttk.core_modules.plugin_loader import get_plugin_function
from ttk.utilities.container_utils import apply_as_list, flatten_nested_sequence

from .api_resolver import resolve_api
from .backends.cpu_backend import CpuBackend
from .framework_api_info_keeper import FrameworkApiInfoKeeper

_cpu_backend = CpuBackend()


def generate_golden(testcase, raw_inputs, plugin_path=None, switches=None, backend='cpu'):
    """
    Generate golden data using three-level fallback.

    Args:
        testcase: TestcaseE2e
        raw_inputs: list of numpy arrays (flat, one per tensor)
        plugin_path: custom plugin path from SWITCHES
        switches: SWITCHES object
        backend: backend name string (e.g. 'cpu', 'npu')

    Returns:
        list of numpy arrays (golden outputs)
    """
    golden_api = getattr(testcase, 'golden_api', None)
    dist = testcase.tensor_list_dist
    api_name = testcase.api_name

    # --- Priority 1: CSV golden_api column (different API) ---
    if golden_api:
        if golden_api.lower() == "disable":
            return ["SUPPRESSED"]
        if golden_api.startswith(('torch_npu.', 'torch.npu')):
            return ["UNSUPPORTED"]
        if not testcase.is_torch_dtype_support():
            logging.debug(f"[golden] Skip golden_api={golden_api} for {api_name}: "
                          f"non-torch-native dtype detected")
            return ['UNSUPPORTED']
        if golden_api == api_name:
            golden_api_info = testcase.get_api_info()
        else:
            golden_api_info = FrameworkApiInfoKeeper().get(golden_api)
        return _run_api_on_cpu(golden_api, raw_inputs, testcase, dist,
                               api_info=golden_api_info)

    # --- Priority 2: Custom plugin via plugin_loader ---
    func, _ = get_plugin_function(api_name, "golden", "e2e", plugin_path)
    if func is not None:
        return _call_plugin_with_plan(testcase, func, switches, backend)

    # --- Priority 3: Same API on CPU ---
    if not testcase.is_torch_dtype_support():
        logging.debug(f"[golden] Skip CPU golden for {api_name}: "
                      f"non-torch-native dtype detected, no custom plugin found")
        return ['UNSUPPORTED']
    elif api_name.startswith(('torch_npu.', 'torch.npu')):
        return ["UNSUPPORTED"]
    try:
        return _run_api_on_cpu(api_name, raw_inputs, testcase, dist)
    except Exception as e:
        raise RuntimeError(
            f"{api_name} cannot run on CPU and has no e2e custom plugin. "
            f"Please provide a custom plugin via --plugin or specify golden_api in CSV."
        ) from e


def _run_api_on_cpu(api_name, raw_inputs, testcase, dist, api_info=None):
    """Execute API on CPU, return numpy result list.

    api_info=None: reuse testcase's param plan (same API path).
    api_info=...: independently match overload (golden_api path).
    """
    cpu_inputs = [_cpu_backend.from_numpy(x.copy()) if x is not None else None
                  for x in raw_inputs]
    if dist:
        nested = apply_as_list(cpu_inputs, dist)
    else:
        nested = cpu_inputs

    if api_info is not None:
        args, kwargs, oidx = build_positional_args(
            api_name, nested, testcase.attributes or {},
            testcase.output_tensor_indexes,
            tensor_distribution=[d > 0 for d in dist] if dist else None,
            api_info=api_info)
    else:
        plan = testcase.get_param_plan()
        args, kwargs, _ = plan.build_args(nested)
        oidx = plan.overload_index

    return _exec_and_convert(api_name, args, kwargs, oidx)


def _call_plugin_with_plan(testcase, func, switches=None, backend='cpu'):
    """Call golden plugin using testcase's param plan — same arg order as profiling.
    Interface is identical to input plugin, except golden returns values.
    Uses testcase.tensors (CPU nested tensors) directly, like aclnn.
    """
    plan = testcase.get_param_plan()
    if plan is None:
        raise RuntimeError(f"No param plan for {testcase.api_name}, cannot call golden plugin")
    use_torch = testcase.is_torch_dtype_support()

    args, kwargs, extra_attrs = plan.build_args(testcase.tensors)
    extra = {
        'backend': backend,
        'tensor_formats': testcase.tensor_formats,
        'tensor_dtypes': testcase.tensor_dtypes,
        'use_torch': use_torch,
        'short_soc_version': getattr(switches, 'short_soc_version', None),
        'testcase_name': testcase.testcase_name,
    }
    extra.update(extra_attrs)
    import inspect
    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs.update(extra)
    else:
        kwargs.update({k: v for k, v in extra.items() if k in sig.parameters})
    result = func(*args, **kwargs)
    return _to_numpy_result(result)


def _to_numpy_result(result):
    """Convert golden result to list of numpy arrays. Handles Tensor and scalar returns."""
    import numpy as np
    import torch
    if isinstance(result, (tuple, list)):
        nps = []
        for r in result:
            if r is None:
                nps.append(None)
            elif isinstance(r, torch.Tensor):
                nps.append(_cpu_backend.to_numpy(r))
            else:
                nps.append(np.array(r))
        return nps
    if isinstance(result, torch.Tensor):
        return [_cpu_backend.to_numpy(result)]
    if result is None:
        return [None]
    return [np.array(result)]


def _exec_and_convert(api_name, args, kwargs, overload_index=0):
    """Execute API and convert result to list of numpy arrays."""
    from .execution import call_api
    resolved, is_tensor_method = resolve_api(api_name)
    if is_tensor_method:
        if args[0] is None:
            return [None]
        result = call_api(api_name, overload_index,
                          getattr(args[0], resolved), args[1:], kwargs)
    else:
        result = call_api(api_name, overload_index, resolved, args, kwargs)

    return _to_numpy_result(result)
