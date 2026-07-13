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
Graph mode execution for framework_api.
Wraps API in GraphNetwork + torch.compile for GE graph mode testing.
"""
import functools
import logging

import torch

from ttk.test_spec import get_spec_attr

from .graph_network import GraphNetwork, split_params
from .profiler import get_profiler
from .profiling_utils import prepare_device_args, result_to_numpy

WARMUP_COUNT = 5


@functools.lru_cache(maxsize=1)
def _get_npu_backend():
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig
    config = CompilerConfig()
    return torchair.get_npu_backend(compiler_config=config)


def _compile_model(model, backend, dynamic, fullgraph):
    """Compile model with torch.compile. Returns compiled callable or raises."""
    compiled = torch.compile(
        model,
        fullgraph=fullgraph,
        backend=backend,
        dynamic=dynamic,
    )
    return compiled


def _run_compiled(compiled, args, kwargs, backend, dev_id, switches,
                  is_inplace, inplace_backup, api_name):
    """Run compiled model with warmup + profiling. Returns (result_nps, perf)."""
    result = compiled(*args, **kwargs)
    backend.synchronize(dev_id)
    result_nps = result_to_numpy(result, backend, copy=is_inplace)

    if is_inplace and inplace_backup is not None:
        args[0][:] = inplace_backup

    run_count = switches.run_time

    if switches.warmup:
        for _ in range(WARMUP_COUNT):
            compiled(*args, **kwargs)
            if is_inplace and inplace_backup is not None:
                args[0][:] = inplace_backup
        backend.synchronize(dev_id)

    profiler = get_profiler(api_name, backend)
    with profiler:
        for _ in range(run_count):
            r = compiled(*args, **kwargs)
            if is_inplace and inplace_backup is not None:
                args[0][:] = inplace_backup
            if not is_inplace:
                result = r
        backend.synchronize(dev_id)

    perf = profiler.result(backend, run_count)

    if not is_inplace:
        result_nps = result_to_numpy(result, backend)

    return result_nps, perf


def _execute_graph(testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs, dynamic):
    """
    Execute API in GE graph mode via torch.compile with profiling.

    Args:
        testcase: TestcaseE2e
        backend: Backend instance
        dev_id: device ID
        switches: SWITCHES
        plan: ParamPlan
        resolved: resolved API callable (or method name string for tensor methods)
        is_tensor_method: True if API is a Tensor method
        is_inplace: True if API modifies input in-place
        raw_inputs: numpy input arrays
        dynamic: True for dynamic shape graph, False for static shape graph

    Returns:
        (list of numpy arrays, ProfileResult) on success, or (None, None) on failure
    """
    if not backend.is_npu():
        logging.warning("Graph mode only supports NPU backend, skipping")
        return [], None

    mode_str = "dynamic" if dynamic else "static"
    logging.info(f"Executing graph mode: {mode_str}")

    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)

    inplace_backup = None

    custom_cls = get_spec_attr(testcase.api_name, "torch_graph", switches.plugin_path)

    if custom_cls:
        logging.info(f"Using custom graph module: {custom_cls.__name__}")
        init_kwargs, fwd_kwargs = split_params(
            custom_cls, plan.overload_params, args, kwargs)
        model = custom_cls(**init_kwargs)
        run_args, run_kwargs = [], fwd_kwargs
        if is_inplace and args and args[0] is not None:
            inplace_backup = args[0].clone()
        run_inplace = is_inplace
    else:
        logging.info("Using generic GraphNetwork")
        if is_inplace:
            inplace_backup = args[0].clone() if args and args[0] is not None else None

        if is_tensor_method:
            def api_caller(*args, **kwargs):
                return getattr(args[0], resolved)(*args[1:], **kwargs)
        else:
            api_caller = resolved

        model = GraphNetwork(api_caller)
        run_args, run_kwargs, run_inplace = args, kwargs, is_inplace
    try:
        npu_backend = _get_npu_backend()
    except Exception as e:
        logging.error(f"Failed to get TorchAir NPU backend: {e}")
        del args, kwargs
        return [], None

    use_fullgraph = bool(switches.fullgraph)
    try:
        compiled = _compile_model(model, npu_backend, dynamic, use_fullgraph)
        result_nps, perf = _run_compiled(
            compiled, run_args, run_kwargs, backend, dev_id, switches,
            run_inplace, inplace_backup if run_inplace else None, testcase.api_name)

    except Exception as e:
        logging.error(f"Graph {mode_str} execution failed: {e}", exc_info=True)
        del args, kwargs
        return [], None
    if inplace_backup is not None:
        del inplace_backup
    del args, kwargs
    return result_nps, perf
