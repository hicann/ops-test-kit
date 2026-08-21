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

from ttk.core_modules.npu_preprocess import invoke_npu_preprocess
from ttk.test_spec import get_spec_attr

from .graph_network import GraphNetwork, split_params
from .profiler import ProfilerConfig, get_profiler
from .profiling_utils import prepare_device_args

WARMUP_COUNT = 1


@functools.lru_cache(maxsize=1)
def _get_npu_backend():
    import torch_npu
    import torchair
    from torchair.configs.compiler_config import CompilerConfig

    config = CompilerConfig()
    return torchair.get_npu_backend(compiler_config=config)


@functools.lru_cache(maxsize=1)
def _get_npu_backend_aclgraph():
    """获取 aclgraph 模式的 NPU backend。"""
    npu_backend = "npugraph_ex"
    return npu_backend


def _compile_model(model, backend, dynamic, fullgraph):
    """Compile model with torch.compile. Returns compiled callable or raises."""
    compiled = torch.compile(
        model,
        fullgraph=fullgraph,
        backend=backend,
        dynamic=dynamic,
    )
    return compiled


def _compile_model_aclgraph(model, backend):
    """以 aclgraph 模式编译模型"""
    compiled = torch.compile(
        model,
        fullgraph=False,
        backend=backend,
        dynamic=False,
    )
    return compiled


def _run_compiled(
    compiled,
    args,
    kwargs,
    backend,
    dev_id,
    switches,
    is_inplace,
    inplace_backup,
    api_name,
    testcase_name="",
    inplace_backups=None,
    inplace_kwargs_keys=None,
):
    profiling_enabled = bool(getattr(switches, "TASK_PROFILING", True))
    deterministic = int(getattr(switches, "deterministic_level", 0) or 0) > 0
    run_count = switches.run_time if profiling_enabled or deterministic else 0
    is_kwargs_mode = inplace_kwargs_keys is not None

    result = compiled(*args, **kwargs)
    backend.synchronize(dev_id)
    result_nps = backend.result_to_numpy(result, copy=is_inplace)

    if switches.warmup and profiling_enabled:
        for _ in range(WARMUP_COUNT):
            if is_kwargs_mode:
                for idx, key in inplace_kwargs_keys.items():
                    if key in kwargs and kwargs[key] is not None:
                        kwargs[key][:] = inplace_backups[idx]
            else:
                if is_inplace and inplace_backup is not None and args:
                    args[0][:] = inplace_backup
                if inplace_backups:
                    for idx, bak in inplace_backups.items():
                        if idx < len(args) and args[idx] is not None:
                            args[idx][:] = bak
            compiled(*args, **kwargs)
        backend.synchronize(dev_id)

    if is_kwargs_mode:
        for idx, key in inplace_kwargs_keys.items():
            if key in kwargs and kwargs[key] is not None:
                kwargs[key][:] = inplace_backups[idx]
    else:
        if is_inplace and inplace_backup is not None and args:
            args[0][:] = inplace_backup
        if inplace_backups:
            for idx, bak in inplace_backups.items():
                if idx < len(args) and args[idx] is not None:
                    args[idx][:] = bak

    inplace_clones = {}
    original_tensors = {}
    if is_kwargs_mode:
        for idx, key in inplace_kwargs_keys.items():
            if key in kwargs and kwargs[key] is not None:
                original_tensors[idx] = (key, kwargs[key])
                inplace_clones[idx] = (key, [backend.clone(kwargs[key]) for _ in range(run_count - 1)])
    else:
        if inplace_backups:
            for idx in inplace_backups:
                if idx < len(args) and args[idx] is not None:
                    original_tensors[idx] = args[idx]
                    inplace_clones[idx] = [backend.clone(args[idx]) for _ in range(run_count - 1)]
        if is_inplace and inplace_backup is not None and 0 not in original_tensors:
            if args and args[0] is not None:
                original_tensors[0] = args[0]
                inplace_clones[0] = [backend.clone(args[0]) for _ in range(run_count - 1)]

    profiler = get_profiler(
        api_name,
        backend,
        ProfilerConfig(
            testcase_name=testcase_name,
            root_path=switches.root_path,
            dev_id=dev_id,
            enabled=profiling_enabled,
        ),
    )
    with profiler:
        for i in range(run_count):
            if i < run_count - 1:
                if is_kwargs_mode:
                    for _idx, (key, clones) in inplace_clones.items():
                        kwargs[key] = clones[i]
                else:
                    for idx, clones in inplace_clones.items():
                        args[idx] = clones[i]
            else:
                if is_kwargs_mode:
                    for _idx, (key, orig) in original_tensors.items():
                        kwargs[key] = orig
                else:
                    for idx, orig in original_tensors.items():
                        args[idx] = orig
            r = compiled(*args, **kwargs)
            if not is_inplace:
                result = r
        backend.synchronize(dev_id)

    perf = profiler.result(backend, max(run_count, 1))

    if not is_inplace:
        result_nps = backend.result_to_numpy(result, copy=is_inplace)

    return result_nps, perf


def _execute_graph(
    testcase,
    backend,
    dev_id,
    switches,
    plan,
    resolved,
    is_tensor_method,
    is_inplace,
    raw_inputs,
    dynamic,
    is_aclgraph=False,
):
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

    import torch_npu

    torch_npu.npu.set_device(dev_id)

    if is_aclgraph:
        mode_str = "aclgraph"
    elif dynamic:
        mode_str = "dynamic"
    else:
        mode_str = "static"
    logging.info(f"Executing graph mode: {mode_str}")

    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)
    invoke_npu_preprocess(
        testcase,
        switches,
        plan,
        args,
        kwargs,
        device_scope=lambda: backend.device_scope(dev_id),
    )

    inplace_backup = None

    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    inplace_backups = {}
    if inplace_input_indexes:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                inplace_backups[idx] = backend.clone(args[idx])

    custom_cls = get_spec_attr(testcase.api_name, "torch_graph", switches.plugin_path)

    inplace_kwargs_keys = None
    if custom_cls:
        logging.info(f"Using custom graph module: {custom_cls.__name__}")
        init_kwargs, fwd_kwargs = split_params(custom_cls, plan.overload_params, args, kwargs)
        model = custom_cls(**init_kwargs)
        run_args, run_kwargs = [], fwd_kwargs
        run_inplace = is_inplace
        if inplace_input_indexes:
            positional_param_names = [p.name for p in plan.overload_params if not p.is_keyword_only]
            inplace_kwargs_keys = {}
            for idx in inplace_input_indexes:
                if idx < len(positional_param_names):
                    key = positional_param_names[idx]
                    if key in run_kwargs:
                        inplace_kwargs_keys[idx] = key
    else:
        logging.info("Using generic GraphNetwork")
        if is_inplace:
            inplace_backup = backend.clone(args[0]) if args and args[0] is not None else None

        if is_tensor_method:

            def api_caller(*args, **kwargs):
                return getattr(args[0], resolved)(*args[1:], **kwargs)
        else:
            api_caller = resolved

        model = GraphNetwork(api_caller)
        run_args, run_kwargs, run_inplace = args, kwargs, is_inplace
    try:
        if is_aclgraph:
            npu_backend = _get_npu_backend_aclgraph()
        else:
            npu_backend = _get_npu_backend()
    except Exception as e:
        logging.error(f"Failed to get TorchAir NPU backend: {e}")
        del args, kwargs
        return [], None

    use_fullgraph = bool(switches.fullgraph)
    try:
        if is_aclgraph:
            compiled = _compile_model_aclgraph(model, npu_backend)
        else:
            compiled = _compile_model(model, npu_backend, dynamic, use_fullgraph)
        result_nps, perf = _run_compiled(
            compiled,
            run_args,
            run_kwargs,
            backend,
            dev_id,
            switches,
            run_inplace,
            inplace_backup if run_inplace else None,
            testcase.api_name,
            testcase_name=testcase.testcase_name,
            inplace_backups=inplace_backups if inplace_input_indexes else None,
            inplace_kwargs_keys=inplace_kwargs_keys,
        )

    except Exception as e:
        logging.error(f"Graph {mode_str} execution failed: {e}", exc_info=True)
        del args, kwargs
        return [], None

    if result_nps:
        if inplace_input_indexes:
            if not custom_cls:
                for idx in sorted(inplace_input_indexes):
                    if idx < len(args) and args[idx] is not None:
                        inplace_np = backend.to_numpy(args[idx].detach().clone())
                        result_nps.append(inplace_np)

    if inplace_backup is not None:
        del inplace_backup
    del args, kwargs
    return result_nps, perf
