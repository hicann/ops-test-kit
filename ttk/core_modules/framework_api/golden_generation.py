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

from ttk.core_modules.plugin_loader import get_plugin_function
from ttk.core_modules.testcase_manager.param_plan import build_positional_args
from ttk.utilities import DTYPE_PROMOTE_MAP
from ttk.utilities.container_utils import apply_as_list

from .api_resolver import resolve_api
from .framework_api_info_keeper import FrameworkApiInfoKeeper
from .framework_detector import detect_framework

_cpu_backend_cache = {}


def _get_cpu_backend(framework="torch"):
    """Get cached CPU backend for the given framework."""
    if framework not in _cpu_backend_cache:
        if framework == "tf":
            from .backends.cpu_tf_backend import CpuTfBackend

            _cpu_backend_cache[framework] = CpuTfBackend()
        else:
            from .backends.cpu_torch_backend import CpuTorchBackend

            _cpu_backend_cache[framework] = CpuTorchBackend()
    return _cpu_backend_cache[framework]


def _promote_raw_inputs(testcase, raw_inputs, switches=None):
    """golden_mode=Promote 时按 DTYPE_PROMOTE_MAP 抬高浮点输入精度(fp32->fp64, fp16/bf16->fp32),整型不动。

    cross_check 用「我方误差 / 竞品误差」判定,前提是 golden 必须是独立于双方的高精度真值。
    E2E 的 golden 与竞品同为 torch aten:golden 若停留在被测 dtype,两者实现同源、竞品误差
    恒为 0,safe_div 的 err 地板接管,判据退化成「是否与 torch 逐位一致」——一次末位舍入
    就会被放大成数倍比值。kernel/aclnn 通路早已用 golden_mode=Promote 规避,E2E 此前没有,
    故在此补齐,语义与 npu/op/output_generation.py 的 __promote_dtype 一致。
    """
    # 模式判定放在此处而非调用点:让 generate_golden 少一层分支。
    # golden_mode_override 由 profiling 侧在 cross_check 判据下设置。
    mode = getattr(testcase, "golden_mode_override", None) or getattr(switches, "golden_mode", None)
    if mode != "Promote":
        return raw_inputs
    # TestcaseE2e 暴露的是 flat_tensor_dtypes(与 raw_inputs 一一对齐);
    # flat_input_dtypes 在该类上并不存在,取到 None 会让本函数直接原样返回、Promote 空转。
    flat_dtypes = getattr(testcase, "flat_tensor_dtypes", None)
    if not flat_dtypes:
        return raw_inputs
    promoted = list(raw_inputs)
    for idx, array in enumerate(promoted):
        if array is None or idx >= len(flat_dtypes):
            continue
        target = DTYPE_PROMOTE_MAP.get(flat_dtypes[idx])
        if target is None:
            continue
        try:
            promoted[idx] = array.astype(target)
        except (TypeError, ValueError):
            # 只兜 numpy astype 实际会抛的两类:目标 dtype 非法(TypeError)、值无法转换(ValueError)。
            # 其余异常(如 array 非 ndarray)属调用方传参错误,不在此吞掉。
            logging.warning(f"[golden] promote input#{idx} {flat_dtypes[idx]}->{target} failed, keep original")
    return promoted


def generate_golden(testcase, raw_inputs, plugin_path=None, switches=None, backend="cpu"):
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
    golden_api = getattr(testcase, "golden_api", None)
    dist = testcase.tensor_list_dist
    api_name = testcase.api_name
    framework = detect_framework(api_name)
    cpu_backend = _get_cpu_backend(framework)

    # 抬高浮点输入精度后再算 golden,使其成为独立于被测与竞品的高精度真值。
    # 覆盖面:golden_api 路径与「同 API 跑 CPU」路径(两者都吃这里的 raw_inputs)。
    # 自定义插件路径(Priority 2)不经 raw_inputs、由插件自行取数,故不受此处影响。
    raw_inputs = _promote_raw_inputs(testcase, raw_inputs, switches)

    # --- Priority 1: CSV golden_api column (different API) ---
    if golden_api:
        if golden_api.lower() == "disable":
            return ["SUPPRESSED"]
        if cpu_backend.is_npu_only(golden_api):
            return ["UNSUPPORTED"]
        if not testcase.is_dtype_support():
            logging.debug(f"[golden] Skip golden_api={golden_api} for {api_name}: non-native dtype detected")
            return ["UNSUPPORTED"]
        if golden_api == api_name:
            golden_api_info = testcase.get_api_info()
        else:
            golden_api_info = FrameworkApiInfoKeeper().get(golden_api)
        return _run_api_on_cpu(
            golden_api, raw_inputs, testcase, dist, api_info=golden_api_info, cpu_backend=cpu_backend
        )

    # --- Priority 2: Custom plugin via plugin_loader ---
    func = get_plugin_function(api_name, "golden", "e2e", plugin_path)
    if func is not None:
        return _call_plugin_with_plan(testcase, func, switches, backend, cpu_backend)

    # --- Priority 3: Same API on CPU ---
    if not testcase.is_dtype_support():
        logging.debug(f"[golden] Skip CPU golden for {api_name}: non-native dtype detected, no custom plugin found")
        return ["UNSUPPORTED"]
    if cpu_backend.is_npu_only(api_name):
        return ["UNSUPPORTED"]
    try:
        return _run_api_on_cpu(api_name, raw_inputs, testcase, dist, cpu_backend=cpu_backend)
    except Exception as e:
        raise RuntimeError(
            f"{api_name} cannot run on CPU and has no e2e custom plugin. "
            f"Please provide a custom plugin via --plugin or specify golden_api in CSV."
        ) from e


def _run_api_on_cpu(api_name, raw_inputs, testcase, dist, api_info=None, cpu_backend=None):
    """Execute API on CPU, return numpy result list.

    api_info=None: reuse testcase's param plan (same API path).
    api_info=...: independently match overload (golden_api path).
    """
    if cpu_backend is None:
        framework = detect_framework(api_name)
        cpu_backend = _get_cpu_backend(framework)
    cpu_inputs = [cpu_backend.from_numpy(x.copy()) if x is not None else None for x in raw_inputs]
    from .tf_stateful import get_mutable_param_indexes

    # is_ref 下标与张量参数对位; TensorList 会造成 flat 下标偏移, 此时不用启用
    # (mutable-ref 算子没有 TensorList 参数; dist 编码: 0=普通张量, >0=TensorList)。
    # import tensorflow 放进分支: torch 进程后加载 TF 的 C 扩展会段错误
    mutable_idx = get_mutable_param_indexes(api_name) if not any(d > 0 for d in (dist or ())) else ()
    if mutable_idx:
        import tensorflow as tf

        for idx in mutable_idx:
            if idx < len(cpu_inputs) and cpu_inputs[idx] is not None:
                cpu_inputs[idx] = tf.Variable(cpu_inputs[idx])
    nested = apply_as_list(cpu_inputs, dist) if dist else cpu_inputs

    if api_info is not None:
        args, kwargs, oidx = build_positional_args(
            api_name,
            nested,
            testcase.attributes or {},
            testcase.output_tensor_indexes,
            tensor_distribution=[d > 0 for d in dist] if dist else None,
            api_info=api_info,
        )
    else:
        plan = testcase.get_param_plan()
        args, kwargs, _ = plan.build_args(nested)
        oidx = plan.overload_index

    return _exec_and_convert(api_name, args, kwargs, oidx, cpu_backend)


def _call_plugin_with_plan(testcase, func, switches=None, backend="cpu", cpu_backend=None):
    """Call golden plugin using testcase's param plan — same arg order as profiling.
    Interface is identical to input plugin, except golden returns values.
    Uses testcase.tensors (CPU nested tensors) directly, like aclnn.
    """
    plan = testcase.get_param_plan()
    if plan is None:
        raise RuntimeError(f"No param plan for {testcase.api_name}, cannot call golden plugin")
    if cpu_backend is None:
        framework = detect_framework(testcase.api_name)
        cpu_backend = _get_cpu_backend(framework)
    use_numpy = cpu_backend.needs_numpy_fallback(testcase)

    args, kwargs, extra_attrs = plan.build_args(testcase.tensors)
    extra = {
        "backend": backend,
        "tensor_formats": testcase.tensor_formats,
        "tensor_dtypes": testcase.tensor_dtypes,
        "use_numpy": use_numpy,
        "short_soc_version": getattr(switches, "short_soc_version", None),
        "testcase_name": testcase.testcase_name,
    }
    extra.update(extra_attrs)

    if hasattr(testcase, "batch_axis") and testcase.batch_axis is not None:
        extra["batch_axis"] = testcase.batch_axis
    if hasattr(testcase, "batch_slice_info") and testcase.batch_slice_info is not None:
        extra["batch_slice_info"] = testcase.batch_slice_info
    if hasattr(testcase, "batch_seed") and testcase.batch_seed is not None:
        extra["batch_seed"] = testcase.batch_seed

    import inspect

    sig = inspect.signature(func)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        kwargs.update(extra)
    else:
        kwargs.update({k: v for k, v in extra.items() if k in sig.parameters})
    result = func(*args, **kwargs)
    return cpu_backend.result_to_numpy(result)


def _exec_and_convert(api_name, args, kwargs, overload_index=0, cpu_backend=None):
    """Execute API and convert result to list of numpy arrays."""
    from .eager_execution import call_api

    resolved, is_tensor_method = resolve_api(api_name)
    if cpu_backend is None:
        framework = detect_framework(api_name)
        cpu_backend = _get_cpu_backend(framework)
    with cpu_backend.device_scope(0):
        if is_tensor_method:
            if args[0] is None:
                return [None]
            result = call_api(api_name, overload_index, getattr(args[0], resolved), args[1:], kwargs)
        else:
            result = call_api(api_name, overload_index, resolved, args, kwargs)
    return cpu_backend.result_to_numpy(result)
