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
Main profiling process function for framework_api.
This function runs in a subprocess — must be top-level importable.
"""
import gc
import json
import logging
import os
import numpy as np
from contextlib import nullcontext

from ttk.utilities.container_utils import get_global_storage, apply_as_list, deep_flatten
from ttk.core_modules.tbe_multiprocessing import get_process_context, DeviceLock
from ttk.core_modules.comparison.comparison import compare
from ttk.core_modules.comparison.resolve import resolve_tolerance
from ttk.core_modules.tbe_logging import default_logging_config
from ttk.utilities import get, dump_to_file, waiting_for_memory
from ttk.test_spec import get_spec_attr

from .backends import get_backend
from .eager_execution import call_api
from .api_resolver import resolve_api
from .golden_generation import generate_golden
from .graph_execution import _execute_graph
from .input_generation import generate_inputs
from .profiler import get_profiler
from .profiling_utils import apply_format_cast, prepare_device_args, result_to_numpy
from .result import FrameworkApiReturnStructure

WARMUP_COUNT = 5

def _print_get_shape(arr):
    return arr.shape if hasattr(arr, 'shape') else arr


def _print_get_dtype(arr):
    return arr.dtype if hasattr(arr, 'dtype') else arr


def _profiling_print(testcase, backend, dev_id, switches):
    separator = "-" * 100
    attr_list = []
    for k, v in testcase.attributes.items() if testcase.attributes else []:
        attr_list.append(f"{k}: {v}")
    attrs = "\n".join(attr_list) if attr_list else "N/A"
    logging.info(
        f"\n{separator}\n"
        f"API Name: {testcase.api_name}\n"
        f"Golden API: {testcase.golden_api}\n"
        f"Backend: {backend.alias()}\n"
        f"////////////// Tensors //////////////\n"
        f"Input View Shapes: {testcase.tensor_view_shapes}\n"
        f"Input Dtypes: {testcase.tensor_dtypes}\n"
        f"Input Storage Shapes: {testcase.tensor_storage_shapes}\n"
        f"Input Formats: {testcase.tensor_formats}\n"
        f"Input View Offsets: {testcase.tensor_view_offsets}\n"
        f"Input View Strides: {testcase.tensor_view_strides}\n"
        f"//////////// Output Info //////////\n"
        f"Output Tensor Indexes: {testcase.output_tensor_indexes}\n"
        f"//////////// Attributes //////////\n"
        f"{attrs}\n"
        f"//////////// Test Config //////////\n"
        f"Input Data Range: {testcase.input_data_ranges}\n"
        f"Precision Tolerance: {testcase.precision_tolerances}\n"
        f"Absolute Precision: {testcase.absolute_precision}\n"
        f"Mode: {switches.mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {dev_id}\n"
        f"{separator}"
    )


def _profiling_end_print(testcase, return_struct, golden_nps=None, switches=None):
    golden_nps = golden_nps or []
    separator_len = 103
    separator = "-" * separator_len
    end_hash = "#" * separator_len

    def _format_kernel_table(kernels_json, label=""):
        try:
            kernels = json.loads(kernels_json)
            if not kernels:
                return ""
            avg_width = 15
            max_width = 14
            min_width = 14
            max_name_len = max(len(k.get("name", "")) for k in kernels)
            max_calls = max(k.get("calls", 0) for k in kernels)
            calls_str_len = len(str(max_calls))
            calls_width = max(calls_str_len + 2, 10)
            calls_header = "# of Calls"
            total_fixed_width = avg_width + max_width + min_width + calls_width
            name_width = min(max_name_len + 2, separator_len - total_fixed_width)
            header = f"{label} Kernels" if label else ""
            table = f"\n{header}\n{separator}\n" if header else f"\n{separator}\n"
            table += f"{'Name':<{name_width}}{'avg':<{avg_width}}{'max':<{max_width}}{'min':<{min_width}}{calls_header}\n"
            table += f"{separator}\n"
            for k in kernels:
                name = k.get("name", "")
                if len(name) > name_width:
                    name = name[:name_width-2] + ".."
                avg = f"{k.get('avg', 0):.3f}"
                max_val = f"{k.get('max', 0):.3f}"
                min_val = f"{k.get('min', 0):.3f}"
                calls = k.get("calls", 0)
                table += f"{name:<{name_width}}{avg:<{avg_width}}{max_val:<{max_width}}{min_val:<{min_width}}{calls}\n"
            table += f"{separator}\n"
            return table
        except (json.JSONDecodeError, TypeError):
            return f"\n{label} Kernel Details: {kernels_json}\n"

    kernel_table = ""
    if return_struct.eager_kernel_details:
        kernel_table += _format_kernel_table(return_struct.eager_kernel_details, "Eager")
    if return_struct.graph_cst_kernel_details:
        kernel_table += _format_kernel_table(return_struct.graph_cst_kernel_details, "Graph Cst")
    if return_struct.graph_dyn_kernel_details:
        kernel_table += _format_kernel_table(return_struct.graph_dyn_kernel_details, "Graph Dyn")
    if return_struct.graph_aclgraph_kernel_details:
        kernel_table += _format_kernel_table(return_struct.graph_aclgraph_kernel_details, "Graph Aclgraph")

    lines = []
    has_eager = return_struct.eager_precision is not None
    has_graph = return_struct.graph_cst_precision is not None or return_struct.graph_dyn_precision is not None or return_struct.graph_aclgraph_precision is not None

    if has_eager:
        lines.append(f"EAGER:       {return_struct.eager_precision}")
        lines.append(f"  DEVICE: {return_struct.eager_device_perf_us} us")
        lines.append(f"  CPU:    {return_struct.eager_cpu_perf_us} us")
    if return_struct.graph_cst_precision is not None:
        lines.append(f"GRAPH CST:   {return_struct.graph_cst_precision}")
        if return_struct.graph_cst_device_perf_us is not None:
            lines.append(f"  DEVICE: {return_struct.graph_cst_device_perf_us} us")
            lines.append(f"  CPU:    {return_struct.graph_cst_cpu_perf_us} us")
    if return_struct.graph_dyn_precision is not None:
        lines.append(f"GRAPH DYN:   {return_struct.graph_dyn_precision}")
        if return_struct.graph_dyn_device_perf_us is not None:
            lines.append(f"  DEVICE: {return_struct.graph_dyn_device_perf_us} us")
            lines.append(f"  CPU:    {return_struct.graph_dyn_cpu_perf_us} us")
    if return_struct.graph_aclgraph_precision is not None:
        lines.append(f"GRAPH ACLGRAPH:   {return_struct.graph_aclgraph_precision}")
        if return_struct.graph_aclgraph_device_perf_us is not None:
            lines.append(f"  DEVICE: {return_struct.graph_aclgraph_device_perf_us} us")
            lines.append(f"  CPU:    {return_struct.graph_aclgraph_cpu_perf_us} us")

    if not lines and return_struct.precision_status is None:
        return

    lines.append(f"STATUS: {return_struct.precision_status}")

    if has_eager and not has_graph:
        title = "Eager Result Summary"
    elif has_graph and not has_eager:
        title = "Graph Result Summary"
    else:
        title = "Result Summary"

    left_pad = (separator_len - len(title) - 4) // 2
    right_pad = separator_len - len(title) - 4 - left_pad
    perf_line = f" {'#' * left_pad} {title}  {'#' * right_pad} "

    golden_shapes = tuple(_print_get_shape(g) if isinstance(g, np.ndarray) else g for g in golden_nps)
    golden_dtypes = tuple(str(_print_get_dtype(g)) if isinstance(g, np.ndarray) else g for g in golden_nps)

    msg_parts = []
    if kernel_table:
        msg_parts.append(kernel_table)
    msg_parts.append(f"\n{perf_line}")
    msg_parts.append("\n".join(lines))
    msg_parts.append(f"Golden Shapes: {golden_shapes}")
    msg_parts.append(f"Golden Dtypes: {golden_dtypes}")
    msg_parts.append(end_hash)
    logging.info("\n".join(msg_parts))


def profile_process(testcase, device_grant_events, device_granted_indices, dev_id):
    """
    Framework API profiling process — executed in subprocess.

    Args:
        testcase: TestcaseE2e
        device_grant_events: dict of device_id → Manager().Event()
        device_granted_indices: dict of device_id → Manager().Value('i', -1)
        dev_id: device ID (passed as kwarg by ProcessGroup)

    Returns:
        FrameworkApiReturnStructure
    """
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(testcase.testcase_name)

    if switches.single_testcase_log_mode:
        default_logging_config(file_handler=switches.logging_to_file,
                               testcase_name=testcase.testcase_name)

    return_struct = FrameworkApiReturnStructure()

    if not testcase.is_valid:
        return_struct.eager_precision = testcase.fail_reason
        return_struct.precision_status = "FAIL"
        _profiling_end_print(testcase, return_struct)
        return return_struct

    if not switches.no_memory_check:
        process_ctx.notify_status("OnWaitingForMemory")
        waiting_for_memory()

    backend = _get_or_create_backend(switches)

    try:
        _do_profile(testcase, backend, device_grant_events, device_granted_indices,
                    dev_id, switches, return_struct)
    except Exception as e:
        logging.error(f"[{testcase.testcase_name}] Error: {e}", exc_info=True)
        return_struct.eager_precision = str(e)
        return_struct.precision_status = "FAIL"

    return return_struct


def _get_or_create_backend(switches):
    """Get or create backend, cached per subprocess via process context."""
    process_ctx = get_process_context()
    cached = process_ctx.storage.get("framework_api_backend")
    if cached is not None:
        return cached
    backend = get_backend(switches.force_cpu)
    process_ctx.storage["framework_api_backend"] = backend
    return backend


def _dump_data(data, file_name, switches):
    """Dump data to file using global dump config."""
    dump_path = os.getenv("NPU_DUMP_PATH") or switches.root_path
    dump_to_file(data, dump_path, file_name,
                 file_format=switches.dump_config.file_format)


def _dump_inputs(testcase, raw_inputs, switches):
    """Dump input data if configured."""
    if not switches.dump_config.is_input_enabled():
        return
    dump_name = testcase.testcase_name
    for idx, arr in enumerate(raw_inputs):
        if arr is not None:
            _dump_data(arr, f"{dump_name}_input_{idx}", switches)


def _dump_outputs(testcase, result_nps, switches):
    """Dump output data if configured."""
    if not switches.dump_config.is_output_enabled():
        return
    dump_name = testcase.testcase_name
    for idx, arr in enumerate(result_nps):
        if arr is not None:
            _dump_data(arr, f"{dump_name}_output_{idx}", switches)


def _dump_goldens(testcase, golden_nps, switches):
    """Dump golden data if configured."""
    if not switches.dump_config.is_golden_enabled():
        return
    dump_name = testcase.testcase_name
    for idx, golden in enumerate(golden_nps):
        if isinstance(golden, np.ndarray):
            _dump_data(golden, f"{dump_name}_golden_{idx}", switches)


def _dump_on_fail(testcase, raw_inputs, result_nps, golden_nps, switches):
    """Force dump all data on precision failure."""
    for idx, arr in enumerate(raw_inputs):
        if arr is not None:
            _dump_data(arr, f"{testcase.testcase_name}_input_{idx}", switches)
    for idx, arr in enumerate(result_nps):
        if arr is not None:
            _dump_data(arr, f"{testcase.testcase_name}_output_{idx}", switches)
    for idx, golden in enumerate(golden_nps):
        if isinstance(golden, np.ndarray):
            _dump_data(golden, f"{testcase.testcase_name}_golden_{idx}", switches)


def _execute_eager(testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs):
    """Build device tensors, run API in eager mode with profiling, return (result_nps, perf) or raises."""
    process_ctx = get_process_context()

    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)

    run_count = switches.run_time
    profiler = get_profiler(testcase.api_name, backend)

    if is_inplace:
        inplace_backup = args[0].clone() if args and args[0] is not None else None
        if is_tensor_method:
            if args[0] is not None:
                result = call_api(testcase.api_name, plan.overload_index,
                                  getattr(args[0], resolved), args[1:], kwargs)
            else:
                result = None
        else:
            result = call_api(testcase.api_name, plan.overload_index, resolved, args, kwargs)
        backend.synchronize(dev_id)
        result_nps = result_to_numpy(result, backend, copy=True)
        if inplace_backup is not None:
            args[0][:] = inplace_backup
    else:
        result = None

    if switches.warmup:
        for _ in range(WARMUP_COUNT):
            if is_tensor_method:
                getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
            else:
                resolved(*args, **kwargs)
        backend.synchronize(dev_id)

    with profiler:
        for _ in range(run_count):
            if is_tensor_method:
                r = getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
            else:
                r = resolved(*args, **kwargs)
            if not is_inplace:
                result = r
        backend.synchronize(dev_id)

    perf = profiler.result(backend, run_count)

    if not is_inplace:
        result_nps = result_to_numpy(result, backend)

    # Plan A: read back in-place modified input tensors specified by inplace_input_indexes
    inplace_input_indexes = getattr(testcase, 'inplace_input_indexes', None) or ()
    if inplace_input_indexes:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                inplace_np = backend.to_numpy(args[idx].detach().clone())
                result_nps.append(inplace_np)

    del args, kwargs
    return result_nps, perf


def _generate_golden_data(testcase, raw_inputs, switches, backend):
    """Generate golden outputs and dump them. Returns golden_nps list."""
    process_ctx = get_process_context()
    process_ctx.notify_status("OnGenGolden")
    if (switches.golden_mode == "Disable" or
            str(testcase.golden_api).lower() == "disable"):
        golden_nps = ["SUPPRESSED"]
    else:
        try:
            golden_nps = generate_golden(
                testcase, raw_inputs, switches.plugin_path, switches,
                backend.alias())
        except Exception:
            logging.exception(f"[{testcase.testcase_name}] Golden generation failure")
            golden_nps = ["GOLDEN_FAILURE"]
    process_ctx.notify_status("OnDumpGolden")
    _dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def _apply_pre_compare(testcase, result_nps, golden_nps, switches):
    """加载并调用 pre_compare, 变换 result_nps 和 golden_nps。
    无 spec / golden 无效时什么都不做。异常向上抛。"""
    func = get_spec_attr(testcase.api_name, "pre_compare", switches.plugin_path)
    if func is None:
        return

    # result_nps / golden_nps 无效(哨兵场景: SUPPRESSED / UNSUPPORTED / GOLDEN_FAILURE)→ 跳过
    if not result_nps or not golden_nps or any(isinstance(g, str) for g in golden_nps):
        return

    # flat → nested(业务视角)
    output_dist = getattr(testcase, "output_dist", None)
    if output_dist:
        nested_result = apply_as_list(result_nps, output_dist)
        nested_golden = apply_as_list(golden_nps, output_dist)
    else:
        nested_result = list(result_nps)
        nested_golden = list(golden_nps)

    ret = func(*nested_result, *nested_golden)

    if ret is None:
        return  # in-place 模式: 用户用 [:] 修改数组值, 已反映到原 flat list

    # 有返回值模式: unfold 回 flat
    n = len(nested_result)
    if not isinstance(ret, (list, tuple)) or len(ret) != n + len(nested_golden):
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare returned len="
            f"{len(ret) if hasattr(ret, '__len__') else '?'}, "
            f"expected {n + len(nested_golden)} (npu={n} + golden={len(nested_golden)})")

    flat_result = deep_flatten(ret[:n])
    flat_golden = deep_flatten(ret[n:])
    if len(flat_result) != len(result_nps) or len(flat_golden) != len(golden_nps):
        raise ValueError(
            f"[{testcase.testcase_name}] pre_compare unfolded len mismatch: "
            f"npu={len(flat_result)}/{len(result_nps)}, "
            f"golden={len(flat_golden)}/{len(golden_nps)} (check tensor-list nesting)")
    for i in range(len(flat_result)):
        result_nps[i] = flat_result[i]
    for i in range(len(flat_golden)):
        golden_nps[i] = flat_golden[i]


def _try_custom_compare(testcase, result_nps, golden_nps, switches):
    """尝试定制 compare。返回 (precision_str, log_str, is_pass) 或 None。"""
    func = get_spec_attr(testcase.api_name, "compare", switches.plugin_path)
    if func is None:
        return None

    # result_nps / golden_nps 无效 → 走内置
    if not result_nps or not golden_nps or any(isinstance(g, str) for g in golden_nps):
        return None

    # fold(与 pre_compare 一致)
    output_dist = getattr(testcase, "output_dist", None)
    if output_dist:
        nested_result = apply_as_list(result_nps, output_dist)
        nested_golden = apply_as_list(golden_nps, output_dist)
    else:
        nested_result = list(result_nps)
        nested_golden = list(golden_nps)

    ret = func(*nested_result, *nested_golden)

    # 适配 dict → (precision_str, log_str, is_pass)
    if isinstance(ret, dict):
        ret = [ret]

    if not isinstance(ret, (list, tuple)):
        raise ValueError(f"[{testcase.testcase_name}] compare must return dict or list[dict]")

    if not ret:
        raise ValueError(f"[{testcase.testcase_name}] compare returned empty list")

    if any(isinstance(item, (list, tuple)) for item in ret):
        dicts = deep_flatten(ret)
    else:
        dicts = list(ret)

    precisions, passes, logs = [], [], ""
    for i, d in enumerate(dicts):
        if not isinstance(d, dict) or "pass" not in d or "precision" not in d:
            raise ValueError(
                f"[{testcase.testcase_name}] compare output[{i}] missing 'pass' or 'precision'")
        p = d["precision"]
        precisions.append(f"{p}%" if isinstance(p, (int, float)) else str(p))
        passes.append(bool(d["pass"]))
        if d.get("error_info"):
            logs += f"Output {i}: {d['error_info']}\n"

    return ",".join(precisions), logs, all(passes)


def _evaluate_eager_precision(testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct):
    """Compare eager mode results against golden, set return_struct. Returns True if pass."""
    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in result_nps)
    tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
    standards = resolve_tolerance(tolerance, testcase.flat_precision_tolerances,
                                  testcase.flat_absolute_precision, output_dtypes,
                                  switches.compare_method)
    try:
        custom_result = _try_custom_compare(testcase, result_nps, golden_nps, switches)
        if custom_result is not None:
            precision_str, log_str, is_pass = custom_result
            metrics = {}
        else:
            precision_str, log_str, is_pass, metrics = compare(
                result_nps, golden_nps, output_dtypes,
                standards=standards, third_parties=None
            )
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Eager comparison failure")
        return_struct.construct("COMPARE_FAILURE", "FAIL", None, metrics={})
        return False

    if log_str:
        logging.debugc(f"\nComparing eager with golden\n{log_str}")

    precision_status = "PASS" if is_pass else "FAIL"
    if not is_pass and switches.dump_config.dump_on_fail:
        _dump_on_fail(testcase, raw_inputs, result_nps, golden_nps, switches)
    return_struct.construct(precision_str, precision_status, perf, metrics=metrics)
    return True


def _evaluate_graph_precision(testcase, raw_inputs, graph_nps, golden_nps, switches, return_struct, mode, perf=None):
    """Compare graph mode results against golden.

    Args:
        mode: "static" or "dynamic"
        perf: ProfileResult from graph execution (may be None)
    """
    if not graph_nps:
        return_struct.construct("GRAPH_EXEC_FAILURE", "FAIL", None, mode=mode)
        return

    _apply_pre_compare(testcase, graph_nps, golden_nps, switches)

    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in graph_nps)
    tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
    standards = resolve_tolerance(tolerance, testcase.flat_precision_tolerances,
                                  testcase.flat_absolute_precision, output_dtypes,
                                  switches.compare_method)
    log_str, metrics = "", {}                  # try 前初始化（防 except 路径 NameError）
    try:
        custom_result = _try_custom_compare(testcase, graph_nps, golden_nps, switches)
        if custom_result is not None:
            precision_str, log_str, is_pass = custom_result   # metrics 保持 {}
        else:
            precision_str, log_str, is_pass, metrics = compare(
                graph_nps, golden_nps, output_dtypes,
                standards=standards, third_parties=None)
    except Exception:
        logging.exception(f"Graph {mode} comparison failure")
        precision_str, log_str, is_pass = "COMPARE_FAILURE", "", False

    if log_str:
        logging.debugc(f"\nComparing graph {mode} with golden\n{log_str}")

    status = "PASS" if is_pass else "FAIL"
    if not is_pass and switches.dump_config.dump_on_fail:
        _dump_on_fail(testcase, raw_inputs, graph_nps, golden_nps, switches)
    return_struct.construct(precision_str, status, perf, mode=mode, metrics=metrics)


def _do_profile(testcase, backend, device_grant_events, device_granted_indices,
                dev_id, switches, return_struct):
    """Core profiling logic."""
    process_ctx = get_process_context()

    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] Cannot resolve param plan for {testcase.api_name}")
        return

    resolved, is_tensor_method = resolve_api(testcase.api_name)
    is_inplace = resolved.endswith('_') if is_tensor_method else getattr(resolved, '__name__', '').endswith('_')

    process_ctx.notify_status("OnGenInput")
    try:
        raw_inputs = generate_inputs(testcase, switches, backend, plan)
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        return_struct.eager_precision = "INPUT_GEN_FAILURE"
        return_struct.precision_status = "FAIL"
        return

    process_ctx.notify_status("OnDumpInput")
    _dump_inputs(testcase, raw_inputs, switches)

    graph_enabled = (switches.cst_switches.enabled
                 or switches.dyn_switches.enabled
                 or getattr(switches, 'aclgraph_enabled', False))

    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.use_device()
    result_nps = None
    perf = None
    graph_cst_nps = None
    graph_dyn_nps = None
    graph_cst_perf = None
    graph_dyn_perf = None
    graph_aclgraph_nps = None
    graph_aclgraph_perf = None
    with DeviceLock(process_ctx, dev_id, use_device=use_device,
                    grant_event=device_grant_events.get(dev_id),
                    granted_idx=device_granted_indices.get(dev_id)):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, dev_id, switches)

        process_ctx.notify_status("OnEagerProfiling")
        result_nps, perf = _execute_eager(
            testcase, backend, dev_id, switches, plan,
            resolved, is_tensor_method, is_inplace, raw_inputs)
        if graph_enabled:
            if getattr(switches, 'aclgraph_enabled', False):
                process_ctx.notify_status("OnGraphAclgraph")
                graph_aclgraph_nps, graph_aclgraph_perf = _execute_graph(
                    testcase, backend, dev_id, switches, plan,
                    resolved, is_tensor_method, is_inplace, raw_inputs,
                    dynamic=False, is_aclgraph=True)
            if switches.cst_switches.enabled:
                process_ctx.notify_status("OnGraphCst")
                graph_cst_nps, graph_cst_perf = _execute_graph(
                    testcase, backend, dev_id, switches, plan,
                    resolved, is_tensor_method, is_inplace, raw_inputs, dynamic=False)
            if switches.dyn_switches.enabled:
                process_ctx.notify_status("OnGraphDyn")
                graph_dyn_nps, graph_dyn_perf = _execute_graph(
                    testcase, backend, dev_id, switches, plan,
                    resolved, is_tensor_method, is_inplace, raw_inputs, dynamic=True)
    gc.collect()

    golden_nps = _generate_golden_data(testcase, raw_inputs, switches, backend)

    if result_nps is None:
        return_struct.eager_precision = "NO_OUTPUT"
        return_struct.precision_status = "FAIL"
        if not graph_enabled:
            return
    else:
        process_ctx.notify_status("OnDumpOutput")
        _dump_outputs(testcase, result_nps, switches)

        _apply_pre_compare(testcase, result_nps, golden_nps, switches)
        process_ctx.notify_status("OnEagerComparison")
        _evaluate_eager_precision(testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct)

    if graph_enabled:
        process_ctx.notify_status("OnGraphComparison")
        if getattr(switches, 'aclgraph_enabled', False) and graph_aclgraph_nps:
            _dump_outputs(testcase, graph_aclgraph_nps, switches)
        if switches.cst_switches.enabled and graph_cst_nps:
            _dump_outputs(testcase, graph_cst_nps, switches)
        if switches.dyn_switches.enabled and graph_dyn_nps:
            _dump_outputs(testcase, graph_dyn_nps, switches)
        if getattr(switches, 'aclgraph_enabled', False):
            _evaluate_graph_precision(testcase, raw_inputs, graph_aclgraph_nps, golden_nps, switches, return_struct, "aclgraph", graph_aclgraph_perf)
        if switches.cst_switches.enabled:
            _evaluate_graph_precision(testcase, raw_inputs, graph_cst_nps, golden_nps, switches, return_struct, "static", graph_cst_perf)
        if switches.dyn_switches.enabled:
            _evaluate_graph_precision(testcase, raw_inputs, graph_dyn_nps, golden_nps, switches, return_struct, "dynamic", graph_dyn_perf)

    _profiling_end_print(testcase, return_struct, golden_nps, switches)
    del raw_inputs

