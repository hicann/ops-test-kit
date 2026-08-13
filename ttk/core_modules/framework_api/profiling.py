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
import shutil
from pathlib import Path

import numpy as np

from ttk.core_modules.comparison.comparison import compare
from ttk.core_modules.comparison.custom import apply_pre_compare, try_custom_compare
from ttk.core_modules.comparison.resolve import resolve_tolerance
from ttk.core_modules.manual_data import (
    load_manual_data_case,
    manual_data_prepare_roles,
    prepare_manual_data_store,
    snapshot_manual_values,
)
from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
from ttk.core_modules.pre_npu import (
    build_pre_npu_profile_runner,
    build_ttk_context,
    execute_pre_npu,
    refresh_ttk_context,
    resolve_pre_npu,
)
from ttk.core_modules.tbe_logging import build_single_log_dir, default_logging_config
from ttk.core_modules.tbe_multiprocessing import DeviceLock, get_process_context
from ttk.test_spec import get_spec_attr
from ttk.utilities.string_utils import stable_path_component
from ttk.utilities import dump_to_file, waiting_for_memory
from ttk.utilities.container_utils import apply_as_list, get_global_storage

from .api_resolver import resolve_api
from .backends import get_backend
from .eager_execution import call_api
from .golden_generation import generate_golden
from .graph_execution import _execute_graph
from .input_generation import generate_inputs
from .profiler import get_profiler
from .profiling_utils import clone_preserving_stride, prepare_device_args, result_to_numpy
from .result import FrameworkApiReturnStructure

WARMUP_COUNT = 5


def _print_get_shape(arr):
    return arr.shape if hasattr(arr, "shape") else arr


def _print_get_dtype(arr):
    return arr.dtype if hasattr(arr, "dtype") else arr


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
            table += (
                f"{'Name':<{name_width}}{'avg':<{avg_width}}{'max':<{max_width}}{'min':<{min_width}}{calls_header}\n"
            )
            table += f"{separator}\n"
            for k in kernels:
                name = k.get("name", "")
                if len(name) > name_width:
                    name = name[: name_width - 2] + ".."
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
    has_graph = (
        return_struct.graph_cst_precision is not None
        or return_struct.graph_dyn_precision is not None
        or return_struct.graph_aclgraph_precision is not None
    )

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


def _e2e_xpu_inputs(testcase, raw_inputs):
    """raw_inputs (flat numpy) re-nested, pure outputs filtered → top-level input slots."""
    dist = testcase.tensor_list_dist
    nested = apply_as_list(list(raw_inputs), dist) if dist else list(raw_inputs)
    out_indices = set(testcase.output_tensor_indexes or ())
    inputs = []
    real_idx = 0
    for slot in nested:
        if real_idx not in out_indices:
            inputs.append(slot)
        real_idx += len(slot) if isinstance(slot, (list, tuple)) else 1
    return inputs


def _e2e_xpu_input_names(testcase):
    """Input tensor param names (pure outputs filtered) for XPU schema."""
    plan = testcase.get_param_plan()
    if plan is None:
        return []
    out_indices = set(testcase.output_tensor_indexes or ())
    names = []
    real_idx = 0
    for p in plan.overload_params:
        if p.is_tensor_like and p.name != "out":
            if real_idx not in out_indices:
                names.append(p.name)
            real_idx += 1
    return names


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
        _log_dir = build_single_log_dir(switches.test_mode, testcase.api_name, switches.root_path)
        default_logging_config(file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name, log_dir=_log_dir)

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
    _ensure_deterministic_level_e2e(process_ctx, backend, testcase)
    try:
        _do_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct)
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
    dump_to_file(data, dump_path, file_name, file_format=switches.dump_config.file_format)


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


def _ensure_deterministic_level_e2e(process_ctx, backend, testcase):
    """e2e 模式：设置 NPU 确定性计算级别"""
    det_level = getattr(get_global_storage(), "deterministic_level", 0)
    if process_ctx.storage.get("_deterministic_level_set"):
        return
    if backend.is_npu():
        try:
            import torch_npu

            torch_npu.npu.set_deterministic_level(det_level)
            logging.info(
                f"NPU deterministic level set to {det_level} (e2e batch consistency for {testcase.testcase_name})"
            )
        except Exception as e:
            logging.warning(f"Failed to set deterministic level: {e}")
    process_ctx.storage["_deterministic_level_set"] = True


def _execute_eager(testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs):
    """Build device tensors, run API in eager mode with profiling, return (result_nps, perf) or raises."""
    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)

    profiling_enabled = bool(getattr(switches, "TASK_PROFILING", True))
    deterministic = int(getattr(switches, "deterministic_level", 0) or 0) > 0
    if profiling_enabled or deterministic:
        run_count = switches.run_time
    else:
        run_count = 0 if is_inplace else 1
    safe_name = stable_path_component(testcase.testcase_name, "testcase")
    profile_path = os.path.join(
        switches.root_path, "msprof", "framework_api", safe_name, "eager"
    )
    profiler = get_profiler(
        testcase.api_name,
        backend,
        enabled=profiling_enabled,
        result_path=profile_path if profiling_enabled else None,
    )

    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    inplace_input_backups = {}
    if inplace_input_indexes:
        for idx in inplace_input_indexes:
            if idx < len(args) and args[idx] is not None:
                inplace_input_backups[idx] = clone_preserving_stride(args[idx])

    if is_inplace:
        inplace_backup = clone_preserving_stride(args[0]) if args and args[0] is not None else None
        if is_tensor_method:
            if args[0] is not None:
                result = call_api(testcase.api_name, plan.overload_index, getattr(args[0], resolved), args[1:], kwargs)
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

    if switches.warmup and profiling_enabled:
        for _ in range(WARMUP_COUNT):
            if is_inplace and inplace_backup is not None:
                args[0][:] = inplace_backup
            for idx, backup in inplace_input_backups.items():
                args[idx][:] = backup
            if is_tensor_method:
                getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
            else:
                resolved(*args, **kwargs)
        backend.synchronize(dev_id)

    for idx, backup in inplace_input_backups.items():
        args[idx][:] = backup
    if is_inplace and inplace_backup is not None:
        args[0][:] = inplace_backup

    inplace_clones = {}
    original_tensors = {}
    for idx in inplace_input_backups:
        original_tensors[idx] = args[idx]
        inplace_clones[idx] = [clone_preserving_stride(args[idx]) for _ in range(run_count - 1)]
    if is_inplace and inplace_backup is not None and 0 not in original_tensors:
        if args and args[0] is not None:
            original_tensors[0] = args[0]
            inplace_clones[0] = [clone_preserving_stride(args[0]) for _ in range(run_count - 1)]

    with profiler:
        for i in range(run_count):
            if i < run_count - 1:
                for idx in inplace_clones:
                    args[idx] = inplace_clones[idx][i]
            else:
                for idx in original_tensors:
                    args[idx] = original_tensors[idx]
            if is_tensor_method:
                r = getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
            else:
                r = resolved(*args, **kwargs)
            if not is_inplace:
                result = r
        backend.synchronize(dev_id)

    perf = profiler.result(backend, max(run_count, 1))

    if not is_inplace:
        result_nps = result_to_numpy(result, backend)

    if inplace_input_indexes:
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                inplace_np = backend.to_numpy(args[idx].detach().clone())
                result_nps.append(inplace_np)

    del args, kwargs
    return result_nps, perf


def _generate_golden_data(
    testcase, raw_inputs, switches, backend, dump=True, ttk_context=None
):
    """Generate golden outputs and dump them. Returns golden_nps list."""
    process_ctx = get_process_context()
    process_ctx.notify_status("OnGenGolden")
    if switches.golden_mode == "Disable" or str(testcase.golden_api).lower() == "disable":
        golden_nps = ["SUPPRESSED"]
    else:
        try:
            golden_nps = generate_golden(
                testcase,
                raw_inputs,
                switches.plugin_path,
                switches,
                backend.alias(),
                ttk_context=ttk_context,
            )
        except Exception:
            logging.exception(f"[{testcase.testcase_name}] Golden generation failure")
            golden_nps = ["GOLDEN_FAILURE"]
    if dump:
        process_ctx.notify_status("OnDumpGolden")
        _dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def _apply_pre_compare(testcase, result_nps, golden_nps, switches, ttk_context=None):
    """加载并调用 pre_compare, 变换 result_nps 和 golden_nps。
    无 spec / golden 无效时什么都不做。异常向上抛。"""
    func = get_spec_attr(testcase.api_name, "pre_compare", switches.plugin_path)
    apply_pre_compare(
        testcase, result_nps, golden_nps, func, ttk_context=ttk_context
    )


def _try_custom_compare(testcase, result_nps, golden_nps, switches, ttk_context=None):
    """尝试定制 compare。返回 (precision_str, log_str, is_pass) 或 None。"""
    func = get_spec_attr(testcase.api_name, "compare", switches.plugin_path)
    return try_custom_compare(
        testcase, result_nps, golden_nps, func, ttk_context=ttk_context
    )


def _evaluate_eager_precision(
    testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct,
    third_parties=None, ttk_context=None,
):
    """Compare eager mode results against golden, set return_struct. Returns True if pass."""
    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in result_nps)
    tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    try:
        custom_result = _try_custom_compare(
            testcase, result_nps, golden_nps, switches, ttk_context=ttk_context
        )
        if custom_result is not None:
            precision_str, log_str, is_pass = custom_result
            metrics = {}
        else:
            precision_str, log_str, is_pass, metrics = compare(
                result_nps, golden_nps, output_dtypes, standards=standards, third_parties=third_parties
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


def _evaluate_graph_precision(
    testcase, raw_inputs, graph_nps, golden_nps, switches, return_struct, mode,
    perf=None, third_parties=None, ttk_context=None,
):
    """Compare graph mode results against golden.

    Args:
        mode: "static" or "dynamic"
        perf: ProfileResult from graph execution (may be None)
    """
    if not graph_nps:
        return_struct.construct("GRAPH_EXEC_FAILURE", "FAIL", None, mode=mode)
        return

    _apply_pre_compare(
        testcase, graph_nps, golden_nps, switches, ttk_context=ttk_context
    )

    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in graph_nps)
    tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    log_str, metrics = "", {}  # try 前初始化（防 except 路径 NameError）
    try:
        custom_result = _try_custom_compare(
            testcase, graph_nps, golden_nps, switches, ttk_context=ttk_context
        )
        if custom_result is not None:
            precision_str, log_str, is_pass = custom_result  # metrics 保持 {}
        else:
            precision_str, log_str, is_pass, metrics = compare(
                graph_nps, golden_nps, output_dtypes, standards=standards, third_parties=third_parties
            )
    except Exception:
        logging.exception(f"Graph {mode} comparison failure")
        precision_str, log_str, is_pass = "COMPARE_FAILURE", "", False

    if log_str:
        logging.debugc(f"\nComparing graph {mode} with golden\n{log_str}")

    status = "PASS" if is_pass else "FAIL"
    if not is_pass and switches.dump_config.dump_on_fail:
        _dump_on_fail(testcase, raw_inputs, graph_nps, golden_nps, switches)
    return_struct.construct(precision_str, status, perf, mode=mode, metrics=metrics)


def _collect_sim_report(testcase, switches):
    """Move the camodel ``instr.bin`` (written to the worker cwd) into the case
    sim_output dir, and when ``--sim-report`` generate the trace report.

    The E2E npusim backend runs torch_npu directly in the forkserver worker
    (no cannsim record), so camodel writes ``instr.bin`` to the worker cwd
    (``switches.root_path``). Collect it per case before the next case
    overwrites it; report generation is best-effort and never affects
    precision results.
    """
    if getattr(switches, "backend", None) != "npusim":
        return
    from ttk.core_modules.simulator.case_writer import case_dir

    src = Path(switches.root_path) / "instr.bin"
    if not src.is_file() or src.stat().st_size == 0:
        logging.warning("[%s] no instr.bin in worker cwd; skip sim report",
                        testcase.testcase_name)
        return
    case_path = case_dir(switches, testcase.testcase_name)
    case_path.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(case_path / "instr.bin"))
    except OSError as e:
        # Without this warning the move failure is silent: instr.bin stays in the
        # worker cwd, gets overwritten by the next case, and the user is left
        # without a sim report and without any hint.
        logging.warning("[%s] failed to move instr.bin into %s: %s",
                        testcase.testcase_name, case_path, e)
        return
    if getattr(switches, "sim_report", False):
        from ttk.core_modules.simulator.report import maybe_generate_sim_report

        maybe_generate_sim_report(switches, case_path, case_path)


def _do_profile(testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct):
    """Core profiling logic."""
    process_ctx = get_process_context()
    return_struct.batch_consistency_id = getattr(testcase, "batch_consistency_id", None)
    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] Cannot resolve param plan for {testcase.api_name}")
        return

    manual_mode = getattr(switches, "manual_data_mode", None)
    _, prepare_goldens = manual_data_prepare_roles(switches)
    manual_case = None
    try:
        prepare_store = prepare_manual_data_store(testcase, "e2e", switches)
    except Exception as exc:
        logging.exception(f"[{testcase.testcase_name}] Manual data preparation failure")
        return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
        _profiling_end_print(testcase, return_struct, switches=switches)
        return
    if manual_mode != "prepare":
        try:
            manual_case = load_manual_data_case(
                testcase,
                "e2e",
                switches,
                before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
                require_goldens=False,
            )
            if manual_case is not None:
                manual_mode = "replay"
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data loading failure")
            return_struct.eager_precision = f"MANUAL_DATA_READ_FAILURE: {exc}"
            return_struct.precision_status = "FAIL"
            return

    ttk_context = build_ttk_context(
        testcase, switches, "e2e", manual_case=manual_case
    )
    process_ctx.notify_status("OnGenInput")
    try:
        if manual_case is not None:
            raw_inputs = generate_inputs(
                testcase,
                switches,
                backend,
                plan,
                stored_inputs=manual_case.inputs,
                ttk_context=ttk_context,
            )
        else:
            raw_inputs = generate_inputs(
                testcase,
                switches,
                backend,
                plan,
                ttk_context=ttk_context,
            )
    except Exception as exc:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        prefix = "MANUAL_DATA_READ_FAILURE" if manual_case is not None else "INPUT_GEN_FAILURE"
        return_struct.eager_precision = f"{prefix}: {exc}" if manual_case is not None else prefix
        return_struct.precision_status = "FAIL"
        return

    if manual_mode != "prepare":
        process_ctx.notify_status("OnDumpInput")
        _dump_inputs(testcase, raw_inputs, switches)

    if manual_mode == "prepare":
        golden_nps = None
        try:
            prepared_inputs = snapshot_manual_values(
                testcase.np_storages if testcase.np_storages is not None else raw_inputs,
                "input",
            )
            if prepare_goldens:
                golden_nps = _generate_golden_data(
                    testcase,
                    raw_inputs,
                    switches,
                    backend,
                    dump=False,
                    ttk_context=ttk_context,
                )
            process_ctx.notify_status("OnWriteManualData")
            case_dir = prepare_store.write_case(
                testcase,
                "e2e",
                prepared_inputs,
                golden_nps if golden_nps is not None else (),
                file_format=switches.dump_config.file_format,
                write_goldens=prepare_goldens,
            )
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data preparation failure")
            return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return
        logging.info(f"[{testcase.testcase_name}] manual data prepared: {case_dir}")
        return_struct.construct("MANUAL_DATA_PREPARED", "PASS", None)
        _profiling_end_print(testcase, return_struct, golden_nps or (), switches)
        return

    resolved, is_tensor_method = resolve_api(testcase.api_name)
    is_inplace = resolved.endswith("_") if is_tensor_method else getattr(resolved, "__name__", "").endswith("_")

    graph_enabled = (
        switches.cst_switches.enabled or switches.dyn_switches.enabled or getattr(switches, "aclgraph_enabled", False)
    )

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
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        pre_npu_func = resolve_pre_npu(testcase, switches)
        if pre_npu_func is not None:
            process_ctx.notify_status("OnPreNpu")
            refresh_ttk_context(ttk_context, testcase)
            try:
                pre_npu_result = execute_pre_npu(
                    testcase,
                    switches,
                    ttk_context,
                    profile_runner=build_pre_npu_profile_runner(
                        testcase,
                        switches,
                        "e2e",
                        synchronize=lambda: backend.synchronize(dev_id),
                    ),
                    pre_npu_func=pre_npu_func,
                )
            except Exception as exc:
                logging.exception(f"[{testcase.testcase_name}] pre-NPU stage failure")
                return_struct.construct(f"PRE_NPU_FAILURE: {exc}", "FAIL", None)
                _profiling_end_print(testcase, return_struct, switches=switches)
                return
            if pre_npu_result.stop:
                detail = (
                    f": {pre_npu_result.reason}" if pre_npu_result.reason else ""
                )
                return_struct.construct(f"PRE_NPU_STOPPED{detail}", "PASS", None)
                _profiling_end_print(testcase, return_struct, switches=switches)
                return

        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, dev_id, switches)

        process_ctx.notify_status("OnEagerProfiling")
        if not getattr(switches, "aclgraph_enabled", False):
            result_nps, perf = _execute_eager(
                testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs
            )
        if graph_enabled:
            if getattr(switches, "aclgraph_enabled", False):
                process_ctx.notify_status("OnGraphAclgraph")
                graph_aclgraph_nps, graph_aclgraph_perf = _execute_graph(
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    resolved,
                    is_tensor_method,
                    is_inplace,
                    raw_inputs,
                    dynamic=False,
                    is_aclgraph=True,
                )
            if switches.cst_switches.enabled:
                process_ctx.notify_status("OnGraphCst")
                graph_cst_nps, graph_cst_perf = _execute_graph(
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    resolved,
                    is_tensor_method,
                    is_inplace,
                    raw_inputs,
                    dynamic=False,
                )
            if switches.dyn_switches.enabled:
                process_ctx.notify_status("OnGraphDyn")
                graph_dyn_nps, graph_dyn_perf = _execute_graph(
                    testcase,
                    backend,
                    dev_id,
                    switches,
                    plan,
                    resolved,
                    is_tensor_method,
                    is_inplace,
                    raw_inputs,
                    dynamic=True,
                )
    gc.collect()
    _collect_sim_report(testcase, switches)

    if manual_case is not None and manual_case.has_goldens:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            reference_outputs = result_nps
            if reference_outputs is None:
                reference_outputs = graph_cst_nps
            if reference_outputs is None:
                reference_outputs = graph_dyn_nps
            if reference_outputs is None:
                reference_outputs = graph_aclgraph_nps
            golden_nps = manual_case.load_goldens(references=reference_outputs)
            _dump_goldens(testcase, golden_nps, switches)
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual golden loading failure")
            return_struct.construct(f"MANUAL_DATA_READ_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return
    else:
        golden_nps = _generate_golden_data(
            testcase,
            raw_inputs,
            switches,
            backend,
            ttk_context=ttk_context,
        )

    # E2E does NOT set golden_mode_override (unlike kernel/aclnn): E2E golden
    # runs the same torch API on CPU, where bfloat16 is computed natively and
    # reliably. kernel/aclnn need Promote because numpy/handwritten goldens
    # cannot compute bfloat16 accurately. cross_check's ratio-based metric is
    # also tolerant to golden's own bf16 rounding (shared in numerator/denominator).
    third_parties = None
    xpu_results = None
    if golden_nps and not any(isinstance(g, str) for g in golden_nps):
        tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
        output_dtypes = tuple(str(g.dtype) if g is not None and hasattr(g, "dtype") else None for g in golden_nps)
        standards = resolve_tolerance(
            tolerance,
            testcase.flat_precision_tolerances,
            testcase.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
        )
        need_3party = any(s.token == "cross_check" for s in standards)
        from ttk.remote.client import collect_third_party, xpu_mode_of

        xpu_mode = xpu_mode_of(switches, need_3party)
        if xpu_mode:
            process_ctx.notify_status("OnXpuProfiling")
            _, third_parties, xpu_results = collect_third_party(
                op_name=testcase.api_name,
                inputs=_e2e_xpu_inputs(testcase, raw_inputs),
                input_names=_e2e_xpu_input_names(testcase),
                op_type=None,
                attributes=testcase.attributes or {},
                testcase_name=testcase.testcase_name,
                switches=switches,
                need_data=need_3party,
            )
            if need_3party and third_parties is None:
                logging.warning(
                    "[%s] cross_check configured but no third_party output "
                    "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                    testcase.testcase_name,
                )
    return_struct.xpu_metrics = _format_xpu_metrics(xpu_results) if xpu_results else {}

    if result_nps is None:
        return_struct.eager_precision = "NO_OUTPUT"
        if not graph_enabled:
            _profiling_end_print(testcase, return_struct, golden_nps, switches)
            return
    else:
        process_ctx.notify_status("OnDumpOutput")
        _dump_outputs(testcase, result_nps, switches)

        _apply_pre_compare(
            testcase, result_nps, golden_nps, switches, ttk_context=ttk_context
        )
        process_ctx.notify_status("OnEagerComparison")
        _evaluate_eager_precision(
            testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct,
            third_parties, ttk_context=ttk_context
        )

    if graph_enabled:
        process_ctx.notify_status("OnGraphComparison")
        if getattr(switches, "aclgraph_enabled", False) and graph_aclgraph_nps:
            _dump_outputs(testcase, graph_aclgraph_nps, switches)
        if switches.cst_switches.enabled and graph_cst_nps:
            _dump_outputs(testcase, graph_cst_nps, switches)
        if switches.dyn_switches.enabled and graph_dyn_nps:
            _dump_outputs(testcase, graph_dyn_nps, switches)
        if getattr(switches, "aclgraph_enabled", False):
            _evaluate_graph_precision(
                testcase,
                raw_inputs,
                graph_aclgraph_nps,
                golden_nps,
                switches,
                return_struct,
                "aclgraph",
                graph_aclgraph_perf,
                third_parties,
                ttk_context,
            )
        if switches.cst_switches.enabled:
            _evaluate_graph_precision(
                testcase,
                raw_inputs,
                graph_cst_nps,
                golden_nps,
                switches,
                return_struct,
                "static",
                graph_cst_perf,
                third_parties,
                ttk_context,
            )
        if switches.dyn_switches.enabled:
            _evaluate_graph_precision(
                testcase,
                raw_inputs,
                graph_dyn_nps,
                golden_nps,
                switches,
                return_struct,
                "dynamic",
                graph_dyn_perf,
                third_parties,
                ttk_context,
            )

    _profiling_end_print(testcase, return_struct, golden_nps, switches)
    del raw_inputs
