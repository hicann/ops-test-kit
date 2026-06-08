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

from ttk.utilities.container_utils import get_global_storage, apply_as_list
from ttk.core_modules.tbe_multiprocessing import get_process_context, DeviceLock
from ttk.core_modules.comparison.comparison import compare
from ttk.core_modules.tbe_logging import default_logging_config
from ttk.utilities import get, dump_to_file, waiting_for_memory

from .backends import get_backend
from .execution import call_api
from .api_resolver import resolve_api
from .golden_generation import generate_golden
from .input_generation import generate_inputs
from .profiler import get_profiler
from .result import FrameworkApiReturnStructure

WARMUP_COUNT = 5
REPEAT_COUNT = 10


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
        f"Backend: {backend.device_name()}\n"
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
    
    kernel_table = ""
    if return_struct.kernel_details:
        try:
            kernels = json.loads(return_struct.kernel_details)
            if kernels:
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
                
                kernel_table = f"\n{separator}\n"
                kernel_table += f"{'Name':<{name_width}}{'avg':<{avg_width}}{'max':<{max_width}}{'min':<{min_width}}{calls_header}\n"
                kernel_table += f"{separator}\n"
                for k in kernels:
                    name = k.get("name", "")
                    if len(name) > name_width:
                        name = name[:name_width-2] + ".."
                    avg = f"{k.get('avg', 0):.3f}"
                    max_val = f"{k.get('max', 0):.3f}"
                    min_val = f"{k.get('min', 0):.3f}"
                    calls = k.get("calls", 0)
                    kernel_table += f"{name:<{name_width}}{avg:<{avg_width}}{max_val:<{max_width}}{min_val:<{min_width}}{calls}\n"
                kernel_table += f"{separator}\n"
        except (json.JSONDecodeError, TypeError):
            kernel_table = f"\nKernel Details: {return_struct.kernel_details}\n"

    perf_hash = "#" * 42
    perf_line = f" {perf_hash} Result Summary  {perf_hash} "
    end_hash = "#" * separator_len

    logging.info(
        f"{kernel_table}"
        f"\n{perf_line}\n"
        f"STATUS: {return_struct.precision_status} \n"
        f"DEVICE: {return_struct.device_perf_us} us\n"
        f"CPU: {return_struct.cpu_perf_us} us\n"
        f"GOLD: {return_struct.precision}\n"
        f"Golden Shapes: {tuple(_print_get_shape(g) if isinstance(g, np.ndarray) else g for g in golden_nps)}\n"
        f"Golden Dtypes: {tuple(str(_print_get_dtype(g)) if isinstance(g, np.ndarray) else g for g in golden_nps)}\n"
        f"{end_hash}"
    )


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
        return_struct.precision = testcase.fail_reason
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
        return_struct.precision = str(e)
        return_struct.precision_status = "FAIL"

    return return_struct


def _get_or_create_backend(switches):
    """Get or create backend, cached per subprocess via process context."""
    process_ctx = get_process_context()
    cached = process_ctx.storage.get("framework_api_backend")
    if cached is not None:
        return cached
    backend = get_backend(switches.backend_name)
    process_ctx.storage["framework_api_backend"] = backend
    return backend


def _build_tol_options(testcase):
    """Build tolerance options dict from testcase, same pattern as OpApi Comparator."""
    if testcase.precision_tolerances is None:
        rtol, ptol = None, None
    else:
        ptols = testcase.flat_precision_tolerances
        rtol = [get(x, 0) for x in ptols]
        ptol = [get(x, 1) for x in ptols]
    return {'rtol': rtol, 'ptol': ptol, 'atol': testcase.flat_absolute_precision}


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


def _result_to_numpy(result, backend, copy=False):
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


def _execute_on_device(testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs):
    """Build device tensors, run API with profiling, return (result_nps, perf) or raises."""
    process_ctx = get_process_context()

    dev_tensors = [backend.to_device(x, dev_id) if x is not None else None for x in raw_inputs]
    if testcase.tensor_formats and backend.device_name() == "npu":
        dev_tensors = apply_format_cast(dev_tensors, testcase.flat_tensor_formats)
    dist = testcase.tensor_list_dist
    if dist:
        nested_tensors = apply_as_list(dev_tensors, dist)
    else:
        nested_tensors = dev_tensors
    args, kwargs, _ = plan.build_args(nested_tensors)

    run_count = switches.run_time if switches.run_time and switches.run_time > 0 else REPEAT_COUNT
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
        result_nps = _result_to_numpy(result, backend, copy=True)
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

    if is_inplace and inplace_backup is not None:
        args[0][:] = inplace_backup
        del inplace_backup

    if not is_inplace:
        result_nps = _result_to_numpy(result, backend)

    del dev_tensors, nested_tensors
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
                backend.device_name())
        except Exception:
            logging.exception(f"[{testcase.testcase_name}] Golden generation failure")
            golden_nps = ["GOLDEN_FAILURE"]
    process_ctx.notify_status("OnDumpGolden")
    _dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def _evaluate_precision(testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct):
    """Compare results against golden, set return_struct. Returns True if pass."""
    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in result_nps)
    tol_options = _build_tol_options(testcase)
    try:
        precision_str, log_str, is_pass = compare(
            result_nps, golden_nps, output_dtypes,
            methods=switches.compare_method, options=tol_options
        )
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Comparison failure")
        return_struct.precision = "COMPARE_FAILURE"
        return_struct.precision_status = "FAIL"
        return False

    if log_str:
        logging.debugc(f"\nComparing {testcase.testcase_name} with golden\n{log_str}")

    precision_status = "PASS" if is_pass else "FAIL"
    if not is_pass and switches.dump_config.dump_on_fail:
        _dump_on_fail(testcase, raw_inputs, result_nps, golden_nps, switches)
    return_struct.construct(precision_str, precision_status, perf)
    _profiling_end_print(testcase, return_struct, golden_nps, switches)
    return True


def _do_profile(testcase, backend, device_grant_events, device_granted_indices,
                dev_id, switches, return_struct):
    """Core profiling logic."""
    process_ctx = get_process_context()

    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.precision = "PARAM_PLAN_FAILURE"
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
        return_struct.precision = "INPUT_GEN_FAILURE"
        return_struct.precision_status = "FAIL"
        return

    process_ctx.notify_status("OnDumpInput")
    _dump_inputs(testcase, raw_inputs, switches)

    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.use_device()
    with DeviceLock(process_ctx, dev_id, use_device=use_device,
                    grant_event=device_grant_events.get(dev_id),
                    granted_idx=device_granted_indices.get(dev_id)):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, dev_id, switches)
        process_ctx.notify_status("OnProfiling")
        result_nps, perf = _execute_on_device(
            testcase, backend, dev_id, switches, plan,
            resolved, is_tensor_method, is_inplace, raw_inputs)
    gc.collect()

    if result_nps is None:
        return_struct.precision = "NO_OUTPUT"
        return_struct.precision_status = "FAIL"
        return

    process_ctx.notify_status("OnDumpOutput")
    _dump_outputs(testcase, result_nps, switches)

    golden_nps = _generate_golden_data(testcase, raw_inputs, switches, backend)

    process_ctx.notify_status("OnComparison")
    _evaluate_precision(testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct)

    del raw_inputs

