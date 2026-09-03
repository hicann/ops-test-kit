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

import contextlib
import gc
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from ttk.core_modules.comparison.comparison import compare
from ttk.core_modules.comparison.custom import apply_pre_compare, try_custom_compare
from ttk.core_modules.comparison.resolve import resolve_tolerance
from ttk.core_modules.manual_data import (
    load_manual_data_case,
    prepare_manual_data_store,
    snapshot_manual_values,
)
from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
from ttk.core_modules.npu_preprocess import invoke_npu_preprocess
from ttk.core_modules.tbe_logging import build_single_log_dir, default_logging_config
from ttk.core_modules.tbe_multiprocessing import DeviceLock, MultiDeviceLock, get_process_context
from ttk.test_spec import get_spec_attr
from ttk.utilities import dump_to_file, waiting_for_memory
from ttk.utilities.container_utils import apply_as_list, get_global_storage
from ttk.utilities.dtypes import resolve_custom_numpy_dtypes

from .api_resolver import resolve_api
from .backends import get_backend
from .input_generation import generate_inputs
from .profiler import ProfilerConfig, get_profiler
from .profiling_utils import compute_output_md5, finalize_det_status, prepare_device_args, unpack_4bit_outputs
from .result import FrameworkApiReturnStructure

WARMUP_COUNT = 1


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
        f"Backend: {backend.device_type()}\n"
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

    if return_struct.deterministic_status is not None:
        lines.append(f"DETERMINISTIC: {return_struct.deterministic_status}")

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


def profile_process(
    testcase, device_grant_events, device_granted_indices, dev_id, is_multi_device=False, device_ids=None
):
    """
    Framework API profiling process — executed in subprocess.

    Args:
        testcase: TestcaseE2e
        device_grant_events: dict of device_id → Manager().Event()
        device_granted_indices: dict of device_id → Manager().Value('i', -1)
        dev_id: device ID (passed as kwarg by ProcessGroup)
        is_multi_device: whether this is a multi-device task
        device_ids: list of device IDs for multi-device tasks

    Returns:
        FrameworkApiReturnStructure
    """
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(testcase.testcase_name)

    if switches.single_testcase_log_mode:
        _log_dir = build_single_log_dir(switches.test_mode, testcase.api_name, switches.root_path)
        default_logging_config(
            file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name, log_dir=_log_dir
        )

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
        if is_multi_device and device_ids:
            testcase.device_ids = tuple(device_ids)
            _do_profile_multi_device(
                testcase, backend, device_grant_events, device_granted_indices, device_ids, switches, return_struct
            )
        else:
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
    framework = getattr(switches, "framework", "torch")
    backend = get_backend(switches.force_cpu, framework=framework)
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
            backend.set_deterministic_level(det_level)
            logging.info(f"NPU deterministic level set (e2e batch consistency for {testcase.testcase_name})")
        except Exception as e:
            logging.warning(f"Failed to set deterministic level: {e}")
    process_ctx.storage["_deterministic_level_set"] = True


_npu_memory_hint_shown = False


def _capture_stdout_npu_memory(testcase, return_struct, fn, *args, **kwargs):
    """Redirect fd 1 to a temp file during *fn*, extract workspace size from captured stdout.

    Requires the user to set ``ASCEND_SLOG_PRINT_TO_STDOUT=1`` and
    ``ASCEND_GLOBAL_LOG_LEVEL=1`` before running TTK; otherwise no CANN logs
    reach stdout and ``npu_memory`` stays ``None``.

    CANN logs workspace size as ``Workspace addr: 0x..., size: 16777728`` in
    op_executor.cpp. Some versions may use ``workspaceSize: xxx``.
    """
    global _npu_memory_hint_shown
    # 捕获仅在日志走 stdout 且级别覆盖 INFO(workspace) 时才可能有结果，否则零开销直通
    if os.environ.get("ASCEND_SLOG_PRINT_TO_STDOUT") != "1" or os.environ.get("ASCEND_GLOBAL_LOG_LEVEL", "3") not in (
        "0",
        "1",
    ):
        if not _npu_memory_hint_shown:
            _npu_memory_hint_shown = True
            logging.info(
                "[E2E] npu_memory not captured: to enable, export "
                "ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=1 before running TTK"
            )
        return fn(*args, **kwargs)
    sys.stdout.flush()
    cap_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", suffix=".stdout", delete=False) as cap:
            cap_name = cap.name
            orig_stdout = os.dup(1)
            try:
                os.dup2(cap.fileno(), 1)
                result = fn(*args, **kwargs)
            finally:
                sys.stdout.flush()
                os.dup2(orig_stdout, 1)
                os.close(orig_stdout)

        with open(cap_name, "rb") as f:
            content = f.read().decode("utf-8", errors="replace")

        for line in content.splitlines():
            if "Workspace addr" not in line and "workspaceSize" not in line:
                continue
            logging.info("[E2E/%s] %s", testcase.testcase_name, line.strip())
            m = re.search(r"(?:Workspace addr:.*?size:|workspaceSize:)\s*(\d+)", line)
            if m:
                return_struct.npu_memory = int(m.group(1))
        return result
    finally:
        if cap_name:
            with contextlib.suppress(OSError):
                os.unlink(cap_name)


def _execute_eager(testcase, backend, dev_id, switches, plan, resolved, is_tensor_method, is_inplace, raw_inputs):
    """Build device tensors, run API in eager mode with profiling, return (result_nps, perf, det_status)."""
    backend.set_device(dev_id)
    resolved = backend.wrap_eager_callable(resolved)
    args, kwargs = prepare_device_args(testcase, backend, dev_id, plan, raw_inputs)

    if backend.is_npu():
        invoke_npu_preprocess(
            testcase,
            switches,
            plan,
            args,
            kwargs,
            device_scope=lambda: backend.device_scope(dev_id),
        )

    profiling_enabled = bool(getattr(switches, "TASK_PROFILING", True))
    deterministic = int(getattr(switches, "deterministic_level", 0) or 0) > 0
    run_count = switches.run_time
    warmup_count = WARMUP_COUNT if (switches.warmup and profiling_enabled) else 0
    profiler = get_profiler(
        testcase.api_name,
        backend,
        ProfilerConfig(
            testcase_name=testcase.testcase_name,
            root_path=switches.root_path,
            dev_id=dev_id,
            enabled=profiling_enabled,
            warmup_count=warmup_count,
        ),
    )

    inplace_input_indexes = getattr(testcase, "inplace_input_indexes", None) or ()
    inplace_idxs = set()
    if is_inplace and args and args[0] is not None:
        inplace_idxs.add(0)
    inplace_idxs.update(i for i in inplace_input_indexes if i < len(args) and args[i] is not None)

    backups = {}
    for idx in inplace_idxs:
        backups[idx] = backend.clone(args[idx])

    total = warmup_count + run_count
    clones = {}
    for idx, backup in backups.items():
        clones[idx] = [backend.clone(backup) for _ in range(total)]

    result = None
    md5_list = []
    with profiler:
        for i in range(total):
            is_warmup = i < warmup_count
            for idx in clones:
                args[idx] = clones[idx][i]
            with backend.device_scope(dev_id):
                if is_tensor_method:
                    r = getattr(args[0], resolved)(*args[1:], **kwargs) if args[0] is not None else None
                else:
                    r = resolved(*args, **kwargs)
                if is_warmup:
                    profiler.step()
                    continue
                if not is_inplace:
                    result = r
                if deterministic:
                    backend.synchronize(dev_id)
                    run_nps = []
                    if r is not None:
                        run_nps.extend(backend.result_to_numpy(r))
                    if inplace_input_indexes:
                        for idx in sorted(inplace_input_indexes):
                            if idx < len(args) and args[idx] is not None:
                                run_nps.append(backend.to_numpy(args[idx], safe=True))
                    md5_list.append(compute_output_md5(run_nps))
        backend.synchronize(dev_id)

    perf = profiler.result(backend, run_count)

    if not is_inplace:
        result_nps = backend.result_to_numpy(result)
    else:
        result_nps = backend.result_to_numpy(r, copy=True) if r is not None else []

    if inplace_input_indexes:
        if result_nps is None:
            result_nps = []
        for idx in sorted(inplace_input_indexes):
            if idx < len(args) and args[idx] is not None:
                inplace_np = backend.to_numpy(args[idx], safe=True)
                result_nps.append(inplace_np)

    det_status = finalize_det_status(md5_list, testcase.testcase_name)

    del args, kwargs
    result_nps = unpack_4bit_outputs(testcase, result_nps)
    return result_nps, perf, det_status


def _needs_golden_promote(testcase, switches, ref_nps):
    """判据是否为 cross_check —— 是则 golden 必须抬成高精度真值。

    cross_check 时 golden 必须走 Promote,与 geir/profiling.py、npu/op/profiling.py 对齐。
    此前 E2E 不设该 override,理由是「比值判据对 golden 自身舍入不敏感(分子分母共享)」——
    该前提只在竞品是独立实现时成立;而 cross_check 的竞品与 E2E golden 同为 torch aten,
    rmse_party/rel_party 恒为 0,safe_div 的 err 地板接管,判据退化成「与 torch 逐位一致」,
    一次末位舍入即被放大成数倍比值。故 resolve 提前到 golden 生成之前,仅为取得该标志。
    """
    if not ref_nps:
        return False
    try:
        dtypes = tuple(str(r.dtype) if r is not None and hasattr(r, "dtype") else None for r in ref_nps)
        standards = resolve_tolerance(
            get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path),
            testcase.flat_precision_tolerances,
            testcase.flat_absolute_precision,
            dtypes,
            switches.compare_method,
        )
        return any(s.token == "cross_check" for s in standards)  # noqa: S105
    except (KeyError, TypeError, ValueError):
        # 只兜 tolerance spec 配置类错误(字段缺失/类型不对/取值非法)。
        # 用 warning 而非 debug:此处静默跳过 Promote 正是本函数要修的症状
        # (cross_check 误判 FAIL 却无任何提示),必须让用户看得见。
        logging.warning(
            "[%s] tolerance spec resolve failed, golden Promote skipped; "
            "cross_check may misjudge without high-precision golden",
            testcase.testcase_name,
            exc_info=True,
        )
        return False


def _generate_golden_maybe_promote(testcase, raw_inputs, switches, backend, ref_nps):
    """cross_check 判据下临时挂上 golden_mode_override=Promote 再生成 golden。

    不用 del 还原(TestcaseE2e 不支持删除该属性),改为记录原值后回写。
    """
    if not _needs_golden_promote(testcase, switches, ref_nps):
        return _generate_golden_data(testcase, raw_inputs, switches, backend)

    prev = getattr(testcase, "golden_mode_override", None)
    restore = False
    try:
        testcase.golden_mode_override = "Promote"
        restore = True
    except AttributeError:
        # golden_mode_override 已在 TestcaseE2e.__slots__ 中声明,正常不会走到这里;
        # 一旦走到说明 Promote 静默空转,用 warning 让它可见。
        logging.warning("[%s] cannot set golden_mode_override, golden Promote skipped", testcase.testcase_name)
    try:
        return _generate_golden_data(testcase, raw_inputs, switches, backend)
    finally:
        if restore:
            try:
                testcase.golden_mode_override = prev
            except AttributeError:
                logging.warning("[%s] restore golden_mode_override failed", testcase.testcase_name, exc_info=True)


def _generate_golden_data(testcase, raw_inputs, switches, backend, dump=True):
    """Generate golden outputs and dump them. Returns golden_nps list."""
    process_ctx = get_process_context()
    process_ctx.notify_status("OnGenGolden")
    if switches.golden_mode == "Disable" or str(testcase.golden_api).lower() == "disable":
        golden_nps = ["SUPPRESSED"]
    else:
        try:
            from .golden_generation import generate_golden

            golden_nps = generate_golden(testcase, raw_inputs, switches.plugin_path, switches, backend.device_type())
        except Exception:
            logging.exception(f"[{testcase.testcase_name}] Golden generation failure")
            golden_nps = ["GOLDEN_FAILURE"]
    if dump:
        process_ctx.notify_status("OnDumpGolden")
        _dump_goldens(testcase, golden_nps, switches)
    return golden_nps


def _apply_pre_compare(testcase, result_nps, golden_nps, switches):
    """加载并调用 pre_compare, 变换 result_nps 和 golden_nps。
    无 spec / golden 无效时什么都不做。异常向上抛。"""
    func = get_spec_attr(testcase.api_name, "pre_compare", switches.plugin_path)
    apply_pre_compare(testcase, result_nps, golden_nps, func)


def _try_custom_compare(testcase, result_nps, golden_nps, switches):
    """尝试定制 compare。返回 (precision_str, log_str, is_pass) 或 None。"""
    func = get_spec_attr(testcase.api_name, "compare", switches.plugin_path)
    return try_custom_compare(testcase, result_nps, golden_nps, func)


def _evaluate_eager_precision(
    testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct, third_parties=None
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
        custom_result = _try_custom_compare(testcase, result_nps, golden_nps, switches)
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
    testcase, raw_inputs, graph_nps, golden_nps, switches, return_struct, mode, perf=None, third_parties=None
):
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
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    log_str, metrics = "", {}  # try 前初始化（防 except 路径 NameError）
    try:
        custom_result = _try_custom_compare(testcase, graph_nps, golden_nps, switches)
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
        logging.warning("[%s] no instr.bin in worker cwd; skip sim report", testcase.testcase_name)
        return
    case_path = case_dir(switches, testcase.testcase_name)
    case_path.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(case_path / "instr.bin"))
    except OSError as e:
        # Without this warning the move failure is silent: instr.bin stays in the
        # worker cwd, gets overwritten by the next case, and the user is left
        # without a sim report and without any hint.
        logging.warning("[%s] failed to move instr.bin into %s: %s", testcase.testcase_name, case_path, e)
        return
    if getattr(switches, "sim_report", False):
        from ttk.core_modules.simulator.report import maybe_generate_sim_report

        maybe_generate_sim_report(switches, case_path, case_path)


def _do_profile_multi_device(  # noqa: PLR0911
    testcase, backend, device_grant_events, device_granted_indices, device_ids, switches, return_struct
):
    """Multi-device profiling for collective communication operators.

    Uses threading + torch.distributed for multi-card execution.
    Each thread gets its own copy of testcase with rank-specific data.
    """
    import subprocess
    import tempfile

    ndev = len(device_ids)
    process_ctx = get_process_context()

    plan = testcase.get_param_plan()
    if plan is None:
        return_struct.eager_precision = "PARAM_PLAN_FAILURE"
        return_struct.precision_status = "FAIL"
        logging.error(f"[{testcase.testcase_name}] Cannot resolve param plan for {testcase.api_name}")
        return

    resolved, is_tensor_method = None, False
    with contextlib.suppress(ValueError):
        resolved, is_tensor_method = resolve_api(testcase.api_name)

    process_ctx.notify_status("OnGenInput")
    try:
        raw_inputs = generate_inputs(testcase, switches, backend, plan)
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Input generation failure")
        return_struct.eager_precision = "INPUT_GEN_FAILURE"
        return_struct.precision_status = "FAIL"
        return

    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.use_device()
    with MultiDeviceLock(
        process_ctx,
        device_ids,
        use_device=use_device,
        grant_events=device_grant_events,
        granted_indices=device_granted_indices,
    ):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, device_ids[0], switches)

        process_ctx.notify_status("OnProfiling")

        tmp_dir = tempfile.mkdtemp(prefix="ttk_e2e_md_")
        input_path = os.path.join(tmp_dir, "inputs.npz")
        api_name = testcase.api_name

        save_arrays = {}
        # Generate per-rank inputs with rank-specific seed (matching ACLNN multi-device behavior)
        base_seed = switches.random_seed or 0
        import numpy as _np

        # Determine if this op needs shared weight (x1/x2 same across ranks)
        api_name_str = api_name or ""
        weight_shared_ops = (
            "MatmulReduceScatter",
            "AllGatherMatmul",
            "MatmulAllReduce",
            "MatmulAlltoAll",
            "AlltoAllMatmul",
            "MatmulReduceScatterV2",
            "QuantMatmulAllReduce",
            "mm_reduce_scatter",
            "all_gather_mm",
            "mm_all_reduce",
            "mm_all_to_all",
            "all_to_all_mm",
            "npu_moe_distribute_combine",
        )
        exclude_ops = (
            "GroupedMatMul",
            "AlltoAllvGrouped",
            "BatchMatMul",
            "bmm_reduce_scatter",
            "bmm_reducescatter",
            "all_to_all_all_gather",
            "alltoall_allgather",
        )
        needs_shared_weight = not any(kw in api_name_str for kw in exclude_ops) and any(
            kw in api_name_str for kw in weight_shared_ops
        )
        # MoE combine: expand_x (idx 0) and expert_ids (idx 1) must be shared across ranks
        # so that ep_send_counts are consistent for HCCL alltoallv.
        moe_combine_shared_all = "npu_moe_distribute_combine" in api_name_str

        for rank_idx in range(ndev):
            per_rank_seed = base_seed + rank_idx * 1000
            _np.random.seed(per_rank_seed)
            rank_inputs = generate_inputs(testcase, switches, backend, plan)
            # For MC2 matmul ops, regenerate x2 (weight) with rank-independent seed so
            # all ranks share the same weight (matching ACLNN multi-device behavior).
            # x1 stays per-rank different (each rank has its own input chunk).
            if moe_combine_shared_all and len(rank_inputs) > 0:
                # Share all inputs across ranks for moe combine (idx 0 and 1)
                for t_idx in range(min(2, len(rank_inputs))):
                    if rank_inputs[t_idx] is not None:
                        flat_dtypes = resolve_custom_numpy_dtypes(testcase.flat_tensor_dtypes)
                        ss = testcase.flat_storage_shape(t_idx)
                        dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                        ranges = testcase.flat_input_data_ranges or ()
                        data_range = ranges[t_idx] if t_idx < len(ranges) else (None, None)
                        low = data_range[0] if data_range and data_range[0] is not None else -1.0
                        high = data_range[1] if data_range and data_range[1] is not None else 1.0
                        rng = _np.random.RandomState(base_seed + t_idx)
                        if "int" in str(dtype):
                            np_arr = rng.randint(int(low), int(high) + 1, ss).astype(dtype, copy=False)
                        else:
                            np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                        rank_inputs[t_idx] = np_arr.reshape(ss)
            elif needs_shared_weight and len(rank_inputs) > 1 and rank_inputs[1] is not None:
                flat_dtypes = resolve_custom_numpy_dtypes(testcase.flat_tensor_dtypes)
                t_idx = 1
                if t_idx < len(rank_inputs) and rank_inputs[t_idx] is not None:
                    ss = testcase.flat_storage_shape(t_idx)
                    dtype = flat_dtypes[t_idx] if t_idx < len(flat_dtypes) else None
                    ranges = testcase.flat_input_data_ranges or ()
                    data_range = ranges[t_idx] if t_idx < len(ranges) else (None, None)
                    low = data_range[0] if data_range and data_range[0] is not None else -1.0
                    high = data_range[1] if data_range and data_range[1] is not None else 1.0
                    rng = _np.random.RandomState(base_seed + t_idx)
                    np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                    rank_inputs[t_idx] = np_arr.reshape(ss)
            # MoE combine: generate consistent ep_send_counts, expand_idx, expert_scales
            # from expert_ids (which is shared across ranks).
            if moe_combine_shared_all and len(rank_inputs) > 1:
                import numpy as _np2

                eid = rank_inputs[1]
                if eid is not None:
                    bs2 = eid.shape[0]
                    k2 = eid.shape[1] if eid.ndim > 1 else 1
                    ep_ws2 = int(testcase.attributes.get("ep_world_size", ndev))
                    moe_exp2 = int(testcase.attributes.get("moe_expert_num", ep_ws2))
                    local_exp2 = moe_exp2 // ep_ws2 if ep_ws2 > 0 else moe_exp2
                    send_counts2 = [0] * ep_ws2
                    for ii in range(bs2):
                        for jj in range(k2):
                            e_id = int(eid[ii][jj]) if eid.ndim > 1 else int(eid[ii])
                            dest2 = e_id // local_exp2 if local_exp2 > 0 else 0
                            if dest2 >= ep_ws2:
                                dest2 = ep_ws2 - 1
                            send_counts2[dest2] += 1
                    cumsum2 = []
                    run2 = 0
                    for ii in range(ep_ws2):
                        run2 += send_counts2[ii]
                        cumsum2.append(run2)
                    if len(rank_inputs) > 3 and rank_inputs[3] is not None:
                        rank_inputs[3] = _np2.array(cumsum2, dtype=_np2.int32)
                    if len(rank_inputs) > 2 and rank_inputs[2] is not None:
                        rank_inputs[2] = _np2.arange(bs2 * k2, dtype=_np2.int32)
                    if len(rank_inputs) > 4 and rank_inputs[4] is not None:
                        rank_inputs[4] = _np2.ones((bs2, k2), dtype=_np2.float32)
            for i, r in enumerate(rank_inputs):
                if r is not None:
                    # For shared weight (idx >= 1 when needs_shared_weight),
                    # only save rank 0's copy; other ranks will fall back to rank 0.
                    if needs_shared_weight and i >= 1 and rank_idx != 0:
                        continue
                    # fp8/hifloat8 dtypes can't round-trip through npz (e5m2 descr '<f1'
                    # fails on load). Save as uint8 view; worker restores via tensor_dtypes.
                    arr_to_save = r
                    if hasattr(r.dtype, "itemsize") and r.dtype.itemsize == 1 and r.dtype.kind in ("V", "f"):
                        # Could be fp8_e4m3fn (|V1), fp8_e5m2 (<f1), fp8_e8m0, hifloat8
                        dtype_str = str(r.dtype)
                        if any(k in dtype_str for k in ("float8", "hifloat8")):
                            arr_to_save = r.view(_np.uint8)
                    save_arrays[f"inp_{rank_idx}_{i}"] = arr_to_save
        np.savez(input_path, **save_arrays)

        script_path = os.path.join(os.path.dirname(__file__), "_e2e_multi_device_worker.py")
        result_path = os.path.join(tmp_dir, "result.npz")
        error_path = os.path.join(tmp_dir, "error.txt")

        plan_info = {
            "api_name": api_name,
            "overload_index": plan.overload_index,
            "output_tensor_indexes": testcase.output_tensor_indexes,
            "attributes": testcase.attributes or {},
            "golden_disabled": str(getattr(testcase, "golden_api", "") or "").lower() == "disable",
            "tensor_dtypes": list(testcase.tensor_dtypes) if testcase.tensor_dtypes else [],
            "remark": testcase.remark or "",
            "tensor_view_shapes": list(testcase.tensor_view_shapes) if testcase.tensor_view_shapes else [],
            "testcase_name": getattr(testcase, "testcase_name", ""),
            "proc_timeout": int(getattr(switches, "proc_timeout", 0) or 3600),
        }

        plan_path = os.path.join(tmp_dir, "plan.json")
        import json

        with open(plan_path, "w") as f:
            json.dump(plan_info, f)

        env = os.environ.copy()
        env["TTK_E2E_INPUT"] = input_path
        env["TTK_E2E_PLAN"] = plan_path
        env["TTK_E2E_RESULT"] = result_path
        env["TTK_E2E_ERROR"] = error_path
        env["TTK_E2E_DEVICES"] = ",".join(str(d) for d in device_ids)
        env["TTK_E2E_NDEV"] = str(ndev)
        # Pass graph mode to worker: "dynamic"/"static"/"" (multi-device mc2 graph path,
        # mirrors single-device _execute_graph driven by -d/-c switches).
        graph_mode = ""
        if switches.dyn_switches.enabled:
            graph_mode = "dynamic"
        elif switches.cst_switches.enabled:
            graph_mode = "static"
        env["TTK_E2E_GRAPH_MODE"] = graph_mode
        # Pass plugin path to worker so it can load per-operator golden specs.
        if switches.plugin_path:
            env["TTK_E2E_PLUGIN"] = ",".join(str(p_) for p_ in switches.plugin_path)

        cmd = [sys.executable, script_path]
        logging.info(f"E2E multi-device: running {cmd} with devices={device_ids} graph_mode={graph_mode or 'none'}")

        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, errors="replace", check=False)
            if proc.returncode != 0:
                err_msg = proc.stderr[-2000:] if proc.stderr else "unknown"
                if os.path.exists(error_path):
                    with open(error_path) as f:
                        err_msg = f.read()[:3000]
                # Fallback: 某些 torch/torch_npu 版本 worker 进程退出阶段
                # 有 "double free or corruption" 析构崩溃（rc=-6），但结果
                # npz 已正常写出。优先采用实际结果，避免误判为 FAIL。
                fallback_used = False
                if os.path.exists(result_path):
                    try:
                        _loaded = np.load(result_path, allow_pickle=True)
                        if "precision_0" in _loaded.files and "pass_0" in _loaded.files:
                            logging.warning(
                                f"E2E worker rc={proc.returncode} but result.npz written; "
                                f"using fallback result (worker exit crash ignored)."
                            )
                            fallback_used = True
                    except Exception:  # noqa: S110
                        pass
                if not fallback_used:
                    logging.error(f"E2E multi-device worker failed (rc={proc.returncode}): {err_msg}")
                    logging.error(f"E2E stderr: {proc.stderr[-2000:] if proc.stderr else 'none'}")
                    logging.error(f"E2E tmp_dir preserved: {tmp_dir}")
                    return_struct.eager_precision = f"MULTI_DEVICE_FAILED: {err_msg}"
                    return_struct.precision_status = "FAIL"
                    return
        except subprocess.TimeoutExpired:
            import shutil

            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir)
            return_struct.eager_precision = "MULTI_DEVICE_TIMEOUT"
            return_struct.precision_status = "FAIL"
            return

        if not os.path.exists(result_path):
            import shutil

            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir)
            return_struct.eager_precision = "NO_OUTPUT"
            return_struct.precision_status = "FAIL"
            return

        loaded = np.load(result_path, allow_pickle=True)

        # Check if worker did in-process comparison (torch.isclose for bf16 compat)
        precision_keys = [k for k in loaded.files if k.startswith("precision_")]
        if precision_keys:
            prec = str(loaded["precision_0"])
            passed = str(loaded["pass_0"])
            import shutil

            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_dir)
            return_struct.construct(prec, passed, None)
            # Multi-device mc2 graph result (dynamic/static) produced by worker
            if "graph_precision_0" in loaded.files:
                gprec = str(loaded["graph_precision_0"])
                gpass = str(loaded["graph_pass_0"])
                gmode = str(loaded["graph_mode_0"]) if "graph_mode_0" in loaded.files else None
                if gmode == "static":
                    return_struct.construct(gprec, gpass, None, mode="static")
                else:
                    return_struct.construct(gprec, gpass, None, mode="dynamic")
            _profiling_end_print(testcase, return_struct, [], switches)
            del raw_inputs
            return

        primary_result = [loaded[f"out_{i}"] for i in range(len(loaded.files)) if f"out_{i}" in loaded]
        golden_result = [loaded[f"golden_{i}"] for i in range(len(loaded.files)) if f"golden_{i}" in loaded]

        import shutil

        with contextlib.suppress(Exception):
            shutil.rmtree(tmp_dir)

    if golden_result and not any(isinstance(g, str) for g in golden_result):
        golden_nps = golden_result
    else:
        golden_nps = _generate_golden_data(testcase, raw_inputs, switches, backend)

    _apply_pre_compare(testcase, primary_result, golden_nps, switches)

    output_dtypes = tuple(str(g.dtype) if g is not None else None for g in primary_result)
    tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        output_dtypes,
        switches.compare_method,
    )
    try:
        custom_result = _try_custom_compare(testcase, primary_result, golden_nps, switches)
        if custom_result is not None:
            precision_str, log_str, is_pass = custom_result
        else:
            precision_str, log_str, is_pass, _ = compare(primary_result, golden_nps, output_dtypes, standards=standards)
    except Exception:
        logging.exception(f"[{testcase.testcase_name}] Eager comparison failure")
        return_struct.eager_precision = "COMPARE_FAILURE"
        return_struct.precision_status = "FAIL"
        return

    precision_status = "PASS" if is_pass else "FAIL"
    return_struct.construct(precision_str, precision_status, None)
    _profiling_end_print(testcase, return_struct, golden_nps, switches)
    del raw_inputs


def _find_free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _do_profile(  # noqa: PLR0911
    testcase, backend, device_grant_events, device_granted_indices, dev_id, switches, return_struct
):
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
            )
            if manual_case is not None:
                manual_mode = "replay"
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data loading failure")
            return_struct.eager_precision = f"MANUAL_DATA_READ_FAILURE: {exc}"
            return_struct.precision_status = "FAIL"
            return

    process_ctx.notify_status("OnGenInput")
    # 提前设定进程设备：TF 的 npu_device.open/as_default 必须先于首个
    # tf.Tensor 创建（generate_inputs 内），否则 TF context 冻结在 CPU、
    # op 静默跑 CPU（假绿）。torch 侧为幂等的 torch_npu.npu.set_device。
    backend.set_device(dev_id)
    try:
        if manual_case is not None:
            raw_inputs = generate_inputs(testcase, switches, backend, plan, stored_inputs=manual_case.inputs)
        else:
            raw_inputs = generate_inputs(testcase, switches, backend, plan)
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
        try:
            prepared_inputs = snapshot_manual_values(
                testcase.np_storages if testcase.np_storages is not None else raw_inputs,
                "input",
            )
            write_goldens = switches.dump_config.is_golden_enabled()
            golden_nps = (
                _generate_golden_data(testcase, raw_inputs, switches, backend, dump=False) if write_goldens else []
            )
            process_ctx.notify_status("OnWriteManualData")
            case_dir = prepare_store.write_case(
                testcase,
                "e2e",
                prepared_inputs,
                golden_nps,
                file_format=switches.dump_config.file_format,
                write_goldens=write_goldens,
            )
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual data preparation failure")
            return_struct.construct(f"MANUAL_DATA_PREPARE_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return
        logging.info(f"[{testcase.testcase_name}] manual data prepared: {case_dir}")
        return_struct.construct("MANUAL_DATA_PREPARED", "PASS", None)
        _profiling_end_print(testcase, return_struct, golden_nps, switches)
        return

    resolved, is_tensor_method = resolve_api(testcase.api_name)
    is_inplace = resolved.endswith("_") if is_tensor_method else getattr(resolved, "__name__", "").endswith("_")

    graph_enabled = (
        switches.cst_switches.enabled or switches.dyn_switches.enabled or getattr(switches, "aclgraph_enabled", False)
    )
    if graph_enabled and not backend.supports_graph_mode():
        logging.warning(f"Graph mode not supported by backend {backend.device_type()}, skipping graph execution")
        graph_enabled = False

    process_ctx.notify_status("OnAcquireLock")
    use_device = backend.has_device()
    result_nps = None
    perf = None
    eager_det_status = None
    graph_cst_nps = None
    graph_dyn_nps = None
    graph_cst_perf = None
    graph_dyn_perf = None
    graph_aclgraph_nps = None
    graph_aclgraph_perf = None
    graph_cst_det = None
    graph_dyn_det = None
    graph_aclgraph_det = None
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        process_ctx.notify_status("OnProfilingPrint")
        _profiling_print(testcase, backend, dev_id, switches)

        process_ctx.notify_status("OnEagerProfiling")
        if not getattr(switches, "aclgraph_enabled", False):
            result_nps, perf, eager_det_status = _capture_stdout_npu_memory(
                testcase,
                return_struct,
                _execute_eager,
                testcase,
                backend,
                dev_id,
                switches,
                plan,
                resolved,
                is_tensor_method,
                is_inplace,
                raw_inputs,
            )
        if graph_enabled:
            from .framework_detector import detect_framework

            if detect_framework(testcase.api_name) == "tf":
                from .tf_graph_execution import _execute_tf_graph

                graph_fn = _execute_tf_graph
            else:
                from .graph_execution import _execute_graph

                graph_fn = _execute_graph
            if getattr(switches, "aclgraph_enabled", False):
                process_ctx.notify_status("OnGraphAclgraph")
                graph_aclgraph_nps, graph_aclgraph_perf, graph_aclgraph_det = graph_fn(
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
                graph_cst_nps, graph_cst_perf, graph_cst_det = graph_fn(
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
                graph_dyn_nps, graph_dyn_perf, graph_dyn_det = graph_fn(
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

    det_statuses = [eager_det_status, graph_cst_det, graph_dyn_det, graph_aclgraph_det]
    det_statuses = [s for s in det_statuses if s is not None]
    if det_statuses:
        return_struct.deterministic_status = "FAIL" if "FAIL" in det_statuses else "PASS"

    # 被测侧输出(device 结果)。容差按它的 dtype 解析,不能按 golden 的——
    # Promote 会把 golden 抬到 fp64,拿 golden dtype 去 resolve_tolerance 要么查不到该
    # dtype 的档、要么把 cross_check 判丢,三方就静默不采集了。
    ref_nps = result_nps or graph_cst_nps or graph_dyn_nps or graph_aclgraph_nps

    if manual_case is not None:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            reference_outputs = result_nps
            if reference_outputs is None:
                reference_outputs = graph_cst_nps
            if reference_outputs is None:
                reference_outputs = graph_dyn_nps
            if reference_outputs is None:
                reference_outputs = graph_aclgraph_nps
            if manual_case.has_goldens:
                golden_nps = manual_case.load_goldens(references=reference_outputs)
                _dump_goldens(testcase, golden_nps, switches)
            else:
                golden_nps = _generate_golden_maybe_promote(testcase, raw_inputs, switches, backend, ref_nps)
        except Exception as exc:
            logging.exception(f"[{testcase.testcase_name}] Manual golden loading failure")
            return_struct.construct(f"MANUAL_DATA_READ_FAILURE: {exc}", "FAIL", None)
            _profiling_end_print(testcase, return_struct, switches=switches)
            return
    else:
        golden_nps = _generate_golden_maybe_promote(testcase, raw_inputs, switches, backend, ref_nps)

    third_parties = None
    xpu_results = None
    if golden_nps and not any(isinstance(g, str) for g in golden_nps):
        tolerance = get_spec_attr(testcase.api_name, "tolerance", switches.plugin_path)
        # 按被测输出的 dtype 解析容差(golden 可能已被 Promote 抬精度,拿它解析会走偏);
        # 拿不到 device 结果时退回 golden,保持原行为。
        dtype_src = ref_nps if ref_nps else golden_nps
        output_dtypes = tuple(str(g.dtype) if g is not None and hasattr(g, "dtype") else None for g in dtype_src)
        standards = resolve_tolerance(
            tolerance,
            testcase.flat_precision_tolerances,
            testcase.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
        )
        need_3party = any(s.token == "cross_check" for s in standards)  # noqa: S105
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

        _apply_pre_compare(testcase, result_nps, golden_nps, switches)
        process_ctx.notify_status("OnEagerComparison")
        _evaluate_eager_precision(
            testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct, third_parties
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
            )

    _profiling_end_print(testcase, return_struct, golden_nps, switches)
    del raw_inputs
