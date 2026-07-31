#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""
GEIR profiling entry — runs in a subprocess.
Generates inputs in Python, writes to files, executes C++ binary, compares.
"""

import gc
import logging
import os
import pathlib
import subprocess
import threading

import numpy as np

from ttk.core_modules.comparison.comparison import compare
from ttk.core_modules.npu.op.input_generation import __gen_input
from ttk.core_modules.npu.op.output_generation import __gen_output
from ttk.core_modules.tbe_logging import default_logging_config
from ttk.core_modules.tbe_multiprocessing import DeviceLock, get_process_context
from ttk.utilities import dump_to_file, get_global_storage, resolve_custom_numpy_dtypes, waiting_for_memory

from .compiler import GeirCompiler
from .graph_builder import GeirGraphBuilder


def _geir_profiling_end_print(result):
    switches = get_global_storage()
    is_bin = getattr(switches, "geir_binary", False)
    cst_label = "CST_BIN_GOLD" if is_bin else "CST_GOLD"
    dyn_label = "DYN_BIN_GOLD" if is_bin else "DYN_GOLD"
    cst_perf_label = "CST_BIN_PERF" if is_bin else "CST_PERF"
    dyn_perf_label = "DYN_BIN_PERF" if is_bin else "DYN_PERF"
    passed_str = "PASS" if result.passed else "FAIL"
    logging.info(
        "\n########################\n"
        "Performance result:                Comparison result:\n"
        f"{cst_perf_label}: {str(result.cst_perf_us).ljust(24)} "
        f"{cst_label}: {result.cst_precision.ljust(24)}\n"
        f"{dyn_perf_label}: {str(result.dyn_perf_us).ljust(24)} "
        f"{dyn_label}: {result.dyn_precision.ljust(24)}\n"
        f"STATUS: {result.status.ljust(26)} PRECISION_STATUS: {passed_str}\n"
        "########################\n"
    )


def geir_profile_process(testcase, device_grant_events, device_granted_indices, dev_id):
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(testcase.testcase_name)

    if switches.single_testcase_log_mode:
        default_logging_config(file_handler=switches.logging_to_file, testcase_name=testcase.testcase_name)

    if not testcase.is_valid:
        from .geir_struct import GeirReturnStructure

        result = GeirReturnStructure()
        result.status = "INVALID"
        result.precision = testcase.fail_reason or "INVALID"
        result.passed = False
        return result

    if not switches.no_memory_check:
        process_ctx.notify_status("OnWaitingForMemory")
        waiting_for_memory()

    try:
        with DeviceLock(
            process_ctx,
            dev_id,
            use_device=True,
            grant_event=device_grant_events.get(dev_id),
            granted_idx=device_granted_indices.get(dev_id),
        ):
            modes = []
            for base in ("const", "dynamic"):
                base_enabled = switches.cst_switches.enabled if base == "const" else switches.dyn_switches.enabled
                if not base_enabled:
                    continue
                if switches.geir_binary:
                    modes.append(f"{base}_binary")
                else:
                    modes.append(base)

            result = None
            for mode in modes:
                mode_result = _geir_run(testcase, dev_id, switches, process_ctx, mode)
                if mode in ("const", "const_binary"):
                    mode_result.cst_precision = mode_result.precision
                elif mode in ("dynamic", "dynamic_binary"):
                    mode_result.dyn_precision = mode_result.precision
                if result is None:
                    result = mode_result
                else:
                    if mode_result.cst_precision:
                        result.cst_precision = mode_result.cst_precision
                    if mode_result.dyn_precision:
                        result.dyn_precision = mode_result.dyn_precision
                    if mode_result.cst_perf_us is not None:
                        result.cst_perf_us = mode_result.cst_perf_us
                    if mode_result.dyn_perf_us is not None:
                        result.dyn_perf_us = mode_result.dyn_perf_us
                    if mode_result.xpu_metrics:
                        result.xpu_metrics.update(mode_result.xpu_metrics)
                    if mode_result.passed and not result.passed:
                        result.precision = mode_result.precision
                        result.passed = True
                        result.status = mode_result.status
                        result.log = mode_result.log
    except Exception as e:
        logging.exception("GEIR profiling exception")
        from .geir_struct import GeirReturnStructure

        result = GeirReturnStructure()
        result.status = f"ERROR: {str(e)[:200]}"
        result.precision = "EXEC_FAILURE"
        result.passed = False

    process_ctx.notify_status("OnReturning")
    _geir_profiling_end_print(result)
    gc.collect()
    return result


def _geir_run(testcase, dev_id, switches, process_ctx, mode="const"):
    process_ctx.notify_status("OnGenerateInput")
    testcase.device_id = dev_id

    # Generate inputs (same as kernel mode)
    __gen_input(testcase)

    # resolve 提前到 __gen_output 之前（唯一目的：cross_check 时设 golden_mode_override
    # 让 golden 走 Promote 高精度真值）。与 npu/op/profiling.py:286-300 对齐。
    process_ctx.notify_status("OnResolveTolerance")
    from ..comparison.resolve import resolve_tolerance

    tolerance = None
    try:
        from ttk.test_spec import get_spec_attr

        tolerance = get_spec_attr(testcase.op_name, "tolerance", getattr(switches, "plugin_path", None))
    except Exception:
        pass
    standards = resolve_tolerance(
        tolerance,
        testcase.flat_precision_tolerances,
        testcase.flat_absolute_precision,
        testcase.flat_output_dtypes,
        switches.compare_method,
    )
    need_3party_outputs = any(s.token == "cross_check" for s in standards)
    if need_3party_outputs:
        testcase.golden_mode_override = "Promote"

    # Generate golden (same as kernel mode, uses testcase.input_arrays)
    process_ctx.notify_status("OnGenerateGolden")
    try:
        __gen_output(testcase)
    finally:
        if hasattr(testcase, "golden_mode_override"):
            del testcase.golden_mode_override
    testcase.output_arrays = ()

    # 三方输出采集（cross_check 需要 / xpu-perf 需要）。GEIR 的 input_names 来自
    # ProtoLoader，op_type 无概念传 None（服务端靠 op_name 推导）。
    from ttk.remote.client import xpu_mode_of

    xpu_mode = xpu_mode_of(switches, need_3party_outputs)
    third_parties = None
    xpu_results = None
    if xpu_mode:
        process_ctx.notify_status("OnXpuProfiling")
        from ttk.remote.client import collect_third_party

        from .proto_loader import ProtoLoader

        try:
            proto_info = ProtoLoader().get_op_info(testcase.op_name)
            input_names = proto_info.inputs[:] if proto_info else []
        except Exception:
            input_names = []
        ori_inputs = getattr(testcase, "original_input_arrays", None) or testcase.input_arrays
        _, third_parties, xpu_results = collect_third_party(
            op_name=testcase.op_name,
            inputs=ori_inputs,
            input_names=input_names,
            op_type=None,
            attributes=getattr(testcase, "attributes", {}) or {},
            testcase_name=testcase.testcase_name,
            switches=switches,
            need_data=need_3party_outputs,
        )

    # Build C++ source and compile
    process_ctx.notify_status("OnGeirCompile")
    builder = GeirGraphBuilder(switches)
    source_path = builder.generate(testcase, mode=mode)
    if source_path is None:
        raise RuntimeError("GEIR source generation failed")

    compiler = GeirCompiler(switches, build_dir=builder.work_dir)
    binary = compiler.compile(source_path, testcase.testcase_name, proto_file=builder.last_proto_file)
    if binary is None:
        raise RuntimeError("GEIR compilation failed")

    # Write input files for C++ program (skip None inputs, use contiguous index)
    input_prefix = os.path.join(compiler.build_dir, f"{testcase.testcase_name}_input")
    data_idx = 0
    for i, arr in enumerate(testcase.input_arrays):
        if arr is not None:
            path = f"{input_prefix}_{data_idx}.bin"
            arr.tofile(path)
            data_idx += 1
    testcase.original_input_arrays = None

    # Execute C++ program
    # Data channel: os.pipe() carries the binary output protocol (8B num_outputs
    # + 8B byte_count + data per output). Keeps stdout/stderr free for GE logs.
    # A reader thread drains the pipe to avoid deadlock when output > pipe buffer
    # (64KB); Popen.communicate() handles stdout/stderr in parallel.
    process_ctx.notify_status("OnGeirExecute")

    prof_path = ""
    if getattr(switches, "TASK_PROFILING", False):
        root = getattr(switches, "root_path", os.getcwd())
        prof_path = os.path.join(root, "msprof", "geir", testcase.testcase_name, mode)
        os.makedirs(prof_path, exist_ok=True)

    data_r, data_w = os.pipe()
    data_holder = []

    def _drain_pipe():
        chunks = []
        while True:
            chunk = os.read(data_r, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        data_holder.append(b"".join(chunks))

    reader = threading.Thread(target=_drain_pipe, name=f"geir_data_{testcase.testcase_name}")
    reader.start()
    try:
        try:
            proc = subprocess.Popen(
                [binary, str(dev_id), input_prefix, str(data_w), prof_path],
                pass_fds=(data_w,),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=compiler.build_dir,
            )
        finally:
            os.close(data_w)  # parent closes write-end so reader gets EOF
    except Exception:
        os.close(data_r)
        reader.join(timeout=5)
        raise

    try:
        stdout_bytes, stderr_bytes = proc.communicate(timeout=switches.proc_timeout or 1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        reader.join(timeout=5)
        os.close(data_r)
        compiler.cleanup(input_prefix, source_path)
        raise RuntimeError("GEIR execution timed out")
    reader.join(timeout=30)
    os.close(data_r)

    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:2000]
        compiler.cleanup(input_prefix, source_path)
        raise RuntimeError(f"GEIR execution failed (rc={proc.returncode}): {stderr}")

    # Capture plog to TTK logging. stdout holds plog echo (when
    # ASCEND_SLOG_PRINT_TO_STDOUT=1); stderr holds [TTK-GEIR] markers.
    if stdout_bytes:
        for line in stdout_bytes.decode("utf-8", errors="replace").splitlines():
            if line.strip():
                logging.info("[GEIR/%s] %s", testcase.testcase_name, line)
    if stderr_bytes:
        logging.debugc("[GEIR/%s] stderr:\n%s", testcase.testcase_name, stderr_bytes.decode("utf-8", errors="replace"))

    # Device profiling: run msprof.py export summary, parse op_summary CSV for Task Duration(us)
    device_perf_us = None
    if prof_path:
        device_perf_us = _parse_msprof_task_duration(prof_path)

    # Parse outputs from data channel
    output_arrays = _parse_stdout(
        data_holder[0] if data_holder else b"",
        testcase.output_dtypes,
        testcase.flat_output_shapes,
        case_name=testcase.testcase_name,
    )
    data_holder.clear()

    golden_arrays = testcase.golden_arrays

    dump_cfg = switches.dump_config
    if not dump_cfg.is_input_enabled() and not dump_cfg.dump_on_fail:
        testcase.input_arrays = None

    # Compare
    process_ctx.notify_status("OnGeirCompare")
    flat_out_dtypes = tuple(resolve_custom_numpy_dtypes(testcase.flat_output_dtypes))

    precision, log_str, passed, _metrics = compare(
        tuple(output_arrays),
        tuple(golden_arrays),
        flat_out_dtypes,
        standards=standards,
        third_parties=third_parties,
    )

    _std_tokens = sorted({str(s.token) for s in standards}) if standards else []
    _std = ",".join(_std_tokens) if _std_tokens else "unknown"
    logging.debugc(f"\nComparing geir_{testcase.testcase_name} with {_std}\n{log_str}")

    # Dump
    dump_path = getattr(switches, "root_path", os.getcwd())
    if dump_cfg.is_input_enabled():
        for i, arr in enumerate(testcase.input_arrays):
            dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_input_{i}", dump_cfg.file_format)
    if dump_cfg.is_output_enabled():
        for i, arr in enumerate(output_arrays):
            dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_output_{i}", dump_cfg.file_format)
    if dump_cfg.is_golden_enabled():
        for i, arr in enumerate(golden_arrays):
            if isinstance(arr, np.ndarray):
                dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_golden_{i}", dump_cfg.file_format)

    if dump_cfg.dump_on_fail and not passed:
        for i, arr in enumerate(testcase.input_arrays):
            dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_fail_input_{i}", dump_cfg.file_format)
        for i, arr in enumerate(output_arrays):
            dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_fail_output_{i}", dump_cfg.file_format)
        for i, arr in enumerate(golden_arrays):
            if isinstance(arr, np.ndarray):
                dump_to_file(arr, dump_path, f"{testcase.testcase_name}_geir_fail_golden_{i}", dump_cfg.file_format)

    compiler.cleanup(input_prefix, source_path)

    from .geir_struct import GeirReturnStructure

    xpu_metrics = _format_xpu_metrics(xpu_results) if xpu_results else {}

    result = GeirReturnStructure()
    result.precision = precision
    result.passed = passed
    result.status = "SUCCESS" if passed else "FAIL"
    result.log = log_str
    result.xpu_metrics = xpu_metrics
    if mode in ("const", "const_binary"):
        result.cst_perf_us = device_perf_us
    elif mode in ("dynamic", "dynamic_binary"):
        result.dyn_perf_us = device_perf_us
    return result


def _parse_stdout(data, output_dtypes, output_shapes, case_name=""):
    import io

    if not data:
        return []

    buf = io.BytesIO(data)
    num_outputs_raw = buf.read(8)
    if len(num_outputs_raw) < 8:
        return []
    num_outputs = int(np.frombuffer(num_outputs_raw, dtype=np.int64)[0])

    results = []
    for i in range(num_outputs):
        byte_count_raw = buf.read(8)
        if len(byte_count_raw) < 8:
            break
        byte_count = int(np.frombuffer(byte_count_raw, dtype=np.int64)[0])
        raw = buf.read(byte_count)
        if len(raw) < byte_count:
            break

        dtype_str = (
            output_dtypes[i]
            if isinstance(output_dtypes, (list, tuple)) and i < len(output_dtypes)
            else str(output_dtypes[0])
        )
        np_dtype = resolve_custom_numpy_dtypes([dtype_str])[0]
        itemsize = np.dtype(np_dtype).itemsize
        try:
            arr = np.frombuffer(raw, dtype=np.dtype(np_dtype))
        except ValueError as e:
            usable = byte_count - (byte_count % itemsize) if itemsize else 0
            logging.warning(
                "[%s] output[%d] frombuffer failed (%s); byte_count=%d not multiple of "
                "itemsize=%d (dtype=%s); truncating to %d usable bytes",
                case_name,
                i,
                e,
                byte_count,
                itemsize,
                dtype_str,
                usable,
            )
            arr = (
                np.frombuffer(raw[:usable], dtype=np.dtype(np_dtype))
                if usable
                else np.array([], dtype=np.dtype(np_dtype))
            )

        expect_shape = output_shapes[i] if isinstance(output_shapes, (list, tuple)) and i < len(output_shapes) else None
        if arr.size == 0 and expect_shape is not None:
            expect_numel = int(np.prod(expect_shape)) if expect_shape else 0
            logging.warning(
                "[%s] output[%d] empty from GE (byte_count=%d, dtype=%s, "
                "expect_shape=%s, expect_numel=%d); skipping reshape to avoid crash",
                case_name,
                i,
                byte_count,
                dtype_str,
                expect_shape,
                expect_numel,
            )
            results.append(arr)
            continue
        if expect_shape is not None:
            try:
                arr = arr.reshape(expect_shape)
            except ValueError as e:
                logging.warning(
                    "[%s] output[%d] reshape failed (%s); arr.size=%d, expect_shape=%s; keeping flat array",
                    case_name,
                    i,
                    e,
                    arr.size,
                    expect_shape,
                )
        results.append(arr)

    return results


def _parse_msprof_task_duration(prof_path: str):
    """Run msprof.py export summary and parse op_summary CSV for Task Duration(us).

    Mirrors Kernel mode OnlineRtsProfiling._process_msprof_cycles
    (rts_sequence.py:468-504). Returns median device time in µs, or None
    when profiling data is unavailable / empty / msprof tool missing.
    """
    # 1. msprof.py export summary (converts binary dumps to CSV)
    opp_path = os.getenv("ASCEND_OPP_PATH", "")
    msprof_py = os.path.join(opp_path, "..", "tools", "profiler", "profiler_tool", "analysis", "msprof", "msprof.py")
    if not os.path.exists(msprof_py):
        logging.warning("msprof.py not found at %s, skip device profiling", msprof_py)
        return None

    for t in ("summary",):
        try:
            subprocess.run(
                ["python3", msprof_py, "export", t, "-dir", prof_path],
                capture_output=True,
                timeout=120,
            )
        except Exception as exc:
            logging.warning("msprof export %s failed: %s", t, exc)
            return None

    # 2. Parse op_summary_*.csv / task_time_*.csv for kernel Task Duration
    KERNEL_TYPE = (
        "AI_CORE",
        "AIV_SQE",
        "AI_VECTOR_CORE",
        "MIX_AIC",
        "MIX_AIV",
        "KERNEL_MIX_AIC",
        "KERNEL_MIX_AIV",
        "KERNEL_AIVEC",
        "KERNEL_AICORE",
    )
    prof_dir = pathlib.Path(prof_path)
    csv_files = sorted(prof_dir.glob("**/*.csv"))
    durations = []
    for item in csv_files:
        if item.name.startswith("op_summary_"):
            durations = _extract_csv_task_duration(item, "Task Duration(us)", "Task Type", KERNEL_TYPE)
        elif item.name.startswith("task_time_"):
            durations = _extract_csv_task_duration(item, "task_time(us)", "kernel_type", KERNEL_TYPE)
        if durations:
            break

    if durations:
        median_us = round(float(np.median(durations)), 3)
        logging.debugc("[GEIR] device perf: median=%.3f us, all=%s", median_us, durations)
        return median_us
    return None


def _extract_csv_task_duration(csv_path, duration_col, type_col, kernel_types):
    """Extract Task Duration from msprof summary CSV, filtered by kernel type."""
    import csv

    results = []
    with open(csv_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get(type_col, "") not in kernel_types:
                continue
            try:
                results.append(float(row[duration_col]))
            except (ValueError, KeyError):
                pass
    return results


def _format_xpu_metrics(xpu_results):
    """Format xpu_results dict into xpu_metrics dict for CSV output, same as kernel mode."""
    if not xpu_results:
        return {}
    metrics = {}
    for provider, entry in xpu_results.items():
        m = {"status": entry.get("status", "FAIL"), "api": entry.get("api", "")}
        if entry.get("perf"):
            m["device_us"] = entry["perf"].get("device_us", "NA")
            m["peak_memory_mb"] = entry["perf"].get("peak_memory_mb", "NA")
        if entry.get("error"):
            m["error"] = entry["error"]
        metrics[provider] = m
    return metrics
