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
Profiling method for Universal testcases
"""
# Standard Packages
import logging
import os
import shutil
from typing import NoReturn, Optional, Tuple, Union

from ttk.remote import DATA, get_tenant_id
from ttk.core_modules.operator import OpInfoKeeper

# third_party key aliases — mirrors server executor.py:_TP_ALIASES; keep in sync.
# Applied in _extract_spec_providers (resolve_providers) and _build_spec (tp -> server).
_TP_ALIASES = {"tensorflow": "tf", "np": "numpy"}

try:
    from contextlib import nullcontext
except ImportError:
    print("Python version too low, from contextlib import nullcontext failed")
    import contextlib


    @contextlib.contextmanager
    def NULLCXT():
        """NULL CONTEXT"""
        pass


    nullcontext = NULLCXT

# Third-Party Packages
import numpy
from .comparison import comparing
from .input_generation import __gen_input
from .output_generation import __gen_output
from .profiling_structure import RTSProfilingParam, RTSProfilingResult
from .profiling_structure import ComparisonResult, ProfilingReturnStructure
from .rts_sequence import rts_profiling
from ...manual_data import (
    load_manual_data_case,
    prepare_manual_data_store,
    snapshot_manual_values,
)
from ...runtime import RTSInterface, RTSInterfaceBase
from ...tbe_logging import default_logging_config
from ...tbe_multiprocessing import get_process_context, DeviceLock
from ...testcase_manager import TestcaseOp
from ...operator import OpInfoKeeper
from ....test_spec import get_spec_attr
from ....utilities import resolve_custom_numpy_dtypes, get, get_global_storage, get_str_tiling_data
from ....utilities import parse_tiling_data, table_print, deep_flatten
from ....utilities import dump_to_file, waiting_for_memory


def __gen_workspaces(workspaces: Tuple[int], debug_buf_size: int) -> Tuple[numpy.ndarray, ...]:
    workspaces = workspaces or ()
    workspace_byte_arrays = []
    for workspace_size in workspaces:
        if workspace_size:
            workspace_byte_arrays.append(numpy.random.randint(0, 255,
                                                              size=(int(workspace_size),),
                                                              dtype=numpy.uint8))
        else:
            workspace_byte_arrays.append(None)
    if debug_buf_size > 0:
        if not workspace_byte_arrays:
            workspace_byte_arrays.append(None)
        first_ws_size = 0 if workspace_byte_arrays[0] is None else workspace_byte_arrays[0].size
        workspace_byte_arrays[0] = numpy.random.randint(0, 255,
                                                        size=(first_ws_size + debug_buf_size,),
                                                        dtype=numpy.uint8)
    return tuple(workspace_byte_arrays)


def construct_compile_result(context):
    dyn_compile_result = context.dyn_compile_result.compile_result
    cst_compile_result = context.cst_compile_result.compile_result
    bin_compile_result = context.bin_compile_result.compile_result

    context.dyn_prof_result = RTSProfilingResult(dyn_compile_result)
    context.cst_prof_result = RTSProfilingResult(cst_compile_result)
    context.bin_prof_result = RTSProfilingResult(bin_compile_result)


def prof_end(context, print_content):
    compare_result = ComparisonResult(print_content)
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, print_content)
    __profiling_end_print(context, compare_result, print_content)
    return return_structure


def _manual_data_prepared_end(context: TestcaseOp):
    compare_result = ComparisonResult(None).set(
        "MANUAL_DATA_PREPARED",
        "MANUAL_DATA_PREPARED",
        "MANUAL_DATA_PREPARED",
        "PASS",
    )
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, "PASS")
    __profiling_end_print(context, compare_result, "PASS")
    return return_structure


def prof_compile_fail_end(context: TestcaseOp):
    construct_compile_result(context)
    passed = handle_profiling_result(context)
    compare_result = ComparisonResult(passed).set(context.dyn_compile_result.compile_result,
                                                  context.cst_compile_result.compile_result,
                                                  context.bin_compile_result.compile_result,
                                                  passed)
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, passed)
    __profiling_end_print(context, compare_result, passed)
    return return_structure


def prof_compile_only_end(context: TestcaseOp):
    construct_compile_result(context)
    compare_result = ComparisonResult("COMPILE_ONLY")
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, "COMPILE_ONLY")
    __compile_only_end_print(context)
    return return_structure


def _extract_spec_providers(tp):
    """Spec-layer provider keys (priority order = insertion order).

    dict -> keys; str -> single provider derived from API prefix; None/empty -> [].
    (Empty -> caller lets EndpointView.resolve_providers use detect∩yaml∩alive.)
    """
    if isinstance(tp, dict):
        return [_TP_ALIASES.get(k, k) for k in tp.keys()]
    if isinstance(tp, str):
        from ttk.remote import _derive_provider_from_api
        return [_derive_provider_from_api(tp, "torch")]
    return []


def _build_spec(provider, tp, spec_file, spec_class, op_name, op_type):
    """Build one ExecutionSpec. api source: third_party dict | str | None.

    - dict[str, str]      -> type='api', api=value
    - dict[str, type]     -> type='spec' (impl class; server resolves
                             cls.third_party[provider]); spec_file/class
                             carried so the server can sync the module
    - str                 -> type='api', api=tp
    - None / no spec      -> type='api', api=None (server _resolve_3party_api
                             derives api from op_name + op_type)
    """
    from pathlib import Path
    from ttk.remote import ExecutionSpec
    if isinstance(tp, dict) and provider in tp:
        v = tp[provider]
        if isinstance(v, str):
            return ExecutionSpec(provider=provider, type="api", api=v)
        # impl class -> spec mode (server resolves cls.third_party[provider])
        return ExecutionSpec(provider=provider, type="spec",
                             spec_file=spec_file,
                             spec_module=Path(spec_file).stem if spec_file else None,
                             spec_class=spec_class)
    if isinstance(tp, str):
        return ExecutionSpec(provider=provider, type="api", api=tp)
    # no spec / no third_party -> api=None, server _resolve_3party_api(op_name, op_type)
    return ExecutionSpec(provider=provider, type="api", api=None)


def _xpu_inputs(context):
    """XPU (torch/tf) needs the logical ori-shape arrays; NPU run-format
    (input_arrays, e.g. NC1HWC0) is unusable on non-NPU accelerators. Fallback if unset."""
    return context.original_input_arrays or context.input_arrays


def _xpu_mode(switches, need_data):
    """按位或：xpu_perf→PERF，need_data→DATA。返回 0/PERF/DATA/DATA|PERF。"""
    from ttk.remote import DATA, PERF
    mode = 0
    if switches.xpu_perf:
        mode |= PERF
    if need_data:
        mode |= DATA
    return mode


def _extract_third_party(xpu_results, priority):
    """从 priority provider 取 outputs（纯函数，直接索引，不靠 dict 序）。
    fail-closed：无 results / 无 priority / 非 PASS / 无 outputs → None（cross_check → GOLDEN_FAILURE）。"""
    if not xpu_results or priority is None:
        return None
    entry = xpu_results.get(priority, {})
    if entry.get("status") != "PASS" or "outputs" not in entry:
        return None
    return entry["outputs"]


def _do_xpu_profiling(context, xpu_mode):
    """Run XPU dispatch，return priority provider name（specs[0].provider）。
    resolve_providers 失败 → xpu_results={} + return None（→ _extract_third_party None → GOLDEN_FAILURE）。"""
    try:
        op_info = OpInfoKeeper().info_of(context.op_name)
        input_names = [ipt["name"] for ipt in op_info["inputs"]] if op_info else []
    except Exception:
        input_names = []

    from ttk.test_spec import get_spec_attr, get_spec_class_meta
    from ttk.remote.endpoint_view import EndpointView, _parse_provider_filter

    op_type = OpInfoKeeper().op_type_of(context.op_name)
    paths = get_global_storage().plugin_path or ()
    tp = get_spec_attr(context.op_name, "third_party", paths)
    if isinstance(tp, dict):
        tp = {_TP_ALIASES.get(k, k): v for k, v in tp.items()}
    meta = get_spec_class_meta(context.op_name, paths)
    spec_file = meta["spec_file"] if meta else None
    spec_class = meta["class_name"] if meta else None

    ev = EndpointView()
    spec_providers = _extract_spec_providers(tp)
    sw = get_global_storage()
    cli_providers = _parse_provider_filter(sw.provider_filter if sw else None)

    try:
        providers = ev.resolve_providers(spec_providers, cli_providers)
    except RuntimeError as e:
        context.xpu_results = {}
        logging.error("[%s] XPU resolve failed: %s",
                      getattr(context, "testcase_name", context.op_name), e)
        return None

    specs = [_build_spec(p, tp, spec_file, spec_class, context.op_name, op_type) for p in providers]

    from ttk.remote.xpu_collector import collect_xpu_results
    _tmp_root = os.path.join(get_global_storage().root_path, ".ttk", "xpu_tmp")
    context.xpu_results = collect_xpu_results(
        specs,
        inputs=_xpu_inputs(context),
        input_names=input_names,
        mode=xpu_mode,
        tenant_id=get_tenant_id(),
        op_name=context.op_name,
        op_type=op_type,
        attrs=context.attributes if hasattr(context, 'attributes') else {},
        tmp_root=_tmp_root,
        runtime=get_global_storage().run_time,
    )
    return specs[0].provider if specs else None


def profile_process(context: TestcaseOp,
                    device_grant_events: dict,
                    device_granted_indices: dict,
                    dev_id: int) -> ProfilingReturnStructure:
    """
    Universal Testcase Profiling Entrance
    """
    from ..error_cleaner import clear_error_manager
    clear_error_manager()

    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(context.testcase_name)
    if switches.single_testcase_log_mode:
        default_logging_config(file_handler=switches.logging_to_file, testcase_name=context.testcase_name)
    manual_mode = getattr(switches, "manual_data_mode", None)
    manual_case = None
    try:
        prepare_store = prepare_manual_data_store(context, "kernel", switches)
    except Exception as exc:
        logging.exception("Manual Kernel data preparation failure")
        return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    process_ctx.notify_status("OnParseParameters")
    __parse_manual_params(context)
    if context.is_valid:
        __parse_dynamic_tiling_data(context)
        __parse_binary_tiling_data(context)
    ####################
    # Check whether there is need to do further test
    ####################
    if not context.is_valid:
        return prof_end(context, context.fail_reason)
    if switches.compile_only:
        return prof_compile_only_end(context)
    if context.compile_failed():
        return prof_compile_fail_end(context)
    if manual_mode != "prepare":
        try:
            manual_case = load_manual_data_case(
                context,
                "kernel",
                switches,
                before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
            )
            if manual_case is not None:
                manual_mode = "replay"
        except Exception as exc:
            logging.exception("Manual Kernel data loading failure")
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
    if not switches.no_memory_check:
        process_ctx.notify_status("OnWaitingForMemory")
        waiting_for_memory()
    logging.debug(f"Expecting {context.input_bytes + context.output_bytes} bytes memory usage")
    process_ctx.notify_status("OnGenInput")
    # noinspection PyBroadException
    try:
        if manual_case is None:
            __gen_input(context)
        else:
            __gen_input(context, stored_inputs=manual_case.inputs)
    except Exception as exc:
        logging.exception("Input data generation failure:")
        if manual_case is not None:
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return prof_end(context, "INPUT_GEN_FAILURE")
    prepared_inputs = None
    if manual_mode == "prepare":
        try:
            prepared_inputs = snapshot_manual_values(
                tuple(deep_flatten(context.input_arrays or ())), "input"
            )
        except Exception as exc:
            logging.exception("Manual Kernel input snapshot failure")
            return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    process_ctx.notify_status("OnGenGolden")
    # resolve 提前到 __gen_output 之前（唯一目的：设 golden_mode_override 让 golden 走 Promote）
    # 3 点相对 import：profiling.py 在 npu/op/，core_modules/comparison 在上 2 级（同 npu/op/comparison.py:20）
    from ...comparison.resolve import resolve_tolerance as _resolve_tolerance
    try:
        tolerance = get_spec_attr(context.op_name, "tolerance",
                                  getattr(get_global_storage(), "plugin_path", None))
        standards = _resolve_tolerance(tolerance,
                                       context.flat_precision_tolerances,
                                       context.flat_absolute_precision,
                                       context.flat_output_dtypes,
                                       get_global_storage().compare_method)
        need_3party_outputs = any(s.token == "cross_check" for s in standards)
        if need_3party_outputs:
            context.golden_mode_override = "Promote"
    except ValueError as e:
        logging.error("[%s] tolerance invalid: %s", context.op_name, e)
        return prof_end(context, "TOLERANCE_INVALID")

    stored_goldens = None
    if manual_case is not None:
        try:
            process_ctx.notify_status("OnLoadManualGolden")
            stored_goldens = manual_case.load_goldens(
                shapes=context.flat_output_shapes,
                dtypes=context.flat_output_dtypes,
            )
        except Exception as exc:
            logging.exception("Manual Kernel golden loading failure")
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")

    # golden gen with override cleanup
    try:
        if manual_case is None:
            __gen_output(context)
        else:
            __gen_output(context, stored_goldens=stored_goldens)
    except Exception as exc:
        logging.exception("Output buffer initialization data or golden data generation failure:")
        if manual_case is not None:
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return prof_end(context, "OUTPUT_GEN_FAILURE")
    finally:
        if hasattr(context, "golden_mode_override"):
            del context.golden_mode_override

    if manual_mode == "prepare":
        try:
            process_ctx.notify_status("OnWriteManualData")
            case_dir = prepare_store.write_case(
                context,
                "kernel",
                prepared_inputs,
                context.golden_arrays,
                file_format=switches.dump_config.file_format,
            )
        except Exception as exc:
            logging.exception("Manual Kernel data preparation failure")
            return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
        logging.info("[%s] manual Kernel data prepared: %s", context.testcase_name, case_dir)
        return _manual_data_prepared_end(context)

    process_ctx.notify_status("OnXpuProfiling")
    xpu_mode = _xpu_mode(get_global_storage(), need_3party_outputs)
    xpu_priority = _do_xpu_profiling(context, xpu_mode) if xpu_mode else None
    third_parties_nested = _extract_third_party(getattr(context, "xpu_results", None), xpu_priority)
    if need_3party_outputs and third_parties_nested is None:
        logging.warning("[%s] cross_check configured but no third_party output (no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE", context.op_name)
    third_parties = list(deep_flatten(third_parties_nested)) if third_parties_nested is not None else None

    process_ctx.notify_status("OnGenWorkspace")
    context.dyn_workspace_arrays = __gen_workspaces(context.dyn_compile_result.workspaces,
                                                    context.dyn_compile_result.debug_buf_size)
    context.cst_workspace_arrays = __gen_workspaces(context.cst_compile_result.workspaces,
                                                    context.cst_compile_result.debug_buf_size)
    context.bin_workspace_arrays = __gen_workspaces(context.bin_compile_result.workspaces,
                                                    context.bin_compile_result.debug_buf_size)
    process_ctx.notify_status("OnDumpInputDataIfRequired")
    __dump_input(context)
    process_ctx.notify_status("OnDumpGoldenDataIfRequired")
    __dump_golden(context)
    # Following actions need to acquire global lock
    process_ctx.notify_status("OnAcquireLock")
    device_id = [dev_id]
    use_device = switches.mode.use_device()
    with DeviceLock(process_ctx, dev_id, use_device=use_device,
                    grant_event=device_grant_events.get(dev_id),
                    granted_idx=device_granted_indices.get(dev_id)):
        context.device_id = device_id[0]
        process_ctx.notify_status("OnProfilingPrint")
        __profiling_print(context)
        process_ctx.notify_status("OnDynProfiling")
        context.dyn_prof_result = do_profiling(context, "dynamic")
        process_ctx.notify_status("OnCstProfiling")
        context.cst_prof_result = do_profiling(context, "const")
        process_ctx.notify_status("OnBinProfiling")
        context.bin_prof_result = do_profiling(context, "binary")
    process_ctx.notify_status("PostProfiling")
    passed = handle_profiling_result(context)
    process_ctx.notify_status("OnDumpOutputDataIfRequired")
    __dump_output(context)
    process_ctx.notify_status("OnComparison")
    __adapt_output_shape_unknown(context)
    compare_result = comparing(context.dyn_compile_result.kernel_name,
                               context.cst_compile_result.kernel_name,
                               context.bin_compile_result.kernel_name,
                               context.dyn_prof_result.output_bytes,
                               context.cst_prof_result.output_bytes,
                               context.bin_prof_result.output_bytes,
                               context.golden_arrays,
                               context.flat_output_dtypes,
                               standards=standards, third_parties=third_parties)
    if compare_result.passed != "PASS" and switches.dump_config.dump_on_fail:
        __dump_on_fail(context)
    process_ctx.notify_status("OnReturning")
    return_structure = ProfilingReturnStructure()
    return_structure.construct(context, compare_result, passed)
    __profiling_end_print(context, compare_result, passed)
    return return_structure


def __profiling_end_print(context: TestcaseOp,
                          compare_result: ComparisonResult, passed: str):
    c = compare_result
    logging.info("\n########################\n"
                 f"Performance result:                Comparison result:                 "
                 f"Memory check:\n"
                 f"DYN_PERF: {str(context.dyn_prof_result.cycle).ljust(24)} "
                 f"DYN_GOLD: {c.dyn_precision.ljust(24)} "
                 f"DYN_MEM: {str(context.dyn_prof_result.oob).ljust(24)}\n"
                 f"CST_PERF: {str(context.cst_prof_result.cycle).ljust(24)} "
                 f"CST_GOLD: {c.cst_precision.ljust(24)} "
                 f"CST_MEM: {str(context.cst_prof_result.oob).ljust(24)}\n"
                 f"BIN_PERF: {str(context.bin_prof_result.cycle).ljust(24)} "
                 f"BIN_GOLD: {c.bin_precision.ljust(24)} "
                 f"BIN_MEM: {str(context.bin_prof_result.oob).ljust(24)}\n"
                 f"STATUS: {passed.ljust(26)} PRECISION_STATUS: {c.passed}\n"
                 "########################\n")


def __compile_only_end_print(context: TestcaseOp):
    def __normalize(compile_time, tiling_time, tiling_key, block_dim, local_memory, kernel_name):
        if isinstance(tiling_time, (tuple, list)):
            tiling_time = numpy.median(tiling_time[1:]) if len(tiling_time) > 1 else tiling_time[0]
        return ("%.3f" % compile_time if isinstance(compile_time, float) else compile_time,
                tiling_time, tiling_key, block_dim, local_memory, kernel_name)

    dc = context.dyn_compile_result
    cc, bc = context.cst_compile_result, context.bin_compile_result
    lines = [("COMPILE", "Compile/s", "Tiling/s", "TilingKey", "BlockDim", "SimtUB/B", "KernelName"),
             ("DYN", *__normalize(dc.compile_time, dc.tiling_result.tiling_time,
                                  dc.tiling_key, dc.block_dim, dc.simt_ub_size, dc.kernel_name)),
             ("CST", *__normalize(cc.compile_time, "\\\\\\\\\\\\", "\\\\\\\\\\\\", cc.block_dim,
                                  cc.simt_ub_size, cc.kernel_name)),
             ("BIN", *__normalize(bc.compile_time, bc.tiling_result.tiling_time,
                                  bc.tiling_key, bc.block_dim, bc.simt_ub_size, bc.kernel_name))
             ]
    logging.info("\n" + table_print(lines))


def __parse_binary_tiling_data(context: TestcaseOp):
    # noinspection PyBroadException
    try:
        context.bin_tiling_data_bytes, context.bin_tuple_tiling_data = \
            parse_tiling_data(context.bin_compile_result.tiling_data)
        context.bin_str_tiling_data = \
            get_str_tiling_data(context.bin_tuple_tiling_data, context.bin_compile_result.compile_info,
                                context.bin_compile_result.tiling_key)
    except:
        logging.exception("Binary tiling data parsing failure")
        context.bin_compile_result.compile_result = "TILING_PARSE_FAILURE"


def __parse_dynamic_tiling_data(context: TestcaseOp):
    # noinspection PyBroadException
    try:
        context.dyn_tiling_data_bytes, context.dyn_tuple_tiling_data = \
            parse_tiling_data(context.dyn_compile_result.tiling_data)
        context.dyn_str_tiling_data = \
            get_str_tiling_data(context.dyn_tuple_tiling_data, context.dyn_compile_result.compile_info,
                                context.dyn_compile_result.tiling_key)
    except:
        logging.exception("Dynamic tiling data parsing failure")
        context.dyn_compile_result.compile_result = "TILING_PARSE_FAILURE"


def __parse_manual_params(context: TestcaseOp):
    switches = get_global_storage()

    def __apply_manual_block_dim(container):
        if container:
            context.dyn_compile_result.block_dim = get(container, 0) or context.dyn_compile_result.block_dim
            context.cst_compile_result.block_dim = get(container, 1) or context.cst_compile_result.block_dim
            context.bin_compile_result.block_dim = get(container, 2) or context.bin_compile_result.block_dim

    def __clear_atomic_if_not_none(idx, if_none):
        if switches.force_clear_atomic[idx] is not None:
            return switches.force_clear_atomic[idx]
        return if_none

    def __set_manual_simt_share_memory_size(idx: int, mode: str):
        if switches.force_simt_ub_size[idx] is not None:
            compile_result = getattr(context, f"{mode}_compile_result")
            compile_result.simt_ub_size = switches.force_simt_ub_size[idx]

    # command line should have the higher priority
    __apply_manual_block_dim(switches.force_block_dim)
    context.dyn_clear_atomic = __clear_atomic_if_not_none(0, context.dyn_clear_atomic)
    context.cst_clear_atomic = __clear_atomic_if_not_none(1, context.cst_clear_atomic)
    context.bin_clear_atomic = __clear_atomic_if_not_none(2, context.bin_clear_atomic)
    __set_manual_simt_share_memory_size(0, "dyn")
    __set_manual_simt_share_memory_size(1, "cst")
    __set_manual_simt_share_memory_size(2, "bin")


def __print_get_shape(golden):
    return golden.shape if hasattr(golden, 'shape') else golden


def __print_get_dtype(golden):
    return golden.dtype if hasattr(golden, 'dtype') else golden


def __profiling_print(context: TestcaseOp):
    if not (context.input_arrays or context.output_arrays):
        raise TypeError("Input or Output Data is not available, please check your custom function")
    flat_input_arrays = tuple(deep_flatten(context.input_arrays)) if context.input_arrays else ()
    logging.info(
        "\n====================================================================\n"
        "=======================================================\n"
        "==================================\n"
        f"Op Name: {context.op_name}\n"
        f"//////////// {get_global_storage().dyn_switches} //////////\n"
        f"Input Shape: {context.dyn_inputs}\n"
        f"Input Dtype: {tuple(str(ipt.dtype) if ipt is not None else None for ipt in flat_input_arrays)}\n"
        f"Input Actual Shape: {tuple(ipt.shape if ipt is not None else None for ipt in flat_input_arrays)}\n"
        f"Kernel Name: {context.dyn_compile_result.kernel_name}\n"
        f"Compilation Result: {context.dyn_compile_result.compile_result}\n"
        f"BlockDim: {context.dyn_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.dyn_compile_result.workspaces}\n"
        f"Tiling Data Parsed Dict: {context.dyn_str_tiling_data}\n"
        f"Tiling Data Parsed Tuple(int32): {context.dyn_tuple_tiling_data}\n"
        f"Tiling Key: {context.dyn_compile_result.tiling_key} ({context.str_tiling_key()})\n"
        f"Clear Atomic: {context.dyn_clear_atomic}\n"
        f"Simt UB: {context.dyn_compile_result.simt_ub_size} Bytes\n"
        f"//////////// {get_global_storage().cst_switches} //////////\n"
        f"Kernel Name: {context.cst_compile_result.kernel_name}\n"
        f"Compilation Result: {context.cst_compile_result.compile_result}\n"
        f"BlockDim: {context.cst_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.cst_compile_result.workspaces}\n"
        f"Clear Atomic: {context.cst_clear_atomic}\n"
        f"Simt UB: {context.cst_compile_result.simt_ub_size} Bytes\n"
        f"//////////// {get_global_storage().bin_switches} //////////\n"
        f"Kernel Name: {context.bin_compile_result.kernel_name}\n"
        f"Compilation Result: {context.bin_compile_result.compile_result}\n"
        f"BlockDim: {context.bin_compile_result.block_dim}\n"
        f"Workspace Bytes: {context.bin_compile_result.workspaces}\n"
        f"Tiling Data Parsed Dict: {context.bin_str_tiling_data}\n"
        f"Tiling Data Parsed Tuple(int32): {context.bin_tuple_tiling_data}\n"
        f"Tiling Key: {context.bin_compile_result.tiling_key} ({context.str_tiling_key(True)})\n"
        f"Clear Atomic: {context.bin_clear_atomic}\n"
        f"Simt UB: {context.bin_compile_result.simt_ub_size} Bytes\n"
        "////////////////////////////\n"
        f"Golden Shape: {tuple(__print_get_shape(ga) for ga in context.golden_arrays)}\n"
        f"Golden Dtype: {tuple(str(__print_get_dtype(ga)) for ga in context.golden_arrays)}\n"
        f"Output Shape: {context.output_shapes}\n"
        f"Output Dtype: {context.output_dtypes}\n"
        f"Input Data Range: {context.actual_input_data_ranges}\n"
        f"Precision Tolerance: {context.precision_tolerances}\n"
        f"Mode: {get_global_storage().mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {context.device_id}\n"
        "==================================\n"
        "=======================================================\n"
        "===================================================================="
    )


def __dump_to_file(data: Union[numpy.ndarray, bytes], file_name: str, dtype: Optional[str] = None):
    switches = get_global_storage()
    file_path = os.getenv("NPU_DUMP_PATH") or switches.root_path
    dump_to_file(data, file_path, file_name,
                 file_format=switches.dump_config.file_format,
                 dtype=dtype)


def __dump_input(context: TestcaseOp, force: bool = False) -> NoReturn:
    dump_input_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_input_enabled():
        logging.info("Dump Dynamic Input data....")
        for idx, _input in enumerate(deep_flatten(context.input_arrays or ())):
            __dump_to_file(_input, f"{dump_input_name}_dyn_input_{idx}")
        logging.info("Dump Tiling data....")
        __dump_to_file(context.dyn_tiling_data_bytes, f"{dump_input_name}_dyn_tiling_data")
        __dump_to_file(context.bin_tiling_data_bytes, f"{dump_input_name}_bin_tiling_data")


def __dump_output(context: TestcaseOp, force: bool = False) -> NoReturn:
    dump_output_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_output_enabled():
        output_dtypes = resolve_custom_numpy_dtypes(context.flat_output_dtypes)
        for typ in ("dyn", "cst", "bin"):
            logging.info(f"Dump {typ} Output data....")
            output_bytes = getattr(getattr(context, f"{typ}_prof_result"), "output_bytes")
            for idx, _output in enumerate(output_bytes):
                __dump_to_file(_output, f"{dump_output_name}_{typ}_output_{idx}", get(output_dtypes, idx))


def __dump_golden(context: TestcaseOp, force: bool = False) -> NoReturn:
    dump_output_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_golden_enabled():
        logging.info(f"Dump Golden data....")
        for idx, golden in enumerate(context.golden_arrays):
            __dump_to_file(golden, f"{dump_output_name}_golden_{idx}")


def __dump_on_fail(context: TestcaseOp) -> NoReturn:
    switches = get_global_storage()
    if not switches.dump_config.is_input_enabled():
        __dump_input(context, force=True)
    if not switches.dump_config.is_output_enabled():
        __dump_output(context, force=True)
    if not switches.dump_config.is_golden_enabled():
        __dump_golden(context, force=True)


def __construct_profiling_param(context: TestcaseOp, mode: str,
                                output_placeholder: bool = True) -> tuple:
    flat_input_arrays = tuple(deep_flatten(context.input_arrays or ()))
    if mode == "dynamic":
        return (context.dyn_compile_result,  # 0
                flat_input_arrays,  # 1
                context.output_arrays,  # 2
                context.dyn_workspace_arrays,  # 3
                context.dyn_tiling_data_bytes,  # 4
                output_placeholder,  # 5
                context.dyn_clear_atomic,  # 6
                get_global_storage().dyn_switches.prof,  # 7
                context.is_valid,  # 8
                context.fail_reason,  # 9
                context.tensor_list_distribution,  # 10
                context.testcase_name,  # 11
                mode)  # 12
    elif mode == "const":
        return (context.cst_compile_result,  # 0
                flat_input_arrays,  # 1
                context.output_arrays,  # 2
                context.cst_workspace_arrays,  # 3
                None,  # 4
                output_placeholder,  # 5
                context.cst_clear_atomic,  # 6
                get_global_storage().cst_switches.prof,  # 7
                context.is_valid,  # 8
                context.fail_reason,  # 9
                context.tensor_list_distribution,  # 10
                context.testcase_name,  # 11
                mode)  # 12
    elif mode == "binary":
        return (context.bin_compile_result,  # 0
                flat_input_arrays,  # 1
                context.output_arrays,  # 2
                context.bin_workspace_arrays,  # 3
                context.bin_tiling_data_bytes,  # 4
                output_placeholder,  # 5
                context.bin_clear_atomic,  # 6
                get_global_storage().bin_switches.prof,  # 7
                context.is_valid,  # 8
                context.fail_reason,  # 9
                context.tensor_list_distribution,  # 10
                context.testcase_name,  # 11
                mode)  # 12
    else:
        raise RuntimeError(f"Unknown profiling mode {mode}")


def _get_rts_interface(device_id: int, testcase_name: str, test_mode: str) -> RTSInterfaceBase:
    switches = get_global_storage()
    device: RTSInterface = get_process_context().storage.get("device", None)
    if not device or device.device_id is None:
        device = RTSInterface(switches.mode.is_model(),
                              short_soc_version=switches.short_soc_version)
        if device.is_model():
            device.set_simt_stack_size(switches.simt_cfg.dcu_stack,
                                       switches.simt_cfg.dvg_stack,
                                       device_id)
        device.set_device(device_id)
        device.set_float_overflow_mode(switches.overflow_mode)
        if not device.is_model():
            device.set_simt_stack_size(switches.simt_cfg.dcu_stack,
                                       switches.simt_cfg.dvg_stack)
        get_process_context().storage["device"] = device
    return device


def do_profiling(context: TestcaseOp, mode: str) -> RTSProfilingResult:
    """
    RTS Profiling wrapper
    """
    logging.debug("Entering profiling sequence with %s mode" % mode)
    switches = get_global_storage()
    op_name = context.op_name
    output_placeholder: bool = OpInfoKeeper().op_output_defined(op_name)
    param = RTSProfilingParam(*__construct_profiling_param(context, mode, output_placeholder))
    if param.switch:
        if not param.compile_result == "SUCC":
            result = RTSProfilingResult.fail(param.compile_result)
        elif not param.is_valid:
            result = RTSProfilingResult.fail(param.fail_reason)
        elif param.block_dim <= 0:
            result = RTSProfilingResult.fail("INVALID_TILING")
        else:
            param.clear_atomic_output_workspace()
            # noinspection PyBroadException
            try:
                device = _get_rts_interface(context.device_id, context.testcase_name, mode)
                result = rts_profiling(device, param)
            except:
                raise RuntimeError("Profiling Sequence of mode %s failed" % mode)
            finally:
                os.chdir(switches.root_path)
    else:
        result = RTSProfilingResult.fail("SUPPRESSED")
    return result


def __adapt_output_shape_unknown(context: TestcaseOp):
    """
    Detect and transfer npu output shape tensor from uint32 to uint64 encoding.
    """
    if not context.output_shape_unknown_indexes:
        return
    for x in ("dyn", "cst", "bin"):
        # translate to uint64 for all.
        # considering stc tbe & dyn asc different implement.
        x_output_bytes = getattr(
            getattr(context, f"{x}_prof_result"),
            "output_bytes"
        )
        if not x_output_bytes:
            continue
        last_array = x_output_bytes[-1]
        if isinstance(last_array, str):
            continue
        np_array = numpy.frombuffer(last_array, dtype=numpy.uint8)
        if np_array[3] == 128:
            # uint64 implement.
            pass
        else:
            # uint32 implement. use half bytes.
            np_array = np_array[:np_array.size//2]
            np_array = np_array.view(numpy.uint32).astype(numpy.uint64, copy=False)
            for i in range(np_array.size // 9):
                effective_num = np_array[i * 9]
                if effective_num <= 8:
                    for j in range(int(effective_num + 1), 9):
                        # uint64 cast view as uint32 will generate 0.
                        if np_array[i * 9 + j] == 0:
                            np_array[i * 9 + j] = 1
            # mark it as uint64 encoding. set uint64 highest bit as 1
            # SE says only the first one needs to be set.
            np_array.view(numpy.uint8)[3] = 128
            x_output_bytes[-1] = np_array


def handle_profiling_result(context: TestcaseOp):
    """
    Returns parsed cycle counts and passing state
    :param context:
    :return:
    """
    fake_fail = ("DYN_OFF", "CST_OFF", "BIN_OFF",
                 "DYN_UNSUPPORTED", "CST_UNSUPPORTED", "BIN_UNSUPPORTED",
                 "DYN_OPERATOR_NOT_FOUND",
                 "CST_OPERATOR_NOT_FOUND", "BIN_OPERATOR_NOT_FOUND",
                 "SUPPRESSED", "DYN_INPUT_MISSING")

    # noinspection PyBroadException
    def _get_cycle(result, off_flag):
        if isinstance(result.cycle, str):
            if result.cycle in off_flag:
                return "PASS"
            cs = result.cycle.split(',')
            if all([s.strip() == "OK" for s in cs]):  # in case  --task-prof=false
                return "PASS"
        elif get_global_storage().mode.is_model() and isinstance(result.cycle, (list, tuple)):
            if all([isinstance(s, int) and s > 0 for s in result.cycle]):
                return "PASS"

        _passed, _cycle_f = "EXCEPTION", "RTS_PROF_INVALID"
        try:
            _cycle_f = float(result.cycle)
            if _cycle_f <= 0:
                _cycle_f = "RTS_PROF_INVALID"
            else:
                _passed = "PASS"
        except:
            _cycle_f = "RTS_PROF_INVALID"
        finally:
            if _passed == "PASS":
                result.cycle = _cycle_f
            return _passed

    dyn_pass = _get_cycle(context.dyn_prof_result, fake_fail)
    cst_pass = _get_cycle(context.cst_prof_result, fake_fail)
    bin_pass = _get_cycle(context.bin_prof_result, fake_fail)

    passed = all([s == "PASS" for s in [dyn_pass, cst_pass, bin_pass]])
    return "PASS" if passed else "FAIL"
