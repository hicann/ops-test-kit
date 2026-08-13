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
Profiling method for Op Api testcases
"""

__all__ = ["profile_process"]


# Standard Packages
import contextlib
import csv
import ctypes
import os
import logging
import numpy
import pathlib
import shutil
import time
from typing import Tuple, Optional

try:
    from contextlib import nullcontext
except ImportError:
    print("Python version too low, from contextlib import nullcontext failed")

    @contextlib.contextmanager
    def NULLCXT():
        """NULL CONTEXT"""
        pass

    nullcontext = NULLCXT

# Third-Party Packages
from .input_generation import InputGenerator
from .golden_generation import GoldenGenerator
from .profiling_structure import ApiComparisonResult, ApiProfilingReturnStructure, ApiProfilingResult
from .comparison import Comparator
from ...manual_data import (
    load_manual_data_case,
    prepare_manual_data_store,
    snapshot_manual_values,
)
from ...testcase_manager import TestcaseAclnn
from ...tbe_multiprocessing import get_process_context, DeviceLock
from ...tbe_logging import build_single_log_dir, default_logging_config
from ...aclnn import AclInterface, OpApiInfoKeeper, OpApiInfo
from ...msprof import MsProfiler, TtkMsProfType
from ....utilities import get_global_storage, get, waiting_for_memory, frameless_table_print
from ....utilities import apply_as_list, resolve_custom_numpy_dtypes, dump_to_file, extract_plog_errors
from ....test_spec import get_spec_attr
from ..op.profiling_structure import _format_xpu_metrics


def __profiling_end_print(context: TestcaseAclnn, compare_result: ApiComparisonResult):
    c = compare_result
    logging.info(
        f"\n########################\nGOLD: {c.precision}\nPRECISION_STATUS: {c.passed}\n########################\n"
    )


def __print_get_shape(golden):
    if hasattr(golden, "shape"):
        return golden.shape
    return golden


def __print_get_dtype(golden):
    if hasattr(golden, "dtype"):
        return golden.dtype
    return golden


def __profiling_print(context: TestcaseAclnn, dev_id: int):
    attr_list = []
    for k, v in context.pure_attrs.items():
        attr_list.append(f"{k}: {v}")
    attrs = "\n".join(attr_list)
    scalars = context.scalars or ()
    scalar_values = []
    for group in scalars:
        if isinstance(group, (list, tuple)):
            scalar_values.append(tuple(s.item() if s is not None else None for s in group))
        else:
            scalar_values.append(group.item() if group is not None else None)
    scalar_values = tuple(scalar_values)
    logging.info(
        "\n"
        "====================================================================\n"
        "=======================================================\n"
        "==================================\n"
        f"Op Api Name: {context.api_name}\n"
        f"////////////////// Tensors //////////////\n"
        f"View Shape: {context.tensor_view_shapes}\n"
        f"View Stride: {context.tensor_view_strides}\n"
        f"View Offset: {context.tensor_view_offsets}\n"
        f"Storage Shape: {context.tensor_storage_shapes}\n"
        f"Format: {context.tensor_formats}\n"
        f"Dtype: {context.tensor_dtypes}\n"
        f"////////////////// Scalars //////////////\n"
        f"Value: {scalar_values}\n"
        f"Dtype: {context.scalar_dtypes}\n"
        f"//////////// Attributes/Arrays //////////\n"
        f"{attrs}\n"
        f"/////////////////////////////////////////\n"
        f"Output Shape: {context.flat_output_view_shapes}\n"
        f"Output Dtype: {context.flat_output_dtypes}\n"
        f"Tensor Data Range: {context.actual_input_data_ranges}\n"
        f"Scalar Data Range: {context.full_scalar_data_ranges}\n"
        f"Precision Tolerance: {context.precision_tolerances}\n"
        f"Mode: {get_global_storage().mode.name}\n"
        f"PID: {os.getpid()}\n"
        f"Device: {dev_id}\n"
        "==================================\n"
        "=======================================================\n"
        "===================================================================="
    )


def prof_end(context, print_content):
    compare_result = ApiComparisonResult(print_content)
    return_structure = ApiProfilingReturnStructure()
    return_structure.construct(context, compare_result)
    __profiling_end_print(context, compare_result)
    return return_structure


def __get_aclnn_device(dev_id: int) -> AclInterface:
    device: AclInterface = get_process_context().storage.get("device", None)
    switches = get_global_storage()
    if not device:
        device = AclInterface(switches.short_soc_version, switches.mode.is_model())
        get_process_context().storage["device"] = device
    if device.device_id is None:
        device.set_device(dev_id)
        device.set_float_overflow_mode(switches.overflow_mode)
    return device


class Phase1ParamBuilder:
    """
    Builder for aclnn Phase-1 parameters.
    """

    CBaseTypeMap = {
        # bool / int8_t / int / int32_t / float
        # int64_t / uint64_t / double
        "bool": ctypes.c_bool,
        "int8_t": ctypes.c_int8,
        "int": ctypes.c_int,
        "int32_t": ctypes.c_int32,
        "float": ctypes.c_float,
        "int64_t": ctypes.c_int64,
        "uint64_t": ctypes.c_uint64,
        "double": ctypes.c_double,
    }

    def __init__(self, context: TestcaseAclnn, device: AclInterface):
        self._ctx = context
        self._dvc = device
        self._flatten_acl_tensor: tuple = ()

    def build(self):
        acl_tensors = self._create_acl_tensor()
        acl_scalars = self._create_acl_scalar()
        case_params = []
        plan = self._ctx.get_param_plan()
        for kind, param_name, acl_type, default in plan.param_layout:
            if kind == "tensor":
                case_params.append(acl_tensors.pop(0))
            elif kind == "scalar":
                case_params.append(acl_scalars.pop(0))
            else:
                # consider remaining as attribute.
                val = self._ctx.attributes.get(param_name, default)
                if "Array" in acl_type:
                    # aclBoolArray/aclIntArray/aclFloatArray
                    typ = acl_type[3 : acl_type.index("Array")]
                    if val is None:
                        case_params.append(None)
                    else:
                        case_params.append(self._dvc.create_acl_array(val, typ))
                elif acl_type == "aclDataType":
                    # int32
                    case_params.append(ctypes.c_int32(val))
                elif acl_type == "char*":
                    val_bytes = val.encode("UTF-8")
                    case_params.append(ctypes.c_char_p(val_bytes))
                else:
                    # Base c types:
                    # bool / int8_t / int / int32_t / float
                    # int64_t / uint64_t / double
                    c_type = self.CBaseTypeMap[acl_type]
                    case_params.append(c_type(val))
        return case_params

    def copy_output_from_hbm(self) -> list:
        output_byte_arrays = []
        dist = self._ctx.tensor_list_dist
        for nested_idx in self._ctx.output_tensor_indexes:
            flat_idx = sum(max(d, 1) for d in dist[:nested_idx]) if dist else nested_idx
            acl_tensor: ctypes.c_void_p = self._flatten_acl_tensor[flat_idx]
            tensor = self._ctx.flatten_tensors[flat_idx]
            if acl_tensor is None:
                output_byte_arrays.append(None)
            else:
                npu_ptr = self._dvc.get_device_mem_addr(acl_tensor)
                if isinstance(tensor, numpy.ndarray):
                    np_storage = tensor
                    while np_storage.base is not None:
                        if isinstance(np_storage.base, numpy.ndarray):
                            np_storage = np_storage.base
                            break
                        if hasattr(np_storage.base, "__array_interface__"):
                            np_storage = numpy.asarray(np_storage.base)
                            break
                        np_storage = np_storage.base
                    byte_size = np_storage.nbytes
                else:
                    byte_size = tensor.storage().nbytes()
                output_byte_arrays.append(self._dvc.get_data_from_hbm(npu_ptr, byte_size))
        return output_byte_arrays

    def collect_output_view_shapes(self) -> list:
        output_view_shapes = []
        dist = self._ctx.tensor_list_dist
        for nested_idx in self._ctx.output_tensor_indexes:
            flat_idx = sum(max(d, 1) for d in dist[:nested_idx]) if dist else nested_idx
            acl_tensor: ctypes.c_void_p = self._flatten_acl_tensor[flat_idx]
            if acl_tensor is None:
                output_view_shapes.append(None)
            else:
                output_view_shapes.append(self._dvc.get_view_shape(acl_tensor))
        return output_view_shapes

    def cleanup(self):
        self._dvc.free_all_memory()

    def _create_acl_tensor(self):
        ptr_lst = []
        for idx, tt in enumerate(self._ctx.flatten_tensors):
            if tt is None:
                ptr_lst.append(None)
            else:
                fmt = get(self._ctx.flat_tensor_formats, idx)
                ss = self._ctx.flat_storage_shape(idx)
                ptr_lst.append(self._dvc.create_acl_tensor(tt, fmt, ss))
        self._flatten_acl_tensor = tuple(ptr_lst)
        dist = self._ctx.tensor_list_dist
        if not dist:
            return ptr_lst

        ptr_lst_lst = []
        as_list = apply_as_list(ptr_lst, dist)
        for maybe_lst in as_list:
            if not isinstance(maybe_lst, (list, tuple)):
                ptr_lst_lst.append(maybe_lst)
            else:
                ptr_lst_lst.append(self._dvc.create_acl_tensor_list(maybe_lst))
        return ptr_lst_lst

    def _create_acl_scalar(self):
        ptr_lst = []
        for s in self._ctx.flatten_scalars:
            if s is None:
                ptr_lst.append(None)
            else:
                ptr_lst.append(self._dvc.create_acl_scalar(s))
        if not self._ctx.scalar_list_dist:
            return ptr_lst

        ptr_lst_lst = []
        as_list = apply_as_list(ptr_lst, self._ctx.scalar_list_dist)
        for maybe_lst in as_list:
            if not isinstance(maybe_lst, (list, tuple)):
                ptr_lst_lst.append(maybe_lst)
            else:
                ptr_lst_lst.append(self._dvc.create_acl_scalar_list(maybe_lst))
        return ptr_lst_lst


class AclOpExecutor:
    def __init__(self, context: TestcaseAclnn, device: AclInterface):
        self._switches = get_global_storage()
        self._phase1_param_builder = Phase1ParamBuilder(context, device)
        self._run_time = self._switches.run_time
        self._ctx = context
        self._dvc = device
        self._prof_type = TtkMsProfType.API if self._switches.TASK_PROFILING else TtkMsProfType.NONE
        self._prof_result_path = os.path.join(self._switches.root_path, "msprof", "op_api", self._ctx.testcase_name)

    @contextlib.contextmanager
    def rts_context(self):
        self._dvc.create_context()
        if self._switches.deterministic_level >= 1:
            self._dvc.set_deterministic_level(self._switches.deterministic_level)
        try:
            yield
        finally:
            self._dvc.destroy_context()
            if self._dvc.is_model():
                self._dvc.reset()

    @contextlib.contextmanager
    def rts_stream(self):
        if self._dvc.is_model():
            yield None
        else:
            stream = self._dvc.create_stream()
            try:
                yield stream
            finally:
                self._dvc.destroy_stream_force(stream)

    def do(self):
        with self.rts_context():
            # Model warmup
            self._dvc.warmup(self._switches)
            with self.rts_stream() as stm:
                output_byte_arrays, output_view_shapes, success, det_status = self._acl_sequence(stm)
            # Cycle Analysis
            if self._dvc.is_model():
                # TODO
                logging.info("TODOOOOOOOOOOOOOOOOOOOOOO")
                api_prof = "UNKNOWN"
                op_prof = "TOTAL_CYCLE_TODO"
            else:
                api_prof, op_prof = self._process_total_cycles()
            return ApiProfilingResult(
                success, api_prof, op_prof, output_byte_arrays, output_view_shapes, deterministic_status=det_status
            )

    @staticmethod
    def _extract_csv_cell(filename, extract_cols, cmp=None) -> list:
        results = []
        with open(filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if cmp is None or cmp(row):
                    data_dict = {}
                    for idx, c in enumerate(extract_cols):
                        try:
                            data_dict.update({c: float(row[c])})
                        except ValueError:
                            data_dict.update({c: row[c]})
                    results.append(data_dict)
        return results

    def _acl_sequence(self, stream: Optional[ctypes.c_void_p] = None):
        # Profiling Preparation
        output_byte_arrays = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        output_view_shapes = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        status = "NOK"
        deterministic = self._switches.deterministic_level == 1
        md5_list = []
        det_status = None
        try:
            prof_start_at = 1 if self._run_time > 1 else 0
            with MsProfiler(
                self._dvc.device_id,
                result_path=self._prof_result_path,
                ttk_prof_type=self._prof_type,
                start_step=prof_start_at,
                is_model=self._dvc.is_model(),
            ) as profiler:
                for repeat_idx in range(self._run_time):
                    profiler.step()
                    self._dvc.clear_l1(self._switches)
                    self._dvc.clear_ub(self._switches)
                    # self._dvc.test_clear_ub(self._switches)
                    # construct phase 1 params
                    phase1_params = self._phase1_param_builder.build()
                    # call L2 interface.
                    try:
                        # call phase 1 interface
                        workspace_size, c_executor = self._dvc.acl_get_workspace(self._ctx.api_name, phase1_params)
                        # call phase 2 interface
                        status = self._dvc.acl_execute(self._ctx.api_name, workspace_size, c_executor, stream)
                    except Exception as e:
                        time.sleep(0.5)
                        plog_errors = extract_plog_errors()
                        if plog_errors:
                            logging.error(
                                f"aclnn interface {self._ctx.api_name} execute failed: \n"
                                f"***************************************************************************\n"
                                f"{os.linesep.join(plog_errors)}\n"
                                f"***************************************************************************"
                            )
                        else:
                            error_detail = str(e)
                            logging.exception(f"aclnn interface {self._ctx.api_name} execute failed:\n{error_detail}")
                        status = "ACLNN_EXECUTE_FAILED"

                    if status == "OK":
                        if deterministic > 0:
                            out_bytes = self._phase1_param_builder.copy_output_from_hbm()
                            import hashlib

                            md5_list.append(
                                hashlib.md5(
                                    b"".join(
                                        bytes(b) if isinstance(b, (bytearray, memoryview)) else b for b in out_bytes
                                    )
                                ).hexdigest()
                            )
                            if repeat_idx == self._run_time - 1:
                                output_byte_arrays = out_bytes
                                output_view_shapes = self._phase1_param_builder.collect_output_view_shapes()
                        elif repeat_idx == self._run_time - 1:
                            # copy output (tensor storage data) from device
                            output_byte_arrays = self._phase1_param_builder.copy_output_from_hbm()
                            output_view_shapes = self._phase1_param_builder.collect_output_view_shapes()
                    self._dvc.free_all_memory()
                    if status != "OK":
                        break
        finally:
            self._dvc.free_all_memory()

        if deterministic > 0 and len(md5_list) > 1:
            if len(set(md5_list)) != 1:
                logging.error(f"[{self._ctx.testcase_name}] MD5 mismatch across {len(md5_list)} runs: {md5_list}")
                status = "DETERMINISTIC_MD5_MISMATCH"
                det_status = "FAIL"
            else:
                logging.info(f"[{self._ctx.testcase_name}] MD5 consistent across {len(md5_list)} runs: {md5_list[0]}")
                det_status = "PASS"
        elif deterministic > 0 and len(md5_list) == 1:
            det_status = "PASS"

        return output_byte_arrays, output_view_shapes, status == "OK", det_status

    def _process_total_cycles(self):
        """
        analysis profiling data & print to stdout.
        """
        api_prof = "UNKNOWN"
        op_prof = "UNKNOWN"
        prof_result_path = pathlib.Path(self._prof_result_path)
        if not prof_result_path.is_dir():
            return api_prof, op_prof
        csv_files = list(prof_result_path.glob("**/*.csv"))
        if not csv_files:
            return api_prof, op_prof
        for item in pathlib.Path(os.path.dirname(csv_files[0])).iterdir():
            if item.is_file():
                shutil.copy(item, prof_result_path.joinpath(item.name))
        for item in prof_result_path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            elif item.name.startswith("api_statistic_"):
                api_prof = self._extract_csv_cell(
                    item,
                    ("API Name", "Time(us)", "Count", "Avg(us)", "Min(us)", "Max(us)"),
                    cmp=lambda row: row["Level"] == "acl",
                )
            elif item.name.startswith("op_statistic_"):
                op_prof = self._extract_csv_cell(
                    item,
                    ("OP Type", "Total Time(us)", "Count", "Avg Time(us)", "Min Time(us)", "Max Time(us)", "Core Type"),
                )
                for p in op_prof:
                    p["OP Type"] = p["OP Type"] + ("_AiCpu" if "cpu" in p["Core Type"].lower() else "_AiCore")
                    del p["Core Type"]
        # print
        lines = [["Name", "Total/us", "Avg/us", "Min/us", "Max/us", "# of Calls"]]
        if not isinstance(api_prof, str):
            lines.extend(
                [
                    [
                        item["API Name"],
                        "%.2f" % item["Time(us)"],
                        "%.2f" % item["Avg(us)"],
                        "%.2f" % item["Min(us)"],
                        "%.2f" % item["Max(us)"],
                        int(item["Count"]),
                    ]
                    for item in api_prof
                ]
            )
        if not isinstance(op_prof, str):
            lines.extend(
                [
                    [
                        item["OP Type"],
                        "%.2f" % item["Total Time(us)"],
                        "%.2f" % item["Avg Time(us)"],
                        "%.2f" % item["Min Time(us)"],
                        "%.2f" % item["Max Time(us)"],
                        int(item["Count"]),
                    ]
                    for item in op_prof
                ]
            )
        logging.info(frameless_table_print(lines))
        return api_prof, op_prof


def do_profiling(context: TestcaseAclnn, dev_id: int) -> ApiProfilingResult:
    """
    Profiling wrapper
    """
    logging.debug("Entering op api profiling sequence")
    switches = get_global_storage()
    if not context.is_valid:
        return ApiProfilingResult.fail(context.fail_reason)
    else:
        # noinspection PyBroadException
        try:
            device = __get_aclnn_device(dev_id)
            return AclOpExecutor(context, device).do()
        except:
            raise RuntimeError("Profiling Sequence of mode %s failed" % switches.mode)
        finally:
            os.chdir(switches.root_path)


def _aclnn_xpu_inputs(context: TestcaseAclnn) -> list:
    """np_storages re-nested, pure outputs filtered → top-level input slots for XPU.

    np_storages is flat (one per flat tensor index). Re-nest by tensor_list_dist
    to get top-level slots, then skip pure_output_indexes (inplace outputs kept).
    """
    dist = context.tensor_list_dist
    nested = apply_as_list(list(context.np_storages), dist) if dist else list(context.np_storages)
    inputs = []
    real_idx = 0
    for slot in nested:
        if real_idx not in context.pure_output_indexes:
            inputs.append(slot)
        real_idx += len(slot) if isinstance(slot, (list, tuple)) else 1
    return inputs


def _aclnn_xpu_input_names(context: TestcaseAclnn) -> list:
    """Input tensor param names (pure outputs filtered) for XPU schema."""
    op_api_info = OpApiInfoKeeper().info_of(context.api_name)
    if op_api_info is None:
        return []
    return [n for i, n in enumerate(op_api_info.tensors) if i not in context.pure_output_indexes]


def _aclnn_param_order(context: TestcaseAclnn) -> list:
    """C header param order (excluding pure outputs) for server-side pool merge."""
    op_api_info = OpApiInfoKeeper().info_of(context.api_name)
    if op_api_info is None:
        return []
    pure_output_names = {op_api_info.tensors[i]
                         for i in context.pure_output_indexes
                         if i < len(op_api_info.tensors)}
    return [name for name in op_api_info.params
            if name not in pure_output_names and name not in ("workspaceSize", "executor")]


def __dump_to_file(data, file_name: str, dtype: Optional[str] = None):
    switches = get_global_storage()
    file_path = os.getenv("NPU_DUMP_PATH") or switches.root_path
    dump_to_file(data, file_path, file_name, file_format=switches.dump_config.file_format, dtype=dtype)


def __dump_input(context: TestcaseAclnn, force: bool = False):
    dump_input_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_input_enabled():
        logging.info("Dump Input Tensor data....")
        for idx, t in enumerate(context.flatten_tensors):
            if idx in context.pure_output_indexes:
                continue
            __dump_to_file(t, f"{dump_input_name}_input_tensor_{idx}")


def __dump_output(context: TestcaseAclnn, force: bool = False):
    dump_output_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_output_enabled():
        output_dtypes = resolve_custom_numpy_dtypes(context.flat_output_dtypes)
        logging.info(f"Dump Output data....")
        output_bytes = context.prof_result.output_bytes
        for idx, _output in enumerate(output_bytes):
            __dump_to_file(_output, f"{dump_output_name}_output_{idx}", get(output_dtypes, idx))


def __dump_golden(context: TestcaseAclnn, force: bool = False):
    dump_golden_name = context.dump_file_prefix or context.testcase_name
    if force or get_global_storage().dump_config.is_golden_enabled():
        logging.info(f"Dump Golden data....")
        for idx, golden in enumerate(context.golden_tensors):
            __dump_to_file(golden, f"{dump_golden_name}_golden_{idx}")


def __dump_on_fail(context: TestcaseAclnn):
    switches = get_global_storage()
    if not switches.dump_config.is_input_enabled():
        __dump_input(context, force=True)
    if not switches.dump_config.is_output_enabled():
        __dump_output(context, force=True)
    if not switches.dump_config.is_golden_enabled():
        __dump_golden(context, force=True)


def profile_process(context: TestcaseAclnn, device_grant_events: dict, device_granted_indices: dict, dev_id: int):
    """
    Op Api Testcase Profiling Entrance
    """
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(context.testcase_name)
    if switches.single_testcase_log_mode:
        _log_dir = build_single_log_dir(switches.test_mode, context.api_name, switches.root_path)
        default_logging_config(file_handler=switches.logging_to_file, testcase_name=context.testcase_name, log_dir=_log_dir)
    process_ctx.notify_status("OnParseParameters")
    ####################
    # Check whether there is need to do further test
    ####################
    if not context.is_valid:
        return prof_end(context, context.fail_reason)
    if not OpApiInfoKeeper().has_api(context.api_name):
        return prof_end(context, "OP_API_NOT_FOUND")
    if not switches.no_memory_check:
        process_ctx.notify_status("OnWaitingForMemory")
        waiting_for_memory()
    logging.debug(f"Expecting {context.tensor_bytes} bytes memory usage")

    manual_mode = getattr(switches, "manual_data_mode", None)
    manual_case = None
    try:
        prepare_store = prepare_manual_data_store(context, "aclnn", switches)
    except Exception as exc:
        logging.exception("Manual data preparation failure")
        return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
    if manual_mode != "prepare":
        try:
            manual_case = load_manual_data_case(
                context,
                "aclnn",
                switches,
                before_load=lambda: process_ctx.notify_status("OnLoadManualData"),
            )
            if manual_case is not None:
                manual_mode = "replay"
        except Exception as exc:
            logging.exception("Manual data loading failure")
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")

    process_ctx.notify_status("OnGenInput")
    # noinspection PyBroadException
    try:
        input_generator = InputGenerator(context)
        if manual_case is not None:
            input_generator.gen(
                stored_inputs=manual_case.inputs,
                stored_scalars=manual_case.scalars,
            )
        else:
            input_generator.gen()
    except Exception as exc:
        logging.exception("Input data generation failure:")
        if manual_case is not None:
            return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
        return prof_end(context, "INPUT_GEN_FAILURE")

    if manual_mode != "prepare":
        process_ctx.notify_status("OnDumpInputDataIfRequired")
        __dump_input(context)

    if manual_mode == "prepare":
        try:
            prepared_inputs = snapshot_manual_values(context.np_storages, "input")
            prepared_scalars = snapshot_manual_values(context.flatten_scalars or (), "scalar")
            process_ctx.notify_status("OnGenGolden")
            GoldenGenerator(context).gen()
            process_ctx.notify_status("OnWriteManualData")
            case_dir = prepare_store.write_case(
                context,
                "aclnn",
                prepared_inputs,
                context.golden_tensors,
                scalars=prepared_scalars,
                file_format=switches.dump_config.file_format,
            )
        except Exception as exc:
            logging.exception("Manual data preparation failure")
            return prof_end(context, f"MANUAL_DATA_PREPARE_FAILURE: {exc}")
        logging.info(f"[{context.testcase_name}] manual data prepared: {case_dir}")
        compare_result = ApiComparisonResult(None).set("MANUAL_DATA_PREPARED", "PASS")
        return_structure = ApiProfilingReturnStructure()
        return_structure.construct(context, compare_result)
        __profiling_end_print(context, compare_result)
        return return_structure

    # Following actions need to acquire global lock
    process_ctx.notify_status("OnAcquireLock")
    use_device = switches.mode.use_device()
    with DeviceLock(
        process_ctx,
        dev_id,
        use_device=use_device,
        grant_event=device_grant_events.get(dev_id),
        granted_idx=device_granted_indices.get(dev_id),
    ):
        process_ctx.notify_status("OnProfilingPrint")
        __profiling_print(context, dev_id)
        process_ctx.notify_status("OnProfiling")
        if get_global_storage().backend == "npusim":
            from ttk.core_modules.simulator import run_aclnn_sim

            context.prof_result = run_aclnn_sim(context, dev_id)
        else:
            context.prof_result = do_profiling(context, dev_id)
    standards = None
    need_3party = False
    if context.prof_result.failed():
        context.golden_tensors = context.prof_result.api_prof
    else:
        from ...comparison.resolve import resolve_tolerance as _resolve_tolerance

        plugin_path = getattr(switches, "plugin_path", None)
        tolerance = get_spec_attr(context.api_name, "tolerance", plugin_path)
        output_dtypes = resolve_custom_numpy_dtypes(context.flat_output_dtypes)
        standards = _resolve_tolerance(
            tolerance,
            context.flat_precision_tolerances,
            context.flat_absolute_precision,
            output_dtypes,
            switches.compare_method,
        )
        need_3party = any(s.token == "cross_check" for s in standards)
        if need_3party:
            context.golden_mode_override = "Promote"
        try:
            if manual_case is not None:
                try:
                    process_ctx.notify_status("OnLoadManualGolden")
                    context.golden_tensors = manual_case.load_goldens(
                        shapes=context.prof_result.output_view_shapes,
                        dtypes=context.flat_output_dtypes,
                    )
                    __dump_golden(context)
                except Exception as exc:
                    logging.exception("Manual golden loading failure")
                    return prof_end(context, f"MANUAL_DATA_READ_FAILURE: {exc}")
            else:
                process_ctx.notify_status("OnGenGolden")
                # noinspection PyBroadException
                try:
                    GoldenGenerator(context).gen()
                    process_ctx.notify_status("OnDumpGoldenDataIfRequired")
                    __dump_golden(context)
                except:
                    logging.exception("Golden data generation failure")
        finally:
            if hasattr(context, "golden_mode_override"):
                del context.golden_mode_override
    process_ctx.notify_status("OnDumpOutputDataIfRequired")
    __dump_output(context)
    third_parties = None
    xpu_results = None
    if not context.prof_result.failed():
        from ttk.remote.client import xpu_mode_of, collect_third_party

        xpu_mode = xpu_mode_of(switches, need_3party)
        if xpu_mode:
            process_ctx.notify_status("OnXpuProfiling")
            _, third_parties, xpu_results = collect_third_party(
                op_name=context.api_name,
                inputs=_aclnn_xpu_inputs(context),
                input_names=_aclnn_xpu_input_names(context),
                op_type=None,
                attributes=context.xpu_attrs,
                testcase_name=context.testcase_name,
                switches=switches,
                need_data=need_3party,
                param_order=_aclnn_param_order(context),
            )
            if need_3party and third_parties is None:
                logging.warning(
                    "[%s] cross_check configured but no third_party output "
                    "(no XPU / endpoint down); cross_check outputs will GOLDEN_FAILURE",
                    context.testcase_name,
                )
    context.xpu_metrics = _format_xpu_metrics(xpu_results) if xpu_results else {}
    process_ctx.notify_status("OnComparison")
    compare_result = Comparator(context, standards, third_parties).compare()
    if compare_result.passed != "PASS" and switches.dump_config.dump_on_fail:
        __dump_on_fail(context)
    process_ctx.notify_status("OnReturning")
    return_structure = ApiProfilingReturnStructure()
    return_structure.construct(context, compare_result)
    return_structure.deterministic_status = getattr(context.prof_result, "deterministic_status", None)
    __profiling_end_print(context, compare_result)
    return return_structure
