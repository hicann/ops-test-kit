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
import copy
import csv
import ctypes
import math
import os
import logging
import numpy
import pathlib
import shutil
import tempfile
import threading
import time
from typing import Tuple, Optional, Dict, List

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
from ...tbe_multiprocessing import MultiDeviceLock
from ...aclnn import AclInterface, OpApiInfoKeeper, OpApiInfo
from ...msprof import MsProfiler, TtkMsProfType
from ...npu_preprocess import invoke_npu_preprocess, resolve_npu_preprocess
from ....utilities import get_global_storage, get, waiting_for_memory, frameless_table_print, get_dtype_width
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
                    declared_dtype = get(self._ctx.flat_tensor_dtypes, flat_idx, None)
                    effective_dtype = (
                        tensor.dtype
                        if ("float4" in str(declared_dtype) or "int4" in str(declared_dtype))
                        else np_storage.dtype
                    )
                    byte_size = int(math.ceil(np_storage.size * get_dtype_width(effective_dtype)))
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
        # fp4 等子字节 dtype：CSV view_shape 是 unpacked 元素数，np_storage 是 packed uint8。
        # aclCreateTensor 需要 unpacked view_shape + 子字节 dtype（对齐 mc2_test）。
        csv_view_shapes = self._ctx.flat_tensor_view_shapes
        csv_dtypes = self._ctx.flat_tensor_dtypes
        from ttk.utilities.dtypes import DATA_TYPE_DICT
        for idx, tt in enumerate(self._ctx.flatten_tensors):
            if tt is None:
                ptr_lst.append(None)
            else:
                fmt = get(self._ctx.flat_tensor_formats, idx)
                ss = self._ctx.flat_storage_shape(idx)
                view_override = None
                dtype_override = None
                csv_dt = get(csv_dtypes, idx)
                csv_dt_name = str(csv_dt) if csv_dt else ''
                if 'float4' in csv_dt_name or 'fp4' in csv_dt_name:
                    csv_vs = get(csv_view_shapes, idx)
                    if csv_vs:
                        view_override = tuple(csv_vs)
                    dt_key = csv_dt_name.split('.')[-1] if '.' in csv_dt_name else csv_dt_name
                    dt_norm = 'float4_e2m1' if 'e2m1' in dt_key else ('float4_e1m2' if 'e1m2' in dt_key else dt_key)
                    if dt_norm in DATA_TYPE_DICT:
                        dtype_override = DATA_TYPE_DICT[dt_norm]
                ptr_lst.append(self._dvc.create_acl_tensor(tt, fmt, ss,
                                                           view_shape_override=view_override,
                                                           acl_dtype_override=dtype_override))
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

    def do(self, stream: ctypes.c_void_p = None, skip_context_creation: bool = False):
        if skip_context_creation and stream is not None:
            output_byte_arrays, output_view_shapes, success, _ = self._acl_sequence(
                stream, skip_profiler=True)
            return ApiProfilingResult(success, "UNKNOWN", "UNKNOWN",
                                      output_byte_arrays, output_view_shapes)
        with self.rts_context():
            self._dvc.warmup(self._switches)
            with self.rts_stream() as stm:
                output_byte_arrays, output_view_shapes, success, det_status, npu_memory = self._acl_sequence(stm)
            # Cycle Analysis
            if self._dvc.is_model():
                api_prof = "UNKNOWN"
                op_prof = "TOTAL_CYCLE_TODO"
            else:
                api_prof, op_prof = self._process_total_cycles()
            return ApiProfilingResult(
                success, api_prof, op_prof, output_byte_arrays, output_view_shapes,
                deterministic_status=det_status, npu_memory=npu_memory,
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

    def _acl_sequence(self, stream: Optional[ctypes.c_void_p] = None,
                      skip_profiler: bool = False):
        output_byte_arrays = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        output_view_shapes = ["NO_OUTPUT"] * len(self._ctx.output_tensor_indexes)
        status = "NOK"
        deterministic = self._switches.deterministic_level == 1
        md5_list = []
        det_status = None
        npu_memory = None
        try:
            prof_start_at = 1 if self._run_time > 1 else 0
            profiler_ctx = MsProfiler(
                self._dvc.device_id,
                result_path=self._prof_result_path,
                ttk_prof_type=self._prof_type,
                start_step=prof_start_at,
                is_model=self._dvc.is_model(),
            ) if not skip_profiler else nullcontext()
            with profiler_ctx as profiler:
                for repeat_idx in range(self._run_time):
                    if not skip_profiler and profiler:
                        profiler.step()
                    self._dvc.clear_l1(self._switches)
                    self._dvc.clear_ub(self._switches)
                    logging.debug(f"[AclOpExecutor dev={self._dvc.device_id}] building phase1 params, "
                                  f"group={self._ctx.attributes.get('group', 'N/A')}")
                    phase1_params = self._phase1_param_builder.build()
                    logging.debug(f"[AclOpExecutor dev={self._dvc.device_id}] calling acl_get_workspace")
                    try:
                        # call phase 1 interface
                        workspace_size, c_executor = self._dvc.acl_get_workspace(self._ctx.api_name, phase1_params)
                        npu_memory = workspace_size
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

        return output_byte_arrays, output_view_shapes, status == "OK", det_status, npu_memory

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
                try:
                    shutil.copy(item, prof_result_path.joinpath(item.name))
                except shutil.SameFileError:
                    pass
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
    pure_output_names = {op_api_info.tensors[i] for i in context.pure_output_indexes if i < len(op_api_info.tensors)}
    return [
        name
        for name in op_api_info.params
        if name not in pure_output_names and name not in ("workspaceSize", "executor")
    ]


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


from .mc2_golden import (__golden_multi_device_compare,
                          get_gmm_exp_token_nums,
                          generate_gmm_alltoallv_matrix,
                           patch_gmm_rank_attributes,
                           patch_gmm_weight_transpose)


def __regenerate_v2_mxfp_inputs(thread_ctx, rank_idx: int, base_seed: int):
    """For aclnnAllGatherMatmulV2 mxfp cases, regenerate correlated x1+x1scale
    and x2+x2scale via mx_quantize so NPU tiling succeeds.

    Detection: x1scale (slot 3) dtype is fp8_e8m0.
    For per_tensor/per_block (slot 3 dtype is fp32), only x2 and x2scale need
    to be shared across ranks (weight_same semantics); InputGenerator uses
    per_rank_seed which makes them differ, breaking the NPU assumption that
    x2 is a shared weight. Regenerate x2/x2scale with a rank-independent seed.
    For non-quant (slot 3 is None), no action needed.

    Layout convention (matches non-mxfp trans_b=1): x2 view = (K, N) contiguous,
    x2scale = (K/64, N, 2) contiguous. aclnn op_api does not set is_trans_b attr,
    so op always sees is_trans_b=false; tiling CheckMXFPScaleInput false branch
    requires x2scale=[ceilK, N, 2].
    """
    flat_dtypes = thread_ctx.flat_tensor_dtypes or []
    if len(flat_dtypes) < 5:
        return
    x1s_dtype = flat_dtypes[3]
    x1_idx, x2_idx, x1s_idx, x2s_idx = 0, 1, 3, 4
    if len(thread_ctx.flatten_tensors) <= x2s_idx:
        return

    # --- non-quant path: x2 already handled by weight_shared regeneration ---
    # Only act on quant cases (x1 is fp8/hif8).
    x1_dtype_str = str(flat_dtypes[0]) if flat_dtypes[0] else ''
    is_quant_x1 = any(d in x1_dtype_str for d in ('float8', 'hifloat8', 'fp8', 'hif8', 'float4', 'fp4'))
    if not is_quant_x1:
        return

    if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
        thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
    else:
        thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)

    # --- per_tensor / per_block path: share x2 + x2scale across ranks ---
    if x1s_dtype not in ('fp8_e8m0', 'float8_e8m0'):
        # x1scale/x2scale are fp32 scalar/block (not e8m0). InputGenerator used
        # per_rank_seed so x2/x1scale/x2scale differ per rank, but NPU
        # all_gather_matmul expects x2 and both scales to be shared (weight).
        # Regenerate with rank-independent seed.
        if thread_ctx.flatten_tensors[x2_idx] is None or thread_ctx.flatten_tensors[x2s_idx] is None:
            return
        if thread_ctx.flatten_tensors[x1s_idx] is None:
            return
        # flatten_tensors/np_storages may be a tuple (rebased upstream); make mutable
        if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
            thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
        else:
            thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
        if thread_ctx.np_storages is not None and not isinstance(thread_ctx.np_storages, list):
            thread_ctx.np_storages = list(thread_ctx.np_storages)
        from ttk.utilities.dtypes import resolve_custom_numpy_dtypes
        dtypes = resolve_custom_numpy_dtypes(flat_dtypes)
        view_shapes = thread_ctx.flat_tensor_view_shapes
        x2_view_shape = (list(view_shapes[1]) if len(view_shapes) > 1 and view_shapes[1] is not None
                         else list(thread_ctx.flat_storage_shape(x2_idx)))
        x2s_view_shape = (list(view_shapes[x2s_idx])
                          if len(view_shapes) > x2s_idx and view_shapes[x2s_idx] is not None
                          else list(thread_ctx.flat_storage_shape(x2s_idx)))
        x1s_view_shape = (list(view_shapes[x1s_idx])
                          if len(view_shapes) > x1s_idx and view_shapes[x1s_idx] is not None
                          else list(thread_ctx.flat_storage_shape(x1s_idx)))
        x2_dtype = dtypes[1] if len(dtypes) > 1 else None
        x2s_dtype_np = dtypes[x2s_idx] if len(dtypes) > x2s_idx else None
        x1s_dtype_np = dtypes[x1s_idx] if len(dtypes) > x1s_idx else None
        # ranges from CSV
        ranges = thread_ctx.flat_input_data_ranges or ()
        def _get_range(idx, default=1.0):
            r = ranges[idx] if len(ranges) > idx and ranges[idx] else (None, None)
            low = r[0] if r[0] is not None else -default
            high = r[1] if r[1] is not None else default
            return low, high
        x2_low, x2_high = _get_range(x2_idx)
        x2s_low, x2s_high = _get_range(x2s_idx)
        x1s_low, x1s_high = _get_range(x1s_idx)
        # Shared seeds (rank-independent) for weight + scales
        rng_x2 = numpy.random.RandomState(base_seed + 1)
        x2_ss = thread_ctx.flat_storage_shape(x2_idx)
        x2s_ss = thread_ctx.flat_storage_shape(x2s_idx)
        x1s_ss = thread_ctx.flat_storage_shape(x1s_idx)
        np_x2 = rng_x2.uniform(x2_low, x2_high, x2_ss).astype(x2_dtype, copy=False)
        thread_ctx.np_storages[x2_idx] = np_x2
        thread_ctx._flat_tensors[x2_idx] = np_x2.reshape(x2_view_shape) if len(x2_view_shape) else np_x2
        # x2scale shared
        rng_x2s = numpy.random.RandomState(base_seed + 2)
        np_x2s = rng_x2s.uniform(x2s_low, x2s_high, x2s_ss).astype(x2s_dtype_np, copy=False)
        thread_ctx.np_storages[x2s_idx] = np_x2s
        thread_ctx._flat_tensors[x2s_idx] = np_x2s.reshape(x2s_view_shape) if len(x2s_view_shape) else np_x2s
        # x1scale shared (per_tensor scalar or per_block block scale)
        rng_x1s = numpy.random.RandomState(base_seed + 3)
        np_x1s = rng_x1s.uniform(x1s_low, x1s_high, x1s_ss).astype(x1s_dtype_np, copy=False)
        thread_ctx.np_storages[x1s_idx] = np_x1s
        thread_ctx._flat_tensors[x1s_idx] = np_x1s.reshape(x1s_view_shape) if len(x1s_view_shape) else np_x1s
        logging.info(f"[Thread dev={thread_ctx.my_rank}] V2 per_tensor/per_block regenerated "
                     f"shared x2 (shape {np_x2.shape}), x1scale (shape {np_x1s.shape}), x2scale (shape {np_x2s.shape})")
        return

    # --- mxfp path (e8m0 scale): full mx_quantize for x1+x1scale and x2+x2scale ---
    if thread_ctx.flatten_tensors[x1s_idx] is None or thread_ctx.flatten_tensors[x2s_idx] is None:
        return
    # flatten_tensors may be a tuple (rebased upstream property); make mutable
    if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
        thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
    else:
        thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
    if thread_ctx.np_storages is not None and not isinstance(thread_ctx.np_storages, list):
        thread_ctx.np_storages = list(thread_ctx.np_storages)
    from ttk.utilities.dtypes import mx_quantize, resolve_custom_numpy_dtypes
    dtypes = resolve_custom_numpy_dtypes(flat_dtypes)
    # Map fp8 dtype class to mx_ele_dtype name expected by mx_quantize
    fp8_dtype_map = {
        'float8_e4m3fn': 'float8_e4m3fn',
        'float8_e5m2': 'float8_e5m2',
    }
    # view_shapes: x1 is (M, K), x2 is (K, N) for trans_b=1 (CSV convention).
    # For trans_b=0, x2 view is already (K, N) (no transpose needed).
    # For fp4_e2m1/fp4_e1m2, CSV K is unpacked element count (对齐 mc2_test 行 223-224
    # 直接传 unpacked K 给 aclnn，aclnn 按 fp4 dtype 算字节 numel*0.5)。
    # 数据 storage 是 packed bytes (K//2)，view shape 用 unpacked K。
    view_shapes = thread_ctx.flat_tensor_view_shapes
    x1_view_shape = (list(view_shapes[0]) if len(view_shapes) > 0 and view_shapes[0] is not None
                     else list(thread_ctx.flat_storage_shape(0)))
    x2_view_shape = (list(view_shapes[1]) if len(view_shapes) > 1 and view_shapes[1] is not None
                     else list(thread_ctx.flat_storage_shape(1)))
    # transposeX2 从 attrs 获取（N=K 时不能靠 shape 推断 x2 layout）
    transpose_x2_attr = bool(thread_ctx.attributes.get('transposeX2', 0))
    x1_dtype = dtypes[0] if len(dtypes) > 0 else None
    x1_dtype_name = str(x1_dtype) if x1_dtype is not None else 'float8_e4m3fn'
    if "'" in x1_dtype_name:
        x1_dtype_name = x1_dtype_name.split("'")[1].split('.')[-1]
    x2_dtype = dtypes[1] if len(dtypes) > 1 else None
    x2_dtype_name = str(x2_dtype) if x2_dtype is not None else 'float8_e4m3fn'
    if "'" in x2_dtype_name:
        x2_dtype_name = x2_dtype_name.split("'")[1].split('.')[-1]
    is_fp4_x1 = 'float4' in x1_dtype_name or 'fp4' in x1_dtype_name
    is_fp4_x2 = 'float4' in x2_dtype_name or 'fp4' in x2_dtype_name
    m_dim = x1_view_shape[0]
    k_dim = x1_view_shape[1]  # CSV fp4 K 已是 unpacked 元素数，直接用
    n_dim = x2_view_shape[1] if len(x2_view_shape) > 1 else None
    # Generate x1 (M, K) + x1scale (M, K/64, 2): quantize along K (axis=-1).
    fp8_dtype_map = {
        'float8_e4m3fn': 'float8_e4m3fn',
        'float8_e5m2': 'float8_e5m2',
        'float4_e2m1': 'float4_e2m1',
        'float4_e1m2': 'float4_e1m2',
    }
    mx_ele_dtype_x1 = fp8_dtype_map.get(x1_dtype_name, 'float8_e4m3fn')
    seed_x1 = base_seed + rank_idx * 1000
    rng_x1 = numpy.random.RandomState(seed_x1)
    fp32_x1 = rng_x1.uniform(-1.0, 1.0, (m_dim, k_dim)).astype(numpy.float32)
    scale_x1, ele_x1 = mx_quantize(fp32_x1, mx_ele_dtype=mx_ele_dtype_x1,
                                    axis=-1, block_size=32, round_mode="rint")
    # ele_x1 (M, K_unpacked) fp4/fp8, scale_x1 (M, K_unpacked/64, 2) e8m0.
    # fp4: mx_quantize 返回 unpacked (M, K_unpacked)，NPU 需要 packed (M, K_unpacked//2)。
    # 对齐 mc2_test 行 95-98: pack_int4 + reshape(K>>1)。
    if is_fp4_x1:
        from ttk.utilities.dtypes import pack_4bits
        ele_x1 = pack_4bits(ele_x1.reshape(-1)).reshape(m_dim, k_dim // 2)
    thread_ctx.np_storages[x1_idx] = ele_x1
    thread_ctx.np_storages[x1s_idx] = numpy.ascontiguousarray(scale_x1)
    thread_ctx._flat_tensors[x1_idx] = ele_x1
    thread_ctx._flat_tensors[x1s_idx] = numpy.ascontiguousarray(scale_x1)
    # Generate x2 (K, N) + x2scale (K/64, N, 2): quantize along K (axis=0).
    # CSV x2 view_shape 对 trans_b=1 是 (N, K_unpacked)，对 trans_b=0 是 (K_unpacked, N)。
    # mx_quantize 沿 K 维量化：trans_b=1 时 axis=-1（最后一维 K），trans_b=0 时 axis=0。
    # fp4: CSV K 是 unpacked 元素数，mx_quantize 直接用，结果 pack_4bits 回 K//2 bytes。
    mx_ele_dtype_x2 = fp8_dtype_map.get(x2_dtype_name, 'float8_e4m3fn')
    seed_x2 = base_seed + 1  # shared across ranks (weight)
    rng_x2 = numpy.random.RandomState(seed_x2)
    # 判断 x2 layout: trans_b=1 时 view (N, K) axis=-1，trans_b=0 时 view (K, N) axis=0
    # 用 transposeX2 attr 判断（N=K 时 shape 推断歧义）
    if transpose_x2_attr:
        # view (N, K): quantize K (最后维)
        x2_fp32_shape = (x2_view_shape[0], k_dim)
        mx_axis = -1
    else:
        # view (K, N): quantize K (第一维)
        x2_fp32_shape = (k_dim, x2_view_shape[1])
        mx_axis = 0
    fp32_x2 = rng_x2.uniform(-1.0, 1.0, x2_fp32_shape).astype(numpy.float32)
    scale_x2, ele_x2 = mx_quantize(fp32_x2, mx_ele_dtype=mx_ele_dtype_x2,
                                    axis=mx_axis, block_size=32, round_mode="rint")
    # fp4 x2: pack 到 K 维减半（storage bytes = numel*0.5，view shape 仍 unpacked）
    if is_fp4_x2:
        from ttk.utilities.dtypes import pack_4bits
        if mx_axis == -1:
            ele_x2 = pack_4bits(ele_x2.reshape(-1)).reshape(x2_view_shape[0], k_dim // 2)
        else:
            ele_x2 = pack_4bits(ele_x2.reshape(-1)).reshape(k_dim // 2, x2_view_shape[1])
    # ele_x2 (K, N) fp8, scale_x2 (K/64, N, 2) e8m0.
    # mx_quantize produces non-contiguous scale (interleaved layout); force
    # C-contiguous so aclCreateTensor sees standard row-major [K/64, N, 2].
    thread_ctx.np_storages[x2_idx] = ele_x2
    thread_ctx.np_storages[x2s_idx] = numpy.ascontiguousarray(scale_x2)
    thread_ctx._flat_tensors[x2_idx] = ele_x2
    thread_ctx._flat_tensors[x2s_idx] = numpy.ascontiguousarray(scale_x2)
    logging.info(f"[Thread dev={thread_ctx.my_rank}] V2 mxfp regenerated "
                 f"x1+scale (shape {ele_x1.shape}/{scale_x1.shape}) and "
                 f"x2+scale (shape {ele_x2.shape}/{scale_x2.shape}) via mx_quantize")


def __regenerate_v4_quant_inputs(thread_ctx, rank_idx: int, base_seed: int):
    """For aclnnQuantMatmulAllReduceV4/V5 quant cases (mxfp/per_block),
    regenerate correlated x1+x1scale and x2+x2scale.

    V4/V5 flatten_tensors layout:
      x1(0), x2(1), bias(2), x3(3), x1Scale(4), x2Scale(5), ..., output(8)
    (slot 4 = x1Scale, slot 5 = x2Scale; differs from V2 where slot 3/4).

    For mxfp (e8m0 scale): full mx_quantize for x1+x1scale and x2+x2scale.
    For per_block (fp32 scale): only share x2 + x2scale + x1scale across ranks
        (weight semantics; InputGenerator used per_rank_seed making them differ).
    For int8/hf8/per_tile (fp32 scale): share x2 + x2scale + x1scale across ranks.
    """
    flat_dtypes = thread_ctx.flat_tensor_dtypes or []
    if len(flat_dtypes) < 6:
        return
    x1_idx, x2_idx, x1s_idx, x2s_idx = 0, 1, 4, 5
    if len(thread_ctx.flatten_tensors) <= x2s_idx:
        return
    x1_dtype_str = str(flat_dtypes[0]) if flat_dtypes[0] else ''
    is_quant_x1 = any(d in x1_dtype_str for d in ('float8', 'hifloat8', 'fp8', 'hif8', 'float4', 'fp4'))
    if not is_quant_x1:
        return

    x1s_dtype = flat_dtypes[x1s_idx]
    if thread_ctx.flatten_tensors[x1s_idx] is None or thread_ctx.flatten_tensors[x2s_idx] is None:
        return
    if thread_ctx.flatten_tensors[x2_idx] is None:
        return

    if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
        thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
    else:
        thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
    if thread_ctx.np_storages is not None and not isinstance(thread_ctx.np_storages, list):
        thread_ctx.np_storages = list(thread_ctx.np_storages)

    # --- mxfp path (e8m0 scale): full mx_quantize ---
    if x1s_dtype in ('fp8_e8m0', 'float8_e8m0'):
        from ttk.utilities.dtypes import mx_quantize, resolve_custom_numpy_dtypes, pack_4bits
        dtypes = resolve_custom_numpy_dtypes(flat_dtypes)
        view_shapes = thread_ctx.flat_tensor_view_shapes
        x1_view_shape = (list(view_shapes[0]) if len(view_shapes) > 0 and view_shapes[0] is not None
                         else list(thread_ctx.flat_storage_shape(x1_idx)))
        x2_view_shape = (list(view_shapes[1]) if len(view_shapes) > 1 and view_shapes[1] is not None
                         else list(thread_ctx.flat_storage_shape(x2_idx)))
        is_trans_b = 'is_trans_b=True' in (thread_ctx.remark or '')
        x1_dtype = dtypes[0] if len(dtypes) > 0 else None
        x1_dtype_name = str(x1_dtype) if x1_dtype is not None else 'float8_e4m3fn'
        if "'" in x1_dtype_name:
            x1_dtype_name = x1_dtype_name.split("'")[1].split('.')[-1]
        x2_dtype = dtypes[1] if len(dtypes) > 1 else None
        x2_dtype_name = str(x2_dtype) if x2_dtype is not None else 'float8_e4m3fn'
        if "'" in x2_dtype_name:
            x2_dtype_name = x2_dtype_name.split("'")[1].split('.')[-1]
        is_fp4_x1 = 'float4' in x1_dtype_name or 'fp4' in x1_dtype_name
        is_fp4_x2 = 'float4' in x2_dtype_name or 'fp4' in x2_dtype_name
        m_dim = x1_view_shape[0]
        k_dim = x1_view_shape[-1]
        fp8_dtype_map = {
            'float8_e4m3fn': 'float8_e4m3fn',
            'float8_e5m2': 'float8_e5m2',
            'float4_e2m1': 'float4_e2m1',
            'float4_e1m2': 'float4_e1m2',
        }
        mx_ele_dtype_x1 = fp8_dtype_map.get(x1_dtype_name, 'float8_e4m3fn')
        seed_x1 = base_seed + rank_idx * 1000
        rng_x1 = numpy.random.RandomState(seed_x1)
        # 3D x1 [B, S, K] → 2D [B*S, K] for mx_quantize
        if len(x1_view_shape) == 3:
            m_dim = x1_view_shape[0] * x1_view_shape[1]
            k_dim = x1_view_shape[2]
        fp32_x1 = rng_x1.uniform(-1.0, 1.0, (m_dim, k_dim)).astype(numpy.float32)
        scale_x1, ele_x1 = mx_quantize(fp32_x1, mx_ele_dtype=mx_ele_dtype_x1,
                                        axis=-1, block_size=32, round_mode="rint")
        if is_fp4_x1:
            ele_x1 = pack_4bits(ele_x1.reshape(-1)).reshape(m_dim, k_dim // 2)
        # 3D x1: reshape ele_x1 and scale_x1 back to 3D view
        if len(x1_view_shape) == 3:
            if is_fp4_x1:
                ele_x1 = ele_x1.reshape(x1_view_shape[0], x1_view_shape[1], k_dim // 2)
            else:
                ele_x1 = ele_x1.reshape(x1_view_shape[0], x1_view_shape[1], k_dim)
            scale_x1 = scale_x1.reshape(x1_view_shape[0], x1_view_shape[1], -1, 2)
        thread_ctx.np_storages[x1_idx] = ele_x1
        thread_ctx.np_storages[x1s_idx] = numpy.ascontiguousarray(scale_x1)
        thread_ctx._flat_tensors[x1_idx] = ele_x1
        thread_ctx._flat_tensors[x1s_idx] = numpy.ascontiguousarray(scale_x1)

        # x2 (K, N) + x2scale (K/64, N, 2): quantize along K
        mx_ele_dtype_x2 = fp8_dtype_map.get(x2_dtype_name, 'float8_e4m3fn')
        seed_x2 = base_seed + 1
        rng_x2 = numpy.random.RandomState(seed_x2)
        # V4 x2 view: trans_b=1 时 xlsx 存 (N, K) → CSV 转成 (K, N)；
        # CSV 中 x2 已是 (K, N)。mx_quantize 沿 K (第一维)。
        x2_fp32_shape = (k_dim, x2_view_shape[-1])
        mx_axis = 0
        fp32_x2 = rng_x2.uniform(-1.0, 1.0, x2_fp32_shape).astype(numpy.float32)
        scale_x2, ele_x2 = mx_quantize(fp32_x2, mx_ele_dtype=mx_ele_dtype_x2,
                                        axis=mx_axis, block_size=32, round_mode="rint")
        if is_fp4_x2:
            ele_x2 = pack_4bits(ele_x2.reshape(-1)).reshape(k_dim // 2, x2_view_shape[-1])
        thread_ctx.np_storages[x2_idx] = ele_x2
        thread_ctx.np_storages[x2s_idx] = numpy.ascontiguousarray(scale_x2)
        thread_ctx._flat_tensors[x2_idx] = ele_x2
        thread_ctx._flat_tensors[x2s_idx] = numpy.ascontiguousarray(scale_x2)
        logging.info(f"[Thread dev={thread_ctx.my_rank}] V4 mxfp regenerated "
                     f"x1+scale (shape {ele_x1.shape}/{scale_x1.shape}) and "
                     f"x2+scale (shape {ele_x2.shape}/{scale_x2.shape}) via mx_quantize")
        return

    # --- per_block / per_tile / hf8 / int8 path: share x2 + x2scale + x1scale ---
    from ttk.utilities.dtypes import resolve_custom_numpy_dtypes
    dtypes = resolve_custom_numpy_dtypes(flat_dtypes)
    view_shapes = thread_ctx.flat_tensor_view_shapes
    x2_view_shape = (list(view_shapes[1]) if len(view_shapes) > 1 and view_shapes[1] is not None
                     else list(thread_ctx.flat_storage_shape(x2_idx)))
    x2s_view_shape = (list(view_shapes[x2s_idx])
                      if len(view_shapes) > x2s_idx and view_shapes[x2s_idx] is not None
                      else list(thread_ctx.flat_storage_shape(x2s_idx)))
    x1s_view_shape = (list(view_shapes[x1s_idx])
                      if len(view_shapes) > x1s_idx and view_shapes[x1s_idx] is not None
                      else list(thread_ctx.flat_storage_shape(x1s_idx)))
    x2_dtype = dtypes[1] if len(dtypes) > 1 else None
    x2s_dtype_np = dtypes[x2s_idx] if len(dtypes) > x2s_idx else None
    x1s_dtype_np = dtypes[x1s_idx] if len(dtypes) > x1s_idx else None
    ranges = thread_ctx.flat_input_data_ranges or ()
    def _get_range(idx, default=1.0):
        r = ranges[idx] if len(ranges) > idx and ranges[idx] else (None, None)
        low = r[0] if r[0] is not None else -default
        high = r[1] if r[1] is not None else default
        return low, high
    x2_low, x2_high = _get_range(x2_idx)
    x2s_low, x2s_high = _get_range(x2s_idx)
    x1s_low, x1s_high = _get_range(x1s_idx)
    rng_x2 = numpy.random.RandomState(base_seed + 1)
    x2_ss = thread_ctx.flat_storage_shape(x2_idx)
    x2s_ss = thread_ctx.flat_storage_shape(x2s_idx)
    x1s_ss = thread_ctx.flat_storage_shape(x1s_idx)
    np_x2 = rng_x2.uniform(x2_low, x2_high, x2_ss).astype(x2_dtype, copy=False)
    thread_ctx.np_storages[x2_idx] = np_x2
    thread_ctx._flat_tensors[x2_idx] = np_x2.reshape(x2_view_shape) if len(x2_view_shape) else np_x2
    rng_x2s = numpy.random.RandomState(base_seed + 2)
    np_x2s = rng_x2s.uniform(x2s_low, x2s_high, x2s_ss).astype(x2s_dtype_np, copy=False)
    thread_ctx.np_storages[x2s_idx] = np_x2s
    thread_ctx._flat_tensors[x2s_idx] = np_x2s.reshape(x2s_view_shape) if len(x2s_view_shape) else np_x2s
    rng_x1s = numpy.random.RandomState(base_seed + 3)
    np_x1s = rng_x1s.uniform(x1s_low, x1s_high, x1s_ss).astype(x1s_dtype_np, copy=False)
    thread_ctx.np_storages[x1s_idx] = np_x1s
    thread_ctx._flat_tensors[x1s_idx] = np_x1s.reshape(x1s_view_shape) if len(x1s_view_shape) else np_x1s
    logging.info(f"[Thread dev={thread_ctx.my_rank}] V4 per_block/per_tile/hf8 regenerated "
                 f"shared x2 (shape {np_x2.shape}), x1scale (shape {np_x1s.shape}), x2scale (shape {np_x2s.shape})")


def __regenerate_quant_allreduce_mxfp_inputs(thread_ctx, rank_idx: int, base_seed: int):
    """For aclnnQuantAllReduce/aclnnQuantReduceScatter mxfp cases, regenerate
    correlated x+scales via mx_quantize so NPU tiling succeeds.

    Detection: scales (slot 1) dtype is fp8_e8m0.
    For non-mxfp (slot 1 dtype is fp32 pertoken-pergroup), no action needed.

    flatten_tensors layout: x(0), scales(1), output(2)
    mxfp scales shape: [BS, H/64, 2] (2D x) or [B, S, H/64, 2] (3D x)
    对齐 mc2_test aclnnQuantAllReduce.generate_mx_fp_data 行 59-82:
      x_fp32 = create_array(..., fp32); x, scale = mx_quantize(x_fp32, dtype, axis=-1, block_size=32)
    """
    flat_dtypes = thread_ctx.flat_tensor_dtypes or []
    if len(flat_dtypes) < 2:
        return
    x_idx, s_idx = 0, 1
    scales_dtype = flat_dtypes[s_idx]
    if scales_dtype not in ('fp8_e8m0', 'float8_e8m0'):
        return
    if thread_ctx.flatten_tensors[x_idx] is None or thread_ctx.flatten_tensors[s_idx] is None:
        return

    if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
        thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
    else:
        thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
    if thread_ctx.np_storages is not None and not isinstance(thread_ctx.np_storages, list):
        thread_ctx.np_storages = list(thread_ctx.np_storages)

    from ttk.utilities.dtypes import mx_quantize, resolve_custom_numpy_dtypes
    dtypes = resolve_custom_numpy_dtypes(flat_dtypes)
    view_shapes = thread_ctx.flat_tensor_view_shapes
    x_view_shape = list(view_shapes[x_idx]) if len(view_shapes) > x_idx and view_shapes[x_idx] is not None \
        else list(thread_ctx.flat_storage_shape(x_idx))
    x_dtype = dtypes[x_idx] if len(dtypes) > x_idx else None
    x_dtype_name = str(x_dtype) if x_dtype is not None else 'float8_e4m3fn'
    if "'" in x_dtype_name:
        x_dtype_name = x_dtype_name.split("'")[1].split('.')[-1]

    fp8_dtype_map = {
        'float8_e4m3fn': 'float8_e4m3fn',
        'float8_e5m2': 'float8_e5m2',
    }
    mx_ele_dtype = fp8_dtype_map.get(x_dtype_name, 'float8_e4m3fn')

    # 3D x [B, S, H] → 2D [B*S, H] for mx_quantize
    if len(x_view_shape) == 3:
        m_dim = x_view_shape[0] * x_view_shape[1]
        k_dim = x_view_shape[2]
    else:
        m_dim = x_view_shape[0]
        k_dim = x_view_shape[-1]

    # x range from CSV (mxfp 时存的是 fp32 range，用于生成 mx_quantize 输入)
    ranges = thread_ctx.flat_input_data_ranges or ()
    x_low, x_high = -1.0, 1.0
    if len(ranges) > x_idx and ranges[x_idx]:
        r = ranges[x_idx]
        if r[0] is not None:
            x_low = r[0]
        if r[1] is not None:
            x_high = r[1]

    seed_x = base_seed + rank_idx * 1000
    rng_x = numpy.random.RandomState(seed_x)
    fp32_x = rng_x.uniform(x_low, x_high, (m_dim, k_dim)).astype(numpy.float32)
    scale_x, ele_x = mx_quantize(fp32_x, mx_ele_dtype=mx_ele_dtype,
                                 axis=-1, block_size=32, round_mode="rint")

    # 3D x: reshape ele_x and scale_x back to 3D view
    if len(x_view_shape) == 3:
        ele_x = ele_x.reshape(x_view_shape[0], x_view_shape[1], k_dim)
        scale_x = scale_x.reshape(x_view_shape[0], x_view_shape[1], -1, 2)
    else:
        # 2D x: scale_x from mx_quantize is [M, H/64, 2].
        # Keep the CSV view as [M, H/64, 2].
        scale_x = scale_x.reshape(m_dim, -1, 2)

    thread_ctx.np_storages[x_idx] = numpy.ascontiguousarray(ele_x)
    thread_ctx.np_storages[s_idx] = numpy.ascontiguousarray(scale_x)
    thread_ctx._flat_tensors[x_idx] = numpy.ascontiguousarray(ele_x)
    thread_ctx._flat_tensors[s_idx] = numpy.ascontiguousarray(scale_x)
    logging.info(f"[Thread dev={thread_ctx.my_rank}] QuantAllReduce/QuantReduceScatter mxfp regenerated "
                 f"x (shape {ele_x.shape}) + scales (shape {scale_x.shape}) via mx_quantize")


def __trans_quant_dequant_scale(thread_ctx, dev_id, ds_slot=4):
    """aclnnQuantMatmulAllReduce (V1): 把 fp32 dequantScale 转成 int64 喂给融合算子。

    A5 tiling 规则（quant_matmul_all_reduce_tiling_950.cpp 行 586-601）：
      - y=fp16 且无 pertokenScale → dequantScale 必须为 int64/uint64
      - y=bf16 → dequantScale 为 bf16（无需转换）
    mc2_test op_class/aclnnQuantMatmulAllReduce.py 行 79-81 用
      npu_trans_quant_param 把 fp32 转成 int64。这里复刻该逻辑。

    检测：slot ds_slot (dequantScale) dtype == float32 且 output dtype == float16。
    V1/V2/V3 ds_slot=4; V4/V5 x2Scale ds_slot=5。
    转换后同步更新 flatten_tensors / np_storages / flat_tensor_dtypes 为 int64，
    使后续 aclCreateTensor 用 int64 dtype。
    CPU golden（mc2_golden）仍用原始 fp32 数值计算，不受此转换影响。
    """
    import torch
    import torch_npu  # noqa: F401
    # ds_slot: V1/V2/V3 dequantScale 在 slot 4; V4/V5 x2Scale 在 slot 5。
    ds_idx = ds_slot
    out_idx_list = list(thread_ctx.output_tensor_indexes or ())
    flat_dtypes = list(thread_ctx.flat_tensor_dtypes or [])
    if ds_idx >= len(flat_dtypes):
        return
    ds_dtype = flat_dtypes[ds_idx]
    if ds_dtype is None or str(ds_dtype).lower() not in ('float32', 'fp32'):
        return  # bf16 / int64 / uint64 无需转换
    # output dtype 判断（flat_output_dtypes 或 flat_tensor_dtypes[out_idx]）
    out_dtypes = thread_ctx.flat_output_dtypes if thread_ctx.flat_output_dtypes else []
    out_dtype_str = out_dtypes[0] if len(out_dtypes) > 0 else (
        flat_dtypes[out_idx_list[0]] if out_idx_list and out_idx_list[0] < len(flat_dtypes) else '')
    if str(out_dtype_str).lower() not in ('float16', 'fp16'):
        return  # bf16 output 走 bf16 dequantScale，无需转换
    if ds_idx >= len(thread_ctx.flatten_tensors) or thread_ctx.flatten_tensors[ds_idx] is None:
        return
    # pertokenScale slot: V1 output=5（slot 5 是 output，无 pertoken）;
    #   V2/V3 pertokenScale 在 slot 5（ds_slot=4）;
    #   V4/V5 x1Scale（pertoken）在 slot 4（ds_slot=5）。
    # 当 pertokenScale 存在（非空）时，NPU 要求 dequantScale 与 pertokenScale 同 dtype（fp32），
    # 不能转 int64（对齐 mc2_test 行 79-81: if not is_pertoken 才转）。
    pt_idx = 5 if ds_slot == 4 else 4
    if pt_idx not in out_idx_list and pt_idx < len(thread_ctx.flatten_tensors):
        pt_t = thread_ctx.flatten_tensors[pt_idx]
        if pt_t is not None and isinstance(pt_t, torch.Tensor) and pt_t.numel() > 0:
            return  # pertokenScale 存在，保持 fp32 dequantScale
    ds_tensor = thread_ctx.flatten_tensors[ds_idx]
    if not isinstance(ds_tensor, torch.Tensor):
        # numpy array → torch
        ds_tensor = torch.from_numpy(numpy.ascontiguousarray(ds_tensor))
    # 保留 fp32 原值给 CPU golden（mc2_golden 读 attributes['_qm_dequant_scale_fp32']）。
    # 用 attributes dict 存（TestcaseAclnn 用 __slots__，不能加自定义属性；
    # Phase1ParamBuilder.build 只读 group/reduceOp 等已知 key，多余 key 被忽略）。
    thread_ctx.attributes['_qm_dequant_scale_fp32'] = ds_tensor.float().clone()
    # NPU 侧转换 fp32 → int64
    ds_npu = ds_tensor.npu()
    ds_int64 = torch_npu.npu_trans_quant_param(ds_npu).cpu()
    # 更新 flatten_tensors / np_storages / flat_tensor_dtypes
    if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
        thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
    else:
        thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
    thread_ctx._flat_tensors[ds_idx] = ds_int64.contiguous()
    # flat_tensor_dtypes 是 property，需写私有 slot _flat_tensor_dtypes
    new_dtypes = list(flat_dtypes)
    new_dtypes[ds_idx] = 'int64'
    thread_ctx._flat_tensor_dtypes = tuple(new_dtypes)
    # np_storages 同步（golden 比对时按 int64 dtype 解析 output bytes，需一致）
    if thread_ctx.np_storages is not None:
        storages = list(thread_ctx.np_storages) if not isinstance(thread_ctx.np_storages, list) \
            else thread_ctx.np_storages
        if ds_idx < len(storages):
            storages[ds_idx] = ds_int64.numpy().astype(numpy.int64, copy=False)
            thread_ctx.np_storages = storages
    logging.info(f"[Thread dev={dev_id}] QuantMatmulAllReduce: converted fp32 dequantScale "
                 f"(shape {tuple(ds_tensor.shape)}) to int64 via npu_trans_quant_param")


def __launch_one_thread(context: TestcaseAclnn,
                         dev_id: int,
                         rank_idx: int,
                         ndev: int,
                         hccl_comm_val: int,
                         c_context_val: int,
                         c_stream_val: int,
                         results: Dict[int, ApiProfilingResult],
                         thread_contexts: Dict[int, TestcaseAclnn],
                         errors: Dict[int, Exception],
                         main_device: 'AclInterface',
                         ep_comm_val: int = 0,
                         tp_comm_val: int = 0,
                         pre_exec_barrier: threading.Barrier = None,
                         post_exec_barrier: threading.Barrier = None,
                         comm_destroyed_set: Optional[set] = None):
    try:
        logging.info(f"[Thread dev={dev_id}] starting, rank_idx={rank_idx}")
        acl_dll = ctypes.CDLL("libascendcl.so")
        acl_dll.aclrtSetCurrentContext.restype = ctypes.c_int
        acl_dll.aclrtSetCurrentContext.argtypes = [ctypes.c_void_p]
        acl_dll.aclrtSetCurrentContext(ctypes.c_void_p(c_context_val))
        hccl_dll = ctypes.CDLL("libhcomm.so")
        hccl_dll.HcclGetCommName.restype = ctypes.c_int
        hcom_name_buf = ctypes.create_string_buffer(128)
        ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(hccl_comm_val), hcom_name_buf)
        if ret != 0:
            raise RuntimeError(f"HcclGetCommName failed with ret={ret} for dev {dev_id}")
        hcom_name = hcom_name_buf.value.decode('utf-8')
        logging.info(f"[Thread dev={dev_id}] HcclGetCommName = '{hcom_name}'")
        thread_ctx = copy.deepcopy(context)
        thread_ctx.my_rank = dev_id
        if 'group' in thread_ctx.attributes:
            thread_ctx.attributes['group'] = hcom_name
        # MC2 APIs may name the EP/TP communicator separately.  When no
        # topology-specific sub-group was requested, the primary world
        # communicator is the valid fallback for both names.
        if 'groupEp' in thread_ctx.attributes and ep_comm_val == 0:
            thread_ctx.attributes['groupEp'] = hcom_name
        if 'groupTp' in thread_ctx.attributes and tp_comm_val == 0:
            thread_ctx.attributes['groupTp'] = hcom_name
        if ep_comm_val != 0 or tp_comm_val != 0:
            if ep_comm_val != 0:
                ep_name_buf = ctypes.create_string_buffer(128)
                ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(ep_comm_val), ep_name_buf)
                if ret == 0:
                    thread_ctx.attributes['groupEp'] = ep_name_buf.value.decode('utf-8')
                    logging.info(f"[Thread dev={dev_id}] groupEp = '{thread_ctx.attributes['groupEp']}'")
            if tp_comm_val != 0:
                tp_name_buf = ctypes.create_string_buffer(128)
                ret = hccl_dll.HcclGetCommName(ctypes.c_void_p(tp_comm_val), tp_name_buf)
                if ret == 0:
                    thread_ctx.attributes['groupTp'] = tp_name_buf.value.decode('utf-8')
                    logging.info(f"[Thread dev={dev_id}] groupTp = '{thread_ctx.attributes['groupTp']}'")
        switches = get_global_storage()
        base_seed = switches.random_seed or 0
        per_rank_seed = base_seed + rank_idx * 1000
        numpy.random.seed(per_rank_seed)
        logging.info(f"[Thread dev={dev_id}] using seed={per_rank_seed} for input generation")
        patch_gmm_rank_attributes(thread_ctx, rank_idx, ndev)
        api_name = thread_ctx.api_name or ''
        exclude_ops = ("BatchMatMulReduceScatter", "AlltoAllAllGather",
                       "GroupedMatMul", "AlltoAllvGrouped")
        if any(kw in api_name for kw in exclude_ops):
            needs_shared_weight = False
        else:
            weight_shared_ops = ("MatmulReduceScatter", "AllGatherMatmul",
                                 "MatmulAllReduce", "MatmulAlltoAll",
                                 "AlltoAllMatmul", "MatmulReduceScatterV2",
                                 "QuantMatmulAllReduce")
            needs_shared_weight = any(kw in api_name for kw in weight_shared_ops)
            # V2 quant (fp8/hif8 x1) needs x1+scale correlated; weight_shared
            # regeneration breaks that correlation. Skip regeneration for V2 quant.
            if needs_shared_weight and 'AllGatherMatmulV2' in api_name:
                v2_dtypes = resolve_custom_numpy_dtypes(thread_ctx.flat_tensor_dtypes)
                x1_dtype_str = str(v2_dtypes[0]) if v2_dtypes and len(v2_dtypes) > 0 else ''
                if any(d in x1_dtype_str for d in ('float8', 'hifloat8', 'fp8', 'hif8')):
                    needs_shared_weight = False
                    logging.info(f"[Thread dev={dev_id}] V2 quant detected (x1 dtype={x1_dtype_str}), "
                                 f"skip weight_shared regeneration")
            # V4/V5 quant (fp8/hif8/fp4 x1) needs x1+scale and x2+scale correlated;
            # weight_shared regeneration breaks that correlation.
            # __regenerate_v4_quant_inputs handles shared weight semantics.
            if needs_shared_weight and ('QuantMatmulAllReduceV4' in api_name
                                        or 'QuantMatmulAllReduceV5' in api_name):
                v4_dtypes = resolve_custom_numpy_dtypes(thread_ctx.flat_tensor_dtypes)
                x1_dtype_str = str(v4_dtypes[0]) if v4_dtypes and len(v4_dtypes) > 0 else ''
                if any(d in x1_dtype_str for d in ('float8', 'hifloat8', 'fp8', 'hif8', 'float4', 'fp4')):
                    needs_shared_weight = False
                    logging.info(f"[Thread dev={dev_id}] V4/V5 quant detected (x1 dtype={x1_dtype_str}), "
                                 f"skip weight_shared regeneration")
        try:
            InputGenerator(thread_ctx).gen()
            if (needs_shared_weight and len(thread_ctx.flatten_tensors) > 1
                    and thread_ctx.flatten_tensors[1] is not None):
                # flatten_tensors/np_storages may be a tuple (rebased upstream); make mutable
                if not hasattr(thread_ctx, '_flat_tensors') or thread_ctx._flat_tensors is None:
                    thread_ctx._flat_tensors = list(thread_ctx.flatten_tensors)
                else:
                    thread_ctx._flat_tensors = list(thread_ctx._flat_tensors)
                if thread_ctx.np_storages is not None and not isinstance(thread_ctx.np_storages, list):
                    thread_ctx.np_storages = list(thread_ctx.np_storages)
                dtypes = resolve_custom_numpy_dtypes(thread_ctx.flat_tensor_dtypes)
                for t_idx in (0, 1):
                    if t_idx >= len(thread_ctx.flatten_tensors) or thread_ctx.flatten_tensors[t_idx] is None:
                        continue
                    ss = thread_ctx.flat_storage_shape(t_idx)
                    dtype = dtypes[t_idx] if t_idx < len(dtypes) else None
                    data_range = ((thread_ctx.flat_input_data_ranges or ())[t_idx]
                                  if t_idx < len(thread_ctx.flat_input_data_ranges or ()) else (None, None))
                    low = data_range[0] if data_range[0] is not None else -1.0
                    high = data_range[1] if data_range[1] is not None else 1.0
                    rng = numpy.random.RandomState(base_seed + t_idx)
                    np_arr = rng.uniform(low, high, ss).astype(dtype, copy=False)
                    thread_ctx.np_storages[t_idx] = np_arr
                    if thread_ctx.is_torch_dtype_support():
                        from ttk.utilities.dtypes import numpy_to_torch_tensor
                        complex32 = "complex32" in str(dtype)
                        t = numpy_to_torch_tensor(np_arr, is_complex32=complex32)
                        thread_ctx._flat_tensors[t_idx] = t.reshape(ss)
                    else:
                        thread_ctx._flat_tensors[t_idx] = np_arr.reshape(ss)
                logging.info(f"[Thread dev={dev_id}] regenerated x1/x2 with base_seed={base_seed}")
            # V2 mxfp: regenerate correlated x1+x1scale, x2+x2scale via mx_quantize.
            # NPU tiling requires x1 (fp8) and x1scale (e8m0) to be correlated
            # (scale_array = 2^share_exp from the same mx_quantize pass).
            # Default InputGenerator produces independent random data which causes
            # errno 561002 (tiling failure).
            if 'AllGatherMatmulV2' in api_name or 'QuantMatmulAlltoAll' in api_name:
                __regenerate_v2_mxfp_inputs(thread_ctx, rank_idx, base_seed)
            # aclnnQuantMatmulAllReduceV4/V5: mxfp 需 correlated x1+x1scale, x2+x2scale;
            # per_block/per_tile/hf8 需 share x2+x2scale+x1scale across ranks (weight 语义)。
            # V4/V5 slot 4=x1Scale, slot 5=x2Scale（与 V2 slot 3/4 不同）。
            if ('QuantMatmulAllReduceV4' in api_name or 'QuantMatmulAllReduceV5' in api_name):
                __regenerate_v4_quant_inputs(thread_ctx, rank_idx, base_seed)
            # aclnnQuantAllReduce/aclnnQuantReduceScatter: mxfp 需 correlated x+scales
            # via mx_quantize (NPU tiling requires correlated fp8 x + e8m0 scales)。
            if api_name in ('aclnnQuantAllReduce', 'aclnnQuantReduceScatter'):
                __regenerate_quant_allreduce_mxfp_inputs(thread_ctx, rank_idx, base_seed)
            # aclnnQuantMatmulAllReduce (V1/V2/V3): A5 tiling 规则要求
            #   y=fp16 且无 pertokenScale 时 dequantScale 必须为 int64/uint64
            #   (quant_matmul_all_reduce_tiling_950.cpp 行 586-601)。
            # xlsx 存 fp32 dequant_scale，mc2_test 在 NPU 侧用 npu_trans_quant_param
            # 转 int64（op_class/aclnnQuantMatmulAllReduce.py 行 79-81）。
            # V1/V2/V3 ds_slot=4; V4/V5 int8 路径 ds_slot=5（x2Scale）。
            # V4/V5 fp8/fp4/per_block/mxfp/per_tile 路径不转换（x2Scale 语义不同）。
            if ('QuantMatmulAllReduce' in api_name and 'Weight' not in api_name
                    and 'QuantMatmulAlltoAll' not in api_name):
                if 'QuantMatmulAllReduceV4' in api_name or 'QuantMatmulAllReduceV5' in api_name:
                    # V4/V5: 仅 int8 路径（x1 dtype == int8）需要转换
                    v4_dtypes = resolve_custom_numpy_dtypes(thread_ctx.flat_tensor_dtypes)
                    v4_x1_dtype = str(v4_dtypes[0]) if v4_dtypes and len(v4_dtypes) > 0 else ''
                    if 'int8' in v4_x1_dtype.lower():
                        __trans_quant_dequant_scale(thread_ctx, dev_id, ds_slot=5)
                else:
                    __trans_quant_dequant_scale(thread_ctx, dev_id)
        except Exception:
            logging.exception(f"[Thread dev={dev_id}] input generation failed:")
            errors[dev_id] = RuntimeError("INPUT_GEN_FAILURE")
            results[dev_id] = ApiProfilingResult.fail("INPUT_GEN_FAILURE")
            return
        patch_gmm_weight_transpose(thread_ctx)
        thread_device = AclInterface.__new__(AclInterface)
        thread_device.__dict__ = main_device.__dict__.copy()
        thread_device._acl_inited = True
        thread_device._device_id = dev_id
        thread_device._acl_tensors = set()
        thread_device._acl_scalars = set()
        thread_device._acl_int_arrays = set()
        thread_device._acl_float_arrays = set()
        thread_device._acl_bool_arrays = set()
        thread_device._acl_device_memories = set()
        thread_device._suppress_finalize = True
        # Suppress per-thread aclrtResetDevice: the main thread performs unified
        # cleanup (HcclCommDestroy / DestroyStream / DestroyContext / ResetDevice)
        # after both ranks join. A child thread resetting its device early freed
        # stream/context handles that the main thread then double-freed, causing
        # a SIGSEGV (exit code -11) on A5 during cleanup.
        thread_device._suppress_reset = True
        from ...runtime.rts_interface import RTSInterface
        from ...aclnn.op_api_info_keeper import OpApiInfoKeeper, _builtin_api_so_map, _ensure_builtin_so_map
        from ....utilities.platform import get_ascend_lib64_path
        opp = os.environ.get('ASCEND_OPP_PATH', '')
        ld = os.environ.get('LD_LIBRARY_PATH', '')
        home = os.environ.get('ASCEND_HOME_PATH', '')
        logging.info(
            f"[Thread dev={dev_id}] env: ASCEND_OPP_PATH='{opp[:60]}' "
            f"LD_HAS_ASCEND={'ascend' in ld} HOME='{home[:60]}'")
        # Resolve arch-specific lib64 dir (x86_64-linux on x86, aarch64-linux on arm).
        # Previously hardcoded 'aarch64-linux', which broke on x86_64 hosts:
        # resolved_so became '' and info._so_path was overwritten to empty,
        # causing "SO path is empty" errors and HCCL deadlock on multi-device ops.
        # get_ascend_lib64_path() reads scene.info and returns the correct
        # <arch>-<os>/lib64 path (same helper used by op_api_info_keeper builtin).
        lib64_dir = get_ascend_lib64_path() if opp else ''
        so_candidates = [os.path.join(lib64_dir, f) for f in
                         ('libopapi_transformer.so', 'libopapi.so') if lib64_dir]
        resolved_so = next((p for p in so_candidates if os.path.isfile(p)), '')
        logging.info(f"[Thread dev={dev_id}] lib64_dir={lib64_dir} so={resolved_so}")
        if resolved_so:
            info = OpApiInfoKeeper().info_of(thread_ctx.api_name)
            if info:
                # Only patch if the resolved SO is more accurate than the
                # current so_path. The builtin OpApiInfoKeeper already sets
                # so_path correctly on x86_64; blindly overwriting it here
                # (the legacy code overwrote with an aarch64 path or '') caused
                # "SO path is empty" errors and HCCL deadlock on multi-device.
                # Skip patching when so_path is already set and valid.
                current_so = info.so_path
                if not current_so or not os.path.isfile(current_so):
                    info._so_path = resolved_so
                    logging.info(f"[Thread dev={dev_id}] patched info._so_path={resolved_so} (was empty/invalid)")
                else:
                    logging.info(f"[Thread dev={dev_id}] keep existing so_path={current_so}")
            else:
                _ensure_builtin_so_map()
                logging.info(
                    f"[Thread dev={dev_id}] info=None, builtin_map_keys="
                    f"{list(_builtin_api_so_map.keys())[:5] if _builtin_api_so_map else 'None'}")
        thread_rts = main_device._rts_interface.__dict__.copy()
        thread_rts['memory_manager'] = {}
        thread_rts['_device_id'] = dev_id
        thread_rts['_acl_inited'] = True
        thread_device._rts_interface = RTSInterface.__new__(RTSInterface)
        thread_device._rts_interface.__dict__ = thread_rts
        c_stream = ctypes.c_void_p(c_stream_val)
        if pre_exec_barrier is not None:
            logging.info(f"[Thread dev={dev_id}] waiting at pre-exec barrier")
            pre_exec_barrier.wait()
            logging.info(f"[Thread dev={dev_id}] passed pre-exec barrier")
        thread_ctx.prof_result = AclOpExecutor(thread_ctx, thread_device).do(
            stream=c_stream, skip_context_creation=True)
        logging.info(f"[Thread dev={dev_id}] execution done, success={not thread_ctx.prof_result.failed()}")
        thread_contexts[dev_id] = thread_ctx
        results[dev_id] = thread_ctx.prof_result
        # 透传 HCCL comm/context/stream 给 golden 阶段（真级联需要）。
        # 由主线程在 golden 完成后统一 HcclCommDestroy + DestroyStream + DestroyContext + ResetDevice。
        # 原代码在子线程末尾立即 destroy，导致 golden 阶段拿不到 comm。
        thread_ctx._hccl_handles = (hccl_comm_val, c_context_val, c_stream_val,
                                     ep_comm_val, tp_comm_val)
    except Exception as e:
        logging.exception(f"[Thread dev={dev_id}] failed:")
        errors[dev_id] = e
        results[dev_id] = ApiProfilingResult.fail("THREAD_EXECUTION_FAILED")


def __init_hccl_comm_for_ranks(hccl_dll, rank_list):
    n = len(rank_list)
    c_devs = (ctypes.c_int32 * n)(*rank_list)
    c_comms = (ctypes.c_void_p * n)()
    ret = hccl_dll.HcclCommInitAll(ctypes.c_uint32(n), c_devs, c_comms)
    if ret != 0:
        raise RuntimeError(f"HcclCommInitAll failed with ret={ret} for ranks {rank_list}")
    handles = []
    for i in range(n):
        v = c_comms[i]
        handles.append(v if isinstance(v, int) else v.value if hasattr(v, 'value') else 0)
    return handles


def __profile_multi_device(context: TestcaseAclnn, device_ids: List[int],
                           device: AclInterface) -> ApiProfilingResult:
    """
    Single-process multi-device execution via threads.
    Each thread gets its own context copy with rank-specific seed.
    Supports dual communication groups (ep + tp) for BatchMatMulReduceScatterAlltoAll.
    """
    ndev = len(device_ids)
    acl_dll = device._acl_dll
    hccl_dll = ctypes.CDLL("libhcomm.so")
    hccl_dll.HcclCommInitAll.restype = ctypes.c_uint32

    is_dual_comm = "BatchMatMulReduceScatter" in context.api_name or "AlltoAllAllGather" in context.api_name

    logging.info(f"Multi-device: aclrtSetDevice for all {device_ids}")
    for did in device_ids:
        try:
            acl_dll.aclrtResetDevice(ctypes.c_int32(did))
        except Exception:
            pass
    for did in device_ids:
        acl_dll.aclrtSetDevice(ctypes.c_int32(did))

    c_devices = (ctypes.c_int32 * ndev)(*device_ids)
    hccl_comms = (ctypes.c_void_p * ndev)()
    logging.info(f"Multi-device: HcclCommInitAll with {device_ids}")
    ret = hccl_dll.HcclCommInitAll(ctypes.c_uint32(ndev), c_devices, hccl_comms)
    if ret != 0:
        raise RuntimeError(f"HcclCommInitAll failed with ret={ret}")
    logging.info(f"Multi-device: HcclCommInitAll success")
    comm_handles = []
    for i in range(ndev):
        v = hccl_comms[i]
        comm_handles.append(v if isinstance(v, int) else v.value
                           if hasattr(v, 'value') else 0)

    ep_comm_map = {}
    tp_comm_map = {}
    all_extra_comms = []
    if is_dual_comm:
        attrs = context.attributes
        ep_ws = int(attrs.get('epWorldSize', 0))
        tp_ws = int(attrs.get('tpWorldSize', 0))
        if ep_ws <= 0 or tp_ws <= 0 or ndev % ep_ws or ndev % tp_ws:
            raise ValueError(
                f"Invalid dual-comm topology: device_count={ndev}, "
                f"epWorldSize={ep_ws}, tpWorldSize={tp_ws}")
        n_ep_groups = ndev // ep_ws
        n_tp_groups = ndev // tp_ws

        # A full-world EP/TP group is exactly the communicator created above.
        # Re-initializing the same full rank set with HcclCommInitAll can block
        # indefinitely on Ascend950, so reuse the primary handles instead.
        if ep_ws == ndev:
            ep_comm_map = dict(enumerate(comm_handles))
        else:
            # EP groups stride by TP width: with ep=2, tp=2 this is
            # [0, 2] and [1, 3], matching the operator reference example.
            for g in range(n_ep_groups):
                grid_base = (g // tp_ws) * ep_ws * tp_ws
                tp_rank = g % tp_ws
                group_ranks = [grid_base + tp_rank + e * tp_ws for e in range(ep_ws)]
                handles = __init_hccl_comm_for_ranks(hccl_dll, group_ranks)
                for i, r in enumerate(group_ranks):
                    ep_comm_map[r] = handles[i]
                all_extra_comms.extend(handles)

        if tp_ws == ndev:
            tp_comm_map = dict(enumerate(comm_handles))
        else:
            # TP groups are contiguous in the same EP/TP grid: [0, 1] and
            # [2, 3] for ep=2, tp=2.
            for g in range(n_tp_groups):
                grid_base = (g // ep_ws) * ep_ws * tp_ws
                ep_rank = g % ep_ws
                group_ranks = [grid_base + ep_rank * tp_ws + t for t in range(tp_ws)]
                handles = __init_hccl_comm_for_ranks(hccl_dll, group_ranks)
                for i, r in enumerate(group_ranks):
                    tp_comm_map[r] = handles[i]
                all_extra_comms.extend(handles)
        logging.info(f"Dual-comm: ep_groups={n_ep_groups} tp_groups={n_tp_groups} "
                     f"ep_ws={ep_ws} tp_ws={tp_ws}")

    c_contexts = [ctypes.c_void_p() for _ in range(ndev)]
    c_streams = [ctypes.c_void_p() for _ in range(ndev)]
    for i, did in enumerate(device_ids):
        acl_dll.aclrtSetDevice(ctypes.c_int32(did))
        acl_dll.aclrtCreateContext(ctypes.pointer(c_contexts[i]), ctypes.c_int32(did))
        acl_dll.aclrtCreateStream(ctypes.pointer(c_streams[i]))
    ctx_vals = [c.value for c in c_contexts]
    stm_vals = [c.value for c in c_streams]

    thread_results: Dict[int, ApiProfilingResult] = {}
    thread_contexts: Dict[int, TestcaseAclnn] = {}
    thread_errors: Dict[int, Exception] = {}
    comm_destroyed_by_thread: set = set()  # ranks whose comm was destroyed in child thread
    pre_exec_barrier = threading.Barrier(ndev, timeout=1800)
    threads = []
    for idx, rank_id in enumerate(device_ids):
        ep_h = ep_comm_map.get(idx, 0)
        tp_h = tp_comm_map.get(idx, 0)
        t = threading.Thread(target=__launch_one_thread,
                             args=(context, rank_id, idx, ndev,
                                   comm_handles[idx], ctx_vals[idx], stm_vals[idx],
                                   thread_results, thread_contexts, thread_errors, device,
                                   ep_h, tp_h, pre_exec_barrier, None, comm_destroyed_by_thread))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    logging.info(f"Multi-device: all threads joined")
    if thread_errors:
        # 执行失败：立即清理（fail-fast），避免泄漏
        __cleanup_multi_device_resources(hccl_dll, acl_dll, device_ids,
                                          comm_handles, ctx_vals, stm_vals,
                                          all_extra_comms, set())
        first_err_dev = min(thread_errors.keys())
        raise thread_errors[first_err_dev]
    # 成功：保留 comm/context/stream 透传给 golden 阶段做真级联，
    # 由 profile_process 在 golden 完成后调用 __cleanup_multi_device_resources。
    context._multi_device_thread_contexts = thread_contexts
    context._multi_device_hccl_handles = {
        'comm_handles': comm_handles,
        'ctx_vals': ctx_vals,
        'stm_vals': stm_vals,
        'extra_comms': all_extra_comms,
    }
    primary_result = thread_results.get(device_ids[0], ApiProfilingResult.fail("UNKNOWN"))
    return primary_result


def __cleanup_multi_device_resources(hccl_dll, acl_dll, device_ids,
                                      comm_handles, ctx_vals, stm_vals,
                                      extra_comms, comm_destroyed_set):
    """统一清理 HCCL comm / stream / context / device。
    comm_destroyed_set 中的 comm 已被销毁，跳过避免 double-destroy SIGSEGV。
    """
    for h in extra_comms:
        try:
            ret = hccl_dll.HcclCommDestroy(ctypes.c_void_p(h))
            logging.info(f"Cleanup: extra HcclCommDestroy ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: extra HcclCommDestroy exception: {e}")
    for i, did in enumerate(device_ids):
        # 必须在每个 device 操作前先 aclrtSetDevice(did)：
        # 多设备场景下主线程 current context 是创建循环里最后 set 的 device（device_ids[-1]），
        # 不切回本 did 就调 HcclCommDestroy / DestroyStream / DestroyContext 会因
        # context 与 handle 不匹配触发 SIGSEGV (exit -11)。
        try:
            ret = acl_dll.aclrtSetDevice(ctypes.c_int32(did))
            logging.info(f"Cleanup: aclrtSetDevice device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtSetDevice device {did} exception: {e}")
        if did in comm_destroyed_set:
            logging.info(f"Cleanup: HcclCommDestroy device {did} skipped (already destroyed)")
        else:
            try:
                ret = hccl_dll.HcclCommDestroy(ctypes.c_void_p(comm_handles[i]))
                logging.info(f"Cleanup: HcclCommDestroy device {did} ret={ret}")
            except Exception as e:
                logging.warning(f"Cleanup: HcclCommDestroy device {did} exception: {e}")
        # 销毁 stream 前先 sync，避免异步 aclnnRun 任务未完成时 destroy 触发 SIGSEGV
        try:
            ret = acl_dll.aclrtSynchronizeStream(ctypes.c_void_p(stm_vals[i]))
            logging.info(f"Cleanup: aclrtSynchronizeStream device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtSynchronizeStream device {did} exception: {e}")
        try:
            ret = acl_dll.aclrtDestroyStream(ctypes.c_void_p(stm_vals[i]))
            logging.info(f"Cleanup: aclrtDestroyStream device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtDestroyStream device {did} exception: {e}")
        try:
            ret = acl_dll.aclrtDestroyContext(ctypes.c_void_p(ctx_vals[i]))
            logging.info(f"Cleanup: aclrtDestroyContext device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtDestroyContext device {did} exception: {e}")
    for did in device_ids:
        try:
            ret = acl_dll.aclrtResetDevice(ctypes.c_int32(did))
            logging.info(f"Cleanup: aclrtResetDevice device {did} ret={ret}")
        except Exception as e:
            logging.warning(f"Cleanup: aclrtResetDevice device {did} exception: {e}")


def __release_retained_multi_device_resources(context: TestcaseAclnn, device_ids: List[int]):
    """Release resources retained for the MC2 golden cascade."""
    handles = getattr(context, '_multi_device_hccl_handles', None)
    if handles is None:
        return
    try:
        __cleanup_multi_device_resources(
            ctypes.CDLL("libhcomm.so"),
            ctypes.CDLL("libascendcl.so"),
            device_ids,
            handles['comm_handles'],
            handles['ctx_vals'],
            handles['stm_vals'],
            handles['extra_comms'],
            set(),
        )
    except Exception:
        logging.exception("Multi-device resource cleanup failed")
    finally:
        context._multi_device_hccl_handles = None


def _invoke_aclnn_npu_preprocess(context, switches, process_ctx, dev_id):
    """Run the optional hook inside the device lock before the main API."""
    func = resolve_npu_preprocess(context, switches)
    if func is None:
        return False
    if getattr(switches, "backend", None) != "npusim":
        import torch_npu

        torch_npu.npu.set_device(dev_id)
    plan = context.get_param_plan()
    args, attributes = plan.build_args(context.tensors, context.scalars, context.attributes)
    process_ctx.notify_status("OnNpuPreprocess")
    return invoke_npu_preprocess(
        context, switches, plan, args, attributes, func=func
    )


def _run_aclnn_npu_preprocess(context, switches, process_ctx, dev_id):
    try:
        _invoke_aclnn_npu_preprocess(context, switches, process_ctx, dev_id)
    except RuntimeError as exc:
        if not str(exc).startswith("NPU_PREPROCESS_FAILURE:"):
            raise
        logging.exception("[%s] NPU preprocess failure", context.testcase_name)
        return str(exc)
    return None


def profile_process(context: TestcaseAclnn,
                    device_grant_events: dict,
                    device_granted_indices: dict,
                    dev_id: int,
                    is_multi_device: bool = False,
                    device_ids: list = None):
    """
    Op Api Testcase Profiling Entrance
    """
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(context.testcase_name)
    if switches.single_testcase_log_mode:
        _log_dir = build_single_log_dir(switches.test_mode, context.api_name, switches.root_path)
        default_logging_config(
            file_handler=switches.logging_to_file, testcase_name=context.testcase_name, log_dir=_log_dir
        )
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
    if is_multi_device and device_ids:
        context.my_rank = dev_id
        context.device_ids = tuple(device_ids)
    else:
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
                prepared_scalars = snapshot_manual_values(
                    context.flatten_scalars or (), "scalar"
                )
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
    use_device = switches.mode.has_device()
    if is_multi_device and device_ids:
        with MultiDeviceLock(process_ctx, device_ids, use_device=use_device,
                             grant_events=device_grant_events,
                             granted_indices=device_granted_indices):
            process_ctx.notify_status("OnProfilingPrint")
            __profiling_print(context, dev_id)
            process_ctx.notify_status("OnProfiling")
            try:
                device = __get_aclnn_device(dev_id)
                context.prof_result = __profile_multi_device(context, list(device_ids), device)
            except Exception:
                logging.exception("Multi-device profiling failed:")
                context.prof_result = ApiProfilingResult.fail("MULTI_DEVICE_FAILED")
    else:
        with DeviceLock(process_ctx, dev_id, use_device=use_device,
                        grant_event=device_grant_events.get(dev_id),
                        granted_idx=device_granted_indices.get(dev_id)):
            preprocess_error = _run_aclnn_npu_preprocess(
                context, switches, process_ctx, dev_id
            )
            if preprocess_error:
                return prof_end(context, preprocess_error)
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
        compare_result = ApiComparisonResult(None)
        compare_result.set("N/A", "N/A")
        process_ctx.notify_status("OnReturning")
        return_structure = ApiProfilingReturnStructure()
        return_structure.construct(context, compare_result)
        __profiling_end_print(context, compare_result)
        return return_structure
    else:
        if is_multi_device and device_ids and switches.golden_mode == "Disable":
            thread_contexts = getattr(context, '_multi_device_thread_contexts', {})
            all_passed = all(
                did in thread_contexts and not thread_contexts[did].prof_result.failed()
                for did in device_ids
            )
            __release_retained_multi_device_resources(context, device_ids)
            compare_result = ApiComparisonResult(None)
            compare_result.set(
                "EXECUTED" if all_passed else "MULTI_DEVICE_EXECUTION_FAILURE",
                "PASS" if all_passed else "FAIL",
            )
            process_ctx.notify_status("OnReturning")
            return_structure = ApiProfilingReturnStructure()
            return_structure.construct(context, compare_result)
            __profiling_end_print(context, compare_result)
            return return_structure

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
            if is_multi_device and device_ids and hasattr(context, '_multi_device_thread_contexts'):
                thread_contexts = context._multi_device_thread_contexts
                all_passed = True
                all_precision = []
                for did in device_ids:
                    tc = thread_contexts.get(did)
                    if tc is None or tc.prof_result.failed():
                        all_passed = False
                        logging.error(f"Multi-device: rank dev={did} execution failed, skipping golden")
                if not all_passed:
                    compare_result = ApiComparisonResult(None)
                    compare_result.set("MULTI_DEVICE_EXECUTION_FAILURE", "FAIL")
                else:
                    __release_retained_multi_device_resources(context, device_ids)
                    __golden_multi_device_compare(thread_contexts, device_ids, all_precision)
                    has_fail = any("FAIL" in p or "EXCEPTION" in p for p in all_precision)
                    compare_result = ApiComparisonResult(None)
                    compare_result.set(",".join(all_precision) if all_precision else "MULTI_DEVICE_COMPARE_FAILURE",
                                       "PASS" if not has_fail else "FAIL")
                    process_ctx.notify_status("OnReturning")
                    return_structure = ApiProfilingReturnStructure()
                    return_structure.construct(context, compare_result)
                    __profiling_end_print(context, compare_result)
                    return return_structure
            elif manual_case is not None:
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
