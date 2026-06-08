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
from ...testcase_manager import TestcaseAclnn
from ...tbe_multiprocessing import get_process_context, DeviceLock
from ...tbe_logging import default_logging_config
from ...aclnn import AclInterface, OpApiInfoKeeper, OpApiInfo
from ...msprof import MsProfiler, TtkMsProfType
from ....utilities import get_global_storage, get, waiting_for_memory, frameless_table_print
from ....utilities import apply_as_list, resolve_custom_numpy_dtypes, dump_to_file, extract_plog_errors


def __profiling_end_print(context: TestcaseAclnn,
                          compare_result: ApiComparisonResult):
    c = compare_result
    logging.info("\n########################\n"
                 f"GOLD: {c.precision}\n"
                 f"PRECISION_STATUS: {c.passed}\n"
                 "########################\n")


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
            scalar_values.append(tuple(
                s.item() if s is not None else None for s in group
            ))
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
        device = AclInterface(switches.short_soc_version,
                              switches.mode.is_model())
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
            if kind == 'tensor':
                case_params.append(acl_tensors.pop(0))
            elif kind == 'scalar':
                case_params.append(acl_scalars.pop(0))
            else:
                # consider remaining as attribute.
                val = self._ctx.attributes.get(param_name, default)
                if "Array" in acl_type:
                    # aclBoolArray/aclIntArray/aclFloatArray
                    typ = acl_type[3:acl_type.index('Array')]
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
                        if hasattr(np_storage.base, '__array_interface__'):
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
        self._prof_result_path = os.path.join(self._switches.root_path,
                                              "msprof", "op_api",
                                              self._ctx.testcase_name)

    @contextlib.contextmanager
    def rts_context(self):
        self._dvc.create_context()
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
                output_byte_arrays, output_view_shapes, success = self._acl_sequence(stm)
            # Cycle Analysis
            if self._dvc.is_model():
                # TODO
                logging.info("TODOOOOOOOOOOOOOOOOOOOOOO")
                api_prof = "UNKNOWN"
                op_prof = "TOTAL_CYCLE_TODO"
            else:
                api_prof, op_prof = self._process_total_cycles()
            return ApiProfilingResult(success, 
                                      api_prof, op_prof, 
                                      output_byte_arrays, 
                                      output_view_shapes, )

    @staticmethod
    def _extract_csv_cell(filename, extract_cols,
                          cmp=None) -> list:
        results = []
        with open(filename, 'r') as f:
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
        try:
            prof_start_at = 1 if self._run_time > 1 else 0
            with MsProfiler(self._dvc.device_id, result_path=self._prof_result_path,
                            ttk_prof_type=self._prof_type, start_step=prof_start_at,
                            is_model=self._dvc.is_model()) as profiler:
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
                        workspace_size, c_executor = self._dvc.acl_get_workspace(self._ctx.api_name,
                                                                                 phase1_params)
                        # call phase 2 interface
                        status = self._dvc.acl_execute(self._ctx.api_name, workspace_size,
                                                       c_executor, stream)
                    except Exception as e:
                        time.sleep(0.5)
                        plog_errors = extract_plog_errors()
                        if plog_errors:
                            logging.error(f"aclnn interface {self._ctx.api_name} execute failed: \n"
                                          f"***************************************************************************\n"
                                          f"{os.linesep.join(plog_errors)}\n"
                                          f"***************************************************************************")
                        else:
                            error_detail = str(e)
                            logging.exception(f"aclnn interface {self._ctx.api_name} execute failed:\n{error_detail}")
                        status = "ACLNN_EXECUTE_FAILED"

                    if repeat_idx == self._run_time - 1 and status == "OK":
                        # copy output (tensor storage data) from device
                        output_byte_arrays = self._phase1_param_builder.copy_output_from_hbm()
                        output_view_shapes = self._phase1_param_builder.collect_output_view_shapes()
                    self._dvc.free_all_memory()
                    if status != "OK":
                        break
        finally:
            self._dvc.free_all_memory()
        return output_byte_arrays, output_view_shapes, status == "OK"

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
                api_prof = self._extract_csv_cell(item, ('API Name', 'Time(us)', 'Count',
                                                         'Avg(us)', 'Min(us)', 'Max(us)'),
                                                  cmp=lambda row: row["Level"] == "acl")
            elif item.name.startswith("op_statistic_"):
                op_prof = self._extract_csv_cell(item, ('OP Type', 'Total Time(us)', 'Count',
                                                        'Avg Time(us)', 'Min Time(us)', 'Max Time(us)',
                                                        'Core Type'))
                for p in op_prof:
                    p['OP Type'] = p['OP Type'] + ("_AiCpu" if 'cpu' in p['Core Type'].lower()
                                                   else "_AiCore")
                    del p['Core Type']
        # print
        lines = [['Name', 'Total/us', 'Avg/us', 'Min/us', 'Max/us', '# of Calls']]
        if not isinstance(api_prof, str):
            lines.extend([[item['API Name'], "%.2f" % item['Time(us)'],
                           "%.2f" % item['Avg(us)'], "%.2f" % item['Min(us)'],
                           "%.2f" % item['Max(us)'], int(item['Count'])]
                          for item in api_prof])
        if not isinstance(op_prof, str):
            lines.extend([[item['OP Type'], "%.2f" % item['Total Time(us)'],
                           "%.2f" % item['Avg Time(us)'], "%.2f" % item['Min Time(us)'],
                           "%.2f" % item['Max Time(us)'], int(item['Count'])]
                          for item in op_prof])
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


def __dump_to_file(data, file_name: str, dtype: Optional[str] = None):
    switches = get_global_storage()
    file_path = os.getenv("NPU_DUMP_PATH") or switches.root_path
    dump_to_file(data, file_path, file_name,
                 file_format=switches.dump_config.file_format,
                 dtype=dtype)


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
            __dump_to_file(_output, f"{dump_output_name}_output_{idx}",
                           get(output_dtypes, idx))


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


def profile_process(context: TestcaseAclnn,
                    device_grant_events: dict,
                    device_granted_indices: dict,
                    dev_id: int):
    """
    Op Api Testcase Profiling Entrance
    """
    switches = get_global_storage()
    process_ctx = get_process_context()
    process_ctx.change_name(context.testcase_name)
    if switches.single_testcase_log_mode:
        default_logging_config(file_handler=switches.logging_to_file,
                               testcase_name=context.testcase_name)
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
    process_ctx.notify_status("OnGenInput")
    # noinspection PyBroadException
    try:
        InputGenerator(context).gen()
    except:
        logging.exception("Input data generation failure:")
        return prof_end(context, "INPUT_GEN_FAILURE")
    # TODO
    process_ctx.notify_status("OnDumpInputDataIfRequired")
    __dump_input(context)
    # Following actions need to acquire global lock
    process_ctx.notify_status("OnAcquireLock")
    use_device = switches.mode.use_device()
    with DeviceLock(process_ctx, dev_id, use_device=use_device,
                    grant_event=device_grant_events.get(dev_id),
                    granted_idx=device_granted_indices.get(dev_id)):
        process_ctx.notify_status("OnProfilingPrint")
        __profiling_print(context, dev_id)
        process_ctx.notify_status("OnProfiling")
        context.prof_result = do_profiling(context, dev_id)
    if context.prof_result.failed():
        context.golden_tensors = context.prof_result.api_prof
    else:
        process_ctx.notify_status("OnGenGolden")
        # noinspection PyBroadException
        try:
            GoldenGenerator(context).gen()
            process_ctx.notify_status("OnDumpGoldenDataIfRequired")
            __dump_golden(context)
        except:
            logging.exception("Golden data generation failure")
    process_ctx.notify_status("OnDumpOutputDataIfRequired")
    __dump_output(context)
    process_ctx.notify_status("OnComparison")
    compare_result = Comparator(context).compare()
    if compare_result.passed != "PASS" and switches.dump_config.dump_on_fail:
        __dump_on_fail(context)
    process_ctx.notify_status("OnReturning")
    return_structure = ApiProfilingReturnStructure()
    return_structure.construct(context, compare_result)
    __profiling_end_print(context, compare_result)
    return return_structure
