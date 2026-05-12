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
RTS Profiling Sequence for Universal testcases
"""
# Standard Packages
import contextlib
import ctypes
import os
import json
import math
import logging

import numpy
import pathlib
import re
import subprocess
from typing import Tuple, Union, List, Optional, Dict

# Third-party Packages
from .profiling_structure import RTSProfilingParam, RTSProfilingResult
from ...runtime import RTSInterface, RTSInterfaceBase, rts_info, LaunchKernelArgs
from ...adump import AdxInterface
from ...msprof import MsProfiler, TtkMsProfType, MsProfOpDfx
from ....utilities import get_global_storage, get_dtype_width, extract_csv_cells


def _process_total_cycles(results: list):
    # kick-off UNKNOWN
    kicked_prof_task = [ele for ele in results if ele != "UNKNOWN"]
    # Check if profiling result is valid
    profiling_data_valid = kicked_prof_task and all([isinstance(ele, (int, float)) for ele in kicked_prof_task])
    logging.debug(f"HWTS Task Profiling result: {results}")
    if profiling_data_valid:
        kicked_prof_task.sort()
        return numpy.median(kicked_prof_task)
    else:
        task_data = tuple(map(str, results))
        return ",".join(task_data)


def rts_profiling(device: RTSInterfaceBase, profiling_param: RTSProfilingParam):
    if isinstance(device, RTSInterface):
        online_obj = OnlineRtsProfiling(device, profiling_param)
        return online_obj.do()
    else:
        raise RuntimeError(f"Unrecognized device type: {type(device)}")


class OnlineRtsProfiling:
    INPUT_MEM, OUTPUT_MEM, WS_MEM, DFX_MEM = 0, 1, 2, 3

    def __init__(self, device: RTSInterface, param: RTSProfilingParam):
        self._switches = get_global_storage()
        self._device = device
        self._param = param
        self._registered_binary = None
        self._registered_func = None
        self._run_time = self._switches.run_time
        self._rts_task_fail_cb_set: bool = False
        self._task_fail_cb_invoked: ctypes.c_uint64 = ctypes.c_uint64(0)
        self._prof_type = TtkMsProfType.OP if self._switches.TASK_PROFILING \
                            else TtkMsProfType.NONE
        self._prof_result_path = os.path.join(self._switches.root_path,
                                              "msprof", "op",
                                              self._param.testcase_name,
                                              self._param.run_mode)
        # IN / OUT / WS / DFX
        self._dvc_mem: List[List[Optional[int, list, ctypes.c_void_p]]] = [[], [], [], []]
        # id(np.ndarray) -> device memory
        self._nd_array_maps: Dict[int, ctypes.c_void_p] = {}
        assert self._param.kernel_json_info, "kernel_json_info is None."

    @contextlib.contextmanager
    def rts_context(self):
        reserved_mem_addr = None
        self._device.create_context()
        try:
            if self._switches.reserve_hbm:
                reserved_mem_addr = self._device.malloc(self._switches.reserve_hbm * 1024 * 1024,
                                                        rts_info.RTS_MEMORY_TYPE.RT_MEMORY_HBM,
                                                        "RT_MEMORY_POLICY_HUGE_PAGE_ONLY")
                logging.debug(f"HBM {self._switches.reserve_hbm}MB is reserved.")
            yield
        finally:
            if reserved_mem_addr is not None:
                self._device.free(reserved_mem_addr)
            self._device.destroy_context()
            if self._device.is_model():
                self._device.reset()

    @contextlib.contextmanager
    def register_kernel(self):
        err_ret = self._register_kernel()
        if err_ret is not None:
            yield err_ret
        else:
            try:
                yield
            finally:
                self._device.unregister_device_binary_kernel(self._registered_binary)

    @contextlib.contextmanager
    def rts_stream(self):
        if self._device.is_model():
            yield None
        else:
            stream = self._device.create_stream()
            try:
                yield stream
            finally:
                self._device.destroy_stream_force(stream)

    def do(self):
        with self.rts_context():
            # warmup
            self._device.warmup(self._switches)
            # Kernel Registration
            with self.register_kernel() as register_result:
                if register_result is not None:
                    return register_result
                with self.rts_stream() as stm:
                    # Launch
                    actual_result_byte_arrays, profiling_data = self._rts_kernel_sequence_v2(stm)
            # Cycle Analysis
            if self._device.is_model():
                total_cycle = self._process_model_cycles()
            else:
                total_cycle = _process_total_cycles(profiling_data["TASK"])
            return RTSProfilingResult(total_cycle, actual_result_byte_arrays,
                                      profiling_data["OOB"])

    def _register_kernel(self) -> Optional[RTSProfilingResult]:
        magic = self._param.kernel_json_info.magic
        if isinstance(magic, str):
            magic = self._device.int_magic(magic)
        kernel_full_path = str(pathlib.Path(self._param.kernel_dir, "%s.o" % self._param.kernel_name))
        if not self._kernel_symbol_exists():
            logging.error(f"Kernel register failed. "
                          f"Expected symbol {self._param.kernel_main_func_name} does not exist "
                          f"in {kernel_full_path}. "
                          f"This is usually caused by wrong tiling key.")
            return RTSProfilingResult.fail("RTS_BINARY_FAILURE")
        try:
            if self._param.kernel_json_info.is_fat_bin:
                self._registered_binary = self._device.register_all_kernel(kernel_full_path, magic)
            else:
                self._registered_binary = self._device.register_device_binary_kernel(kernel_full_path, magic)
                self._registered_func = self._device.register_function(self._registered_binary,
                                                                       self._param.kernel_main_func_name)
            return None
        except:
            if self._registered_binary is None:
                logging.exception(f"RTS Register Binary failed, "
                                  f"kernel object {kernel_full_path} does not exist or is invalid.")
                return RTSProfilingResult.fail("RTS_BINARY_FAILURE")
            else:
                logging.error(f"RTS Register Function failed, "
                              f"Expect symbol {self._param.kernel_main_func_name} does not exist "
                              f"in {kernel_full_path}. "
                              f"this is usually caused by wrong tiling key")
                return RTSProfilingResult.fail("RTS_FUNCTION_FAILURE")

    def _rts_kernel_sequence_v2(self, stream: Optional[ctypes.c_void_p] = None):
        # Profiling Preparation
        actual_result_byte_arrays = ["NO_OUTPUT"]
        profiling_data = {"TASK": [],
                          "OOB": "UNKNOWN"}
        status = "OK"
        # Profiling Main Sequence
        try:
            op_dfx = MsProfOpDfx(self._param.kernel_name, self._param.block_dim,
                                 self._param.kernel_json_info.core_type,
                                 self._param.kernel_json_info.task_ration)
            with MsProfiler(self._device.device_id, result_path=self._prof_result_path,
                            ttk_prof_type=self._prof_type, start_step=0,
                            is_model=(self._device.is_model())
                            ) as profiler:
                for repeat_idx in range(self._run_time):
                    profiler.step()
                    self._device.clear_l1(self._switches)
                    self._device.clear_ub(self._switches)
                    # self._device.test_clear_ub(self._switches)
                    # Prepare Memory on HBM
                    self._alloc_device_memory(repeat_idx)
                    # Launch Kernel
                    self._kernel_assert_setup()
                    # noinspection PyBroadException
                    try:
                        self._launch_kernel(stream)
                        profiler.report_op_dfx(op_dfx)
                    except:
                        logging.exception("RTSProfilingCall encountered an unknown rts error "
                                          "during kernel launch stage:")
                        status = "LAUNCH_FAILED"
                    else:
                        try:
                            rts_timeout = 0 if self._device.is_model() else self._switches.run_timeout
                            self._device.synchronize_with_stream(stream, timeout=rts_timeout)
                        except RuntimeError as e:
                            status = self._device.handle_stream_sync_exception(e.args[0])
                        if repeat_idx == 0:
                            self._print_dump_workspace()
                    # Collect Data
                    profiling_data["TASK"].append(status)
                    if repeat_idx == self._run_time - 1 and status == "OK":
                        actual_result_byte_arrays = self._copy_output_from_hbm()
                        profiling_data["OOB"] = self._check_output_out_of_bounds()
                    # Free memory
                    if not self._switches.reuse_hbm:
                        self._free_device_memory()
                    if status != "OK":
                        break
        finally:
            self._free_device_memory()
        if self._switches.TASK_PROFILING:
            task_duration = self._process_msprof_cycles()
            if isinstance(task_duration, str):
                # check TASK execute status.
                if all([x == "OK"] for x in profiling_data["TASK"]):
                    # no msprof python tool
                    profiling_data["TASK"] = ["UNKNOWN"]
            else:
                profiling_data["TASK"] = [task_duration]
        return actual_result_byte_arrays, profiling_data

    def _alloc_device_memory(self, repeat_idx: int):
        if repeat_idx == 0 or not self._switches.reuse_hbm:
            self._alloc_hbm()
        else:
            self._reuse_hbm()

    def _get_inplace_array_ipt_ids(self) -> list:
        ipt_ids = [id(i) for i in self._param.flatten_input_arrays
                   if isinstance(i, numpy.ndarray)]
        inplace_ids = []
        for o in self._param.flatten_output_arrays:
            if isinstance(o, numpy.ndarray):
                if id(o) in ipt_ids:
                    inplace_ids.append(id(o))
        return inplace_ids

    def _alloc_hbm(self):
        inplace_ids = self._get_inplace_array_ipt_ids()
        self._alloc_input_hbm(inplace_ids)
        self._alloc_output_hbm(inplace_ids)
        self._alloc_workspace_hbm()
        self._alloc_dfx_hbm()

    def _alloc_input_hbm(self, inplace_ids: list):
        for ia in self._param.input_arrays:
            if ia is None:
                mem = ctypes.c_void_p(None)
            elif isinstance(ia, int):
                # some outer users wanna to set a B64 argument.
                # for example: AscendC
                mem = ia
            elif isinstance(ia, (list, tuple)):
                # tensor list
                mem = []
                for t in ia:
                    m = self._device.copy_nparray_to_hbm(t, fill_oob_flag=id(t) in inplace_ids)
                    mem.append(m)
                    self._nd_array_maps[id(t)] = m
            else:
                # tensor
                mem = self._device.copy_nparray_to_hbm(ia, fill_oob_flag=id(ia) in inplace_ids)
                self._nd_array_maps[id(ia)] = mem
            self._dvc_mem[self.INPUT_MEM].append(mem)

    def _alloc_output_hbm(self, inplace_ids: list):
        for oa in self._param.output_arrays:
            if oa is None:
                mem = ctypes.c_void_p(None)
            elif isinstance(oa, (list, tuple)):
                # tensor list
                refs = []
                for t in oa:
                    if isinstance(t, numpy.ndarray):
                        if id(t) in inplace_ids:
                            refs.append(id(t))
                if refs:
                    # inplace dynamic tensor-list
                    if len(refs) != len(oa):
                        raise RuntimeError(f"Dynamic output with partial inplace inputs.")
                    # find the inplace-ed input tensor list with address.
                    mem = None
                    for ipt_idx, ipt in enumerate(self._param.input_arrays):
                        if isinstance(ipt, (list, tuple)):
                            ipts = [id(x) for x in ipt]
                            if sorted(ipts) == sorted(refs):
                                mem = self._dvc_mem[self.INPUT_MEM][ipt_idx]
                                break
                    if mem is None:
                        raise RuntimeError(f"Dynamic output inplace-ed dynamic input is not found.")
                else:
                    # non-inplace dynamic tensor-list
                    mem = []
                    for t in oa:
                        m = self._device.copy_nparray_to_hbm(t, fill_oob_flag=True)
                        mem.append(m)
                        self._nd_array_maps[id(t)] = m
            elif id(oa) in inplace_ids:
                # inplace tensor
                mem = self._nd_array_maps[id(oa)]
            else:
                # tensor
                mem = self._device.copy_nparray_to_hbm(oa, fill_oob_flag=True)
                self._nd_array_maps[id(oa)] = mem
            self._dvc_mem[self.OUTPUT_MEM].append(mem)

    def _alloc_workspace_hbm(self):
        for ws in self._param.workspace_arrays:
            if ws is None:
                mem = ctypes.c_void_p(None)
            else:
                mem = self._device.copy_nparray_to_hbm(ws, fill_oob_flag=True)
                self._nd_array_maps[id(ws)] = mem
            self._dvc_mem[self.WS_MEM].append(mem)

    def _alloc_dfx_hbm(self):
        for da in self._param.dfx_arrays:
            if da is None:
                mem = ctypes.c_void_p(None)
            else:
                mem = self._device.copy_nparray_to_hbm(da)
                self._nd_array_maps[id(da)] = mem
            self._dvc_mem[self.DFX_MEM].append(mem)

    def _reuse_hbm(self):
        self._reuse_input_hbm()
        # do not touch outputs to match scenario of other tools when reusing hbm.
        self._reuse_common_hbm(self._param.workspace_arrays)
        self._reuse_common_hbm(self._param.dfx_arrays)

    def _reuse_input_hbm(self):
        # only recover the inplace inputs
        inplace_ids = self._get_inplace_array_ipt_ids()
        for o in self._param.flatten_output_arrays:
            if not isinstance(o, numpy.ndarray):
                continue
            if id(o) in inplace_ids:
                mem = self._nd_array_maps[id(o)]
            else:
                continue
            self._device.copy_nparray_to_hbm_ptr(o, mem)

    def _reuse_common_hbm(self, arrays: tuple):
        for a in arrays:
            if a is None:
                continue
            else:
                mem = self._nd_array_maps[id(a)]
                self._device.copy_nparray_to_hbm_ptr(a, mem)

    def _free_device_memory(self):
        for nd_array_id, dev_address in self._nd_array_maps.items():
            if not isinstance(dev_address, ctypes.c_void_p):
                continue
            if not dev_address.value:  # nullptr: None or 0
                continue
            self._device.free(dev_address)
        self._nd_array_maps.clear()
        for x_mem in self._dvc_mem:
            x_mem.clear()

    def _copy_output_from_hbm(self) -> list:
        byte_results = []
        for oa in self._param.flatten_output_arrays:
            if oa is None:  # nullptr
                byte_results.append(None)
                continue
            hbm_addr = self._nd_array_maps[id(oa)]
            byte_size = int(math.ceil(oa.size * get_dtype_width(oa.dtype)))
            data = self._device.get_data_from_hbm(hbm_addr, byte_size)
            byte_results.append(data)
        return byte_results

    def _check_output_out_of_bounds(self) -> str:
        oob_results = []
        total_arrays = self._param.flatten_output_arrays + self._param.workspace_arrays
        output_count = len(self._param.flatten_output_arrays)
        for idx, a in enumerate(total_arrays):
            if a is None:  # nullptr
                oob_results.append("OK")
                continue
            hbm_addr = self._nd_array_maps[id(a)]
            array_size = int(math.ceil(a.size * get_dtype_width(a.dtype)))
            oob_offset, oob_size = self._device.get_oob_check_offset_bytes(array_size)
            hbm_addr = ctypes.c_void_p(hbm_addr.value + oob_offset)
            oob_bytes = self._device.get_data_from_hbm(hbm_addr, oob_size)
            oob_sentinels = numpy.frombuffer(oob_bytes, dtype="uint8")
            if numpy.all(oob_sentinels == self._device.OOB_SENTINEL):
                oob_results.append("OK")
            else:
                oob_results.append("FAIL")
                log_idx = idx if idx < output_count else idx - output_count
                log_info = "output" if idx < output_count else "workspace"
                logging.error(f"The {log_idx}th (count from 0) {log_info} "
                              f"memory out of bound !! "
                              f"All data in the list below should all be "
                              f"{self._device.OOB_SENTINEL} (uint8), "
                              f"but got:\n{oob_sentinels}")
        return ','.join(oob_results)

    def _pack_launch_op_args(self):
        mem_group = [
            self._dvc_mem[self.INPUT_MEM],
            self._dvc_mem[self.WS_MEM]
        ]
        arr_group = [
            self._param.input_arrays,
            self._param.workspace_arrays
        ]

        if self._param.output_placeholder:  # likely
            mem_group.insert(1, self._dvc_mem[self.OUTPUT_MEM])
            arr_group.insert(1, self._param.output_arrays)

        op_args = [item for sublist in mem_group for item in sublist]
        nd_arrays = [item for sublist in arr_group for item in sublist]

        for idx, arg in enumerate(op_args):
            if isinstance(arg, (list, tuple)):
                tensor_list = self._device.prepare_tensor_list_info(
                    arg, nd_arrays[idx])
                op_args[idx] = tensor_list

        return op_args

    def _launch_kernel(self, stream: Optional[ctypes.c_void_p] = None):
        op_args = self._pack_launch_op_args()
        mix_kernel = self._param.kernel_json_info.is_mix_kernel
        if self._param.kernel_json_info.is_fat_bin:
            launch_args = LaunchKernelArgs(
                                func_or_binary_hdl=self._registered_binary,
                                op_args=op_args, dfx_args=self._dvc_mem[self.DFX_MEM],
                                block_dim=self._param.block_dim,
                                tiling_key=self._param.tiling_key,
                                tiling_data=self._param.tiling_data,
                                mix_kernel=mix_kernel,
                                schedule_mode=self._param.kernel_json_info.schedule_mode,
                                simt_share_memory_size=self._param.simt_share_memory_size,
                            )
            self._device.launch_kernel_with_handle(launch_args, stream)
        else:
            launch_args = LaunchKernelArgs(
                                func_or_binary_hdl=self._registered_func,
                                op_args=op_args, dfx_args=self._dvc_mem[self.DFX_MEM],
                                block_dim=self._param.block_dim,
                                mix_kernel=mix_kernel,
                                tiling_data=self._param.tiling_data,
                                schedule_mode=self._param.kernel_json_info.schedule_mode,
                                simt_share_memory_size=self._param.simt_share_memory_size
                            )
            self._device.launch_kernel_with_flag(launch_args, stream)

    def _process_model_cycles(self):
        return "UNKNOWN"

    def _process_msprof_cycles(self):
        KERNEL_TYPE = ("AI_CORE", "AIV_SQE", "AI_VECTOR_CORE",
                       "MIX_AIC", "MIX_AIV",
                       "KERNEL_AIVEC", "KERNEL_AICORE")
        total_cycle = "UNKNOWN"
        prof_result_path = pathlib.Path(self._prof_result_path)
        if not prof_result_path.is_dir():
            return total_cycle
        csv_files = list(prof_result_path.glob("**/*.csv"))
        if not csv_files:
            return total_cycle

        op_prof = []
        for item in csv_files:
            if op_prof:
                break
            if item.name.startswith("op_summary_"):
                op_prof = extract_csv_cells(item, ('Task Duration(us)',),
                                            col_name_mapping={'Task Duration(us)': 'duration'},
                                            cmp=lambda row: row["Task Type"] in KERNEL_TYPE)
            elif item.name.startswith("task_time_"):
                op_prof = extract_csv_cells(item, ('task_time(us)',),
                                            col_name_mapping={'task_time(us)': 'duration'},
                                            cmp=lambda row: row["kernel_type"] in KERNEL_TYPE)
            else:
                pass

        if op_prof:
            op_prof = list(x['duration'] for x in op_prof)
            results = [op_prof[0]]
            if len(op_prof) > 1:
                results = op_prof[1:]
            logging.debug(f"MsProf Task Profiling result: {results}")
            total_cycle = numpy.round(numpy.median(results), 3)

        return total_cycle

    def _kernel_symbol_exists(self):
        kernel_full_path = str(pathlib.Path(self._param.kernel_dir, "%s.o" % self._param.kernel_name))
        cmds = ["nm", kernel_full_path]
        try:
            out = subprocess.check_output(cmds).decode('utf-8')
        except:
            pass
        else:
            if out:
                splits = re.split(r'[ \n]', out)
                exist = self._param.kernel_main_func_name in splits
                if not exist and self._param.kernel_json_info.is_mix_kernel:
                    exist = f"{self._param.kernel_main_func_name}_mix_aic" in splits \
                            or f"{self._param.kernel_main_func_name}_mix_aiv" in splits
                return exist
        return True

    def _print_dump_workspace(self):
        if not self._dvc_mem[self.WS_MEM]:
            return
        first_workspace_addr = self._dvc_mem[self.WS_MEM][0]

        if self._param.kernel_json_info.printf_enabled() or \
                (self._param.kernel_json_info.assert_enabled() and
                 self._task_fail_cb_invoked.value == 1):
            try:
                debug_buf_size = self._param.kernel_json_info.debug_buf_size
                c_buffer = self._device.get_data_from_hbm(first_workspace_addr, debug_buf_size)
                AdxInterface.print_dump_info(ctypes.c_void_p(ctypes.addressof(c_buffer)),
                                             debug_buf_size,
                                             self._switches.short_soc_version)
            except RuntimeError:
                return

    def _kernel_assert_setup(self):
        self._task_fail_cb_invoked = ctypes.c_uint64(0)
        if not self._param.kernel_json_info.assert_enabled():
            return
        if not self._dvc_mem[self.WS_MEM]:
            return
        if not self._rts_task_fail_cb_set:
            self._device.set_task_fail_callback()
            self._rts_task_fail_cb_set = True
        # rtSetExceptionExtInfo
        try:
            self._device.set_exception_extend_info(
                ctypes.c_void_p(ctypes.addressof(self._task_fail_cb_invoked)))
        except RuntimeError:
            # if fail, just pass.
            pass
