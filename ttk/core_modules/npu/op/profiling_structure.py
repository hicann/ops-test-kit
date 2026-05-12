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
Dynamic Shape NPU Profiling Structure
"""
# Standard Packages
import logging
import math
import numpy
import os
from typing import Any, Optional, Sequence, Tuple, Union
# Third-party Packages
from ...testcase_manager import UniversalTestcaseStructure
from ....utilities import BaseCompilationResult
from ....utilities import (
    get_global_storage,
    align,
    input_apply_as_list,
    output_apply_as_list
)


class RTSProfilingParam:
    def __init__(self,
                 compile_result: BaseCompilationResult,
                 input_arrays: tuple,
                 output_arrays: tuple,
                 workspace_arrays: tuple,
                 tiling_data_bytes: bytes,
                 output_placeholder: bool,
                 clear_atomic: bool,
                 switch: bool,
                 is_valid: bool,
                 fail_reason: str,
                 tensor_list_distribution: Optional[tuple],
                 testcase_name: str,
                 run_mode: str):
        self.compile_result = compile_result.compile_result
        self.tiling_key: Optional[int] = compile_result.tiling_key
        self.kernel_name = compile_result.kernel_name
        self.block_dim = int(compile_result.block_dim)
        self.kernel_dir = compile_result.kernel_dir or get_global_storage().kernel_meta
        if not os.path.isabs(self.kernel_dir):  # in case model && --cce
            self.kernel_dir = os.path.join(get_global_storage().root_path, self.kernel_dir)
        self.simt_share_memory_size: int = int(compile_result.simt_ub_size)
        self.switch = switch
        self.is_valid = is_valid
        self.fail_reason = fail_reason
        self.testcase_name = testcase_name
        self.run_mode = run_mode
        self.kernel_json_info = compile_result.kernel_json_info
        self.output_placeholder = output_placeholder
        self.workspace_arrays: Tuple[numpy.ndarray] = workspace_arrays
        self.flatten_input_arrays = input_arrays
        self.flatten_output_arrays = output_arrays
        if self.is_valid and self.block_dim > 0 and self.compile_result == 'SUCC':
            self.flatten_output_arrays = tuple([
                input_arrays[o] if isinstance(o, int) else o
                for o in output_arrays])
        # private params
        self._tiling_data_bytes = tiling_data_bytes
        self._clear_atomic = clear_atomic
        self._tensor_list_distribution = tensor_list_distribution
        self._dfx_arrays: Optional[tuple] = None
        self._input_arrays: Optional[tuple] = None
        self._output_arrays: Optional[tuple] = None
        self._ipt_ids = [id(i) for i in self.flatten_input_arrays
                         if isinstance(i, numpy.ndarray)]

    @property
    def kernel_main_func_name(self) -> str:
        return self.kernel_json_info.kernel_name

    @property
    def input_arrays(self) -> Optional[tuple]:
        if self._input_arrays is not None:
            return self._input_arrays
        if not self.kernel_json_info:  # only when compile failed.
            return None

        # handle tensor list.
        if (
            self.kernel_json_info.dynamic_param_is_folded() and
            self._tensor_list_distribution
        ):
            input_arrays = input_apply_as_list(self.flatten_input_arrays,
                                               self._tensor_list_distribution)
        else:
            input_arrays = self.flatten_input_arrays

        # handle nullptr placeholder.
        if not self.kernel_json_info.optional_input_gen_placeholder():
            input_arrays = [i for i in input_arrays if i is not None]

        self._input_arrays = tuple(input_arrays)
        return self._input_arrays

    @property
    def output_arrays(self) -> Optional[tuple]:
        if self._output_arrays is not None:
            return self._output_arrays
        if not self.kernel_json_info:  # only when compile failed.
            return None

        # handle tensor list.
        if (
            self.kernel_json_info.dynamic_param_is_folded() and
            self._tensor_list_distribution
        ):
            output_arrays = output_apply_as_list(self.flatten_output_arrays,
                                                 self._tensor_list_distribution,
                                                 input_count=len(self.flatten_input_arrays))
        else:
            output_arrays = self.flatten_output_arrays

        # handle nullptr placeholder.
        if not self.kernel_json_info.optional_output_gen_placeholder():
            output_arrays = [o for o in output_arrays if o is not None]

        self._output_arrays = tuple(output_arrays)
        return self._output_arrays

    @property
    def tiling_data(self) -> Optional[bytes]:
        if self._tiling_data_bytes is not None:
            if get_global_storage().oom_enabled() or self.kernel_json_info.oom_enabled():
                op_original_para_size = self.kernel_json_info.op_original_para_size
                tiling_data_len = len(self._tiling_data_bytes)
                if op_original_para_size > 0 and tiling_data_len < op_original_para_size:
                    tiling_data_len = op_original_para_size
                align_size = align(tiling_data_len, 8)
                tiling_data = self._tiling_data_bytes.ljust(align_size, b'\x00')
                import struct
                tensors_size = self._construct_kernel_asan_info()
                tiling_data += struct.pack('<' + 'q' * len(tensors_size), *tensors_size)
                logging.debug(f"Tiling data with oom info is :\n"
                              f"{numpy.frombuffer(tiling_data, dtype=numpy.int64)}")
                return tiling_data
            else:
                return self._tiling_data_bytes
        else:
            return None

    @property
    def dfx_arrays(self) -> tuple:
        if self._dfx_arrays is not None:
            return self._dfx_arrays
        global_workspace_size = self.kernel_json_info.global_workspace_size
        dfx_arrays = []
        if global_workspace_size > 0:
            da = numpy.zeros(global_workspace_size, dtype=numpy.byte)
            dfx_arrays.append(da)
        self._dfx_arrays = tuple(dfx_arrays)
        return self._dfx_arrays

    def has_tensor_list(self) -> bool:
        return (any(True for i in self.input_arrays
                    if isinstance(i, (list, tuple)))
                or
                any(True for o in self.output_arrays
                    if isinstance(o, (list, tuple)))
                )

    def clear_atomic_output_workspace(self):
        self._init_outputs()
        if not self.kernel_json_info:
            # only when compile failed.
            return

        expected_param_num = len(self.kernel_json_info.parameters)
        knl_json_ws_num = len(self.kernel_json_info.workspaces or ())
        input_arrays_num = len(self.input_arrays)
        output_arrays_num = len(self.output_arrays) if self.output_placeholder else 0
        ws_num = len(self.workspace_arrays)

        actual_param_num = (input_arrays_num
                            + output_arrays_num
                            + ws_num
                            + (1 if self._tiling_data_bytes is not None else 0)
                            )
        if (
            expected_param_num != 0 and
            (actual_param_num - expected_param_num) not in (0, knl_json_ws_num)
        ):
            logging.warning(f"CCE Expects {expected_param_num} arguments, "
                            f"received {actual_param_num}")

        if not self._clear_atomic:
            return
        if not self.kernel_json_info.clear_atomic:  # force clear atomic
            self._force_clear_atomic()
        else:
            for idx, p in enumerate(self.kernel_json_info.parameters):
                if idx < input_arrays_num:
                    if isinstance(p, dict) or p == 1:
                        logging.warning("Compile JSON gives input array atomic sign, "
                                        "something must be wrong here!")
                elif idx < input_arrays_num + output_arrays_num:
                    output_idx = idx - input_arrays_num
                    if not isinstance(p, dict) and p != 1:
                        pass
                    elif isinstance(self.output_arrays[output_idx], (list, tuple)):
                        # tensor list
                        logging.warning("Compile JSON gives tensor list atomic sign, "
                                        "something must be wrong here!")
                    elif (
                        isinstance(self.output_arrays[output_idx], numpy.ndarray) and
                        id(self.output_arrays[output_idx]) in self._ipt_ids  # reference.
                    ):
                        logging.warning("Compile JSON gives inplace output array atomic sign, "
                                        "something must be wrong here!")
                    else:
                        atomic_value = 0 if not isinstance(p, dict) else p.get("init_value", 0)
                        atomic_value = self.output_arrays[output_idx].dtype.type(atomic_value)
                        self.output_arrays[output_idx].fill(atomic_value)
                elif idx < input_arrays_num + output_arrays_num + ws_num:
                    if not isinstance(p, dict) and p != 1:
                        pass
                    else:
                        workspace_idx = idx - input_arrays_num - output_arrays_num
                        atomic_value = 0 if not isinstance(p, dict) else p.get("init_value", 0)
                        atomic_value = self.workspace_arrays[workspace_idx].dtype.type(atomic_value)
                        self.workspace_arrays[workspace_idx].fill(atomic_value)
                elif isinstance(p, dict) or p == 1:
                    logging.warning("Compile JSON gives tiling_data array atomic sign, "
                                    "something terrible happens!")
                else:
                    pass

    def _bytes(self, tensor: Optional[Union[numpy.ndarray, int]]):
        if tensor is None:
            return 0
        else:
            return align(tensor.nbytes if ("int4" not in tensor.dtype.name
                                           and "float4" not in tensor.dtype.name)
                         else int(math.ceil(tensor.nbytes / 2)),
                         32)

    def _append_xput_asan_info(self,
                               xputs: Union[list, tuple],
                               tensors_size: list):
        for x in xputs:
            if x is None:
                tensors_size.append(0)
            elif isinstance(x, (list, tuple)):
                data_size = 8  # ptr offset
                for tensor in x:
                    if tensor is None:
                        continue
                    # dimNum (int32) / count (int32) / dim (int64) / ptr (int64)
                    data_size += 2 + 2 + tensor.ndim * 8 + 8
                tensors_size.append(data_size)
            else:
                tensors_size.append(self._bytes(x))

    def _construct_kernel_asan_info(self) -> list:
        tensors_size = []
        # inputs
        self._append_xput_asan_info(self.input_arrays, tensors_size)
        # outputs
        if self.output_placeholder:
            self._append_xput_asan_info(self.output_arrays, tensors_size)
        # workspace
        self._append_xput_asan_info(self.workspace_arrays, tensors_size)
        return tensors_size

    def _init_outputs(self):
        for o in self.flatten_output_arrays:
            if o is None or id(o) in self._ipt_ids:
                continue
            # recover to init value 1 for non-atomic-clean
            if o.size > 0 and o.take(0) != o.dtype.type(1):
                o.fill(o.dtype.type(1))

    def _force_clear_atomic(self):
        for idx in range(len(self.flatten_output_arrays)):
            if not isinstance(self.flatten_output_arrays[idx], int):
                self.flatten_output_arrays[idx].fill(self.flatten_output_arrays[idx].dtype.type(0))
        for idx in range(len(self.workspace_arrays)):
            if self.workspace_arrays[idx] is not None:
                self.workspace_arrays[idx].fill(self.workspace_arrays[idx].dtype.type(0))


class RTSProfilingResult:
    """
    RTS Profiling output
    """

    def __init__(self, cycle=None, output_bytes=(None,), oob: str = "UNKNOWN"):
        self.cycle: Optional[Union[str, int, float, tuple]] = cycle
        self.output_bytes: Optional[Union[tuple, list]] = output_bytes
        self.oob: Optional[str] = oob

    @classmethod
    def fail(cls, fail_result: str) -> "RTSProfilingResult":
        return cls(fail_result,
                   (fail_result,),
                   "UNKNOWN")

    @property
    def oob_status(self):
        if not self.oob:
            return "PASS"
        oob_lst = self.oob.split(',')
        return "FAIL" if "FAIL" in oob_lst else "PASS"


class ProfilingReturnStructure:
    """
    Structure for Return
    """

    __slots__ = ("dyn_tiling_time_us",
                 "bin_tiling_time_us",
                 "dyn_kernel_name",
                 "cst_kernel_name",
                 "bin_kernel_name",
                 "dyn_block_dim",
                 "cst_block_dim",
                 "bin_block_dim",
                 "dyn_perf_us",
                 "cst_perf_us",
                 "bin_perf_us",
                 "dyn_compile_s",
                 "cst_compile_s",
                 "bin_compile_s",
                 "perf_status",
                 "dyn_precision",
                 "cst_precision",
                 "bin_precision",
                 "precision_status",
                 "dyn_oob_result",
                 "cst_oob_result",
                 "bin_oob_result",
                 "memory_oob_status",
                 "dyn_tiling_key",
                 "dyn_tiling_data",
                 "bin_tiling_key",
                 "bin_tiling_data",
                 "dyn_workspaces",
                 "cst_workspaces",
                 "bin_workspaces",
                 "data_input_size_b",
                 "data_output_size_b",
                 "soc")

    def __init__(self, default_value=None):
        self.dyn_tiling_time_us = default_value
        self.bin_tiling_time_us = default_value
        # DYN
        self.dyn_kernel_name = default_value
        self.dyn_block_dim = default_value
        self.dyn_perf_us = default_value
        self.dyn_compile_s = default_value
        self.dyn_precision = default_value
        self.dyn_oob_result = default_value
        # CST
        self.cst_kernel_name = default_value
        self.cst_block_dim = default_value
        self.cst_perf_us = default_value
        self.cst_compile_s = default_value
        self.cst_precision = default_value
        self.cst_oob_result = default_value
        # BIN
        self.bin_kernel_name = default_value
        self.bin_block_dim = default_value
        self.bin_perf_us = default_value
        self.bin_compile_s = default_value
        self.bin_precision = default_value
        self.bin_oob_result = default_value

        # Performance
        self.perf_status = default_value
        # Precision
        self.precision_status = default_value
        # Memory result
        self.memory_oob_status = default_value
        # Special
        self.data_input_size_b = default_value
        self.data_output_size_b = default_value
        self.dyn_tiling_data = default_value
        self.dyn_tiling_key = default_value
        self.bin_tiling_data = default_value
        self.bin_tiling_key = default_value
        self.dyn_workspaces = default_value
        self.cst_workspaces = default_value
        self.bin_workspaces = default_value
        self.soc = get_global_storage().dev_plat

    # noinspection DuplicatedCode
    def construct(self, context: UniversalTestcaseStructure,
                  compare_result: "ComparisonResult",
                  passed):
        """Construct the structure with context"""
        # Check prof_results and construct one if necessary
        input_size, output_size = context.input_bytes, context.output_bytes
        total_size = input_size + output_size
        if not isinstance(context.dyn_prof_result, RTSProfilingResult):
            context.dyn_prof_result = RTSProfilingResult(passed, None)
        if not isinstance(context.cst_prof_result, RTSProfilingResult):
            context.cst_prof_result = RTSProfilingResult(passed, None)
        if not isinstance(context.bin_prof_result, RTSProfilingResult):
            context.bin_prof_result = RTSProfilingResult(passed, None)
        self.dyn_tiling_time_us = context.dyn_compile_result.tiling_result.tiling_time
        self.bin_tiling_time_us = context.bin_compile_result.tiling_result.tiling_time
        # DYN
        self.dyn_kernel_name = context.dyn_compile_result.kernel_name
        self.dyn_block_dim = str(context.dyn_compile_result.block_dim) \
            if context.dyn_compile_result.block_dim != 0 else context.dyn_prof_result.cycle
        self.dyn_perf_us = context.dyn_prof_result.cycle
        self.dyn_compile_s = context.dyn_compile_result.compile_time
        self.dyn_precision = compare_result.dyn_precision
        self.dyn_oob_result = context.dyn_prof_result.oob
        # CST
        self.cst_kernel_name = context.cst_compile_result.kernel_name
        self.cst_block_dim = str(context.cst_compile_result.block_dim) \
            if context.cst_compile_result.block_dim != 0 else context.cst_prof_result.cycle
        self.cst_perf_us = context.cst_prof_result.cycle
        self.cst_compile_s = context.cst_compile_result.compile_time
        self.cst_precision = compare_result.cst_precision
        self.cst_oob_result = context.cst_prof_result.oob
        # BIN
        self.bin_kernel_name = context.bin_compile_result.kernel_name
        self.bin_block_dim = str(context.bin_compile_result.block_dim) \
            if context.bin_compile_result.block_dim != 0 else context.bin_prof_result.cycle
        self.bin_perf_us = context.bin_prof_result.cycle
        self.bin_compile_s = context.bin_compile_result.compile_time
        self.bin_precision = compare_result.bin_precision
        self.bin_oob_result = context.bin_prof_result.oob

        self.perf_status = passed
        self.precision_status = compare_result.passed
        self.memory_oob_status = "PASS" if all(getattr(getattr(context, f"{x}_prof_result"), "oob_status") == "PASS"
                                               for x in ("dyn", "cst", "bin")) else "FAIL"
        self.data_input_size_b = input_size
        self.data_output_size_b = output_size
        self.dyn_tiling_data = context.dyn_str_tiling_data
        self.dyn_tiling_key = f"{context.dyn_compile_result.tiling_key} ({context.str_tiling_key()})"
        self.bin_tiling_data = context.bin_str_tiling_data
        self.bin_tiling_key = f"{context.bin_compile_result.tiling_key} ({context.str_tiling_key(True)})"
        self.dyn_workspaces = context.dyn_compile_result.workspaces
        self.cst_workspaces = context.cst_compile_result.workspaces
        self.bin_workspaces = context.bin_compile_result.workspaces

    @staticmethod
    def get_titles(custom: bool = False) -> Tuple[str]:
        if get_global_storage().custom_columns and custom:
            result_titles = tuple(title for title in get_global_storage().custom_columns
                                  if title in ProfilingReturnStructure.__slots__)
        else:
            result_titles = ProfilingReturnStructure.__slots__
        if len(result_titles) < 1:
            raise RuntimeError("Profiling Result Title length must not be lower than 1")
        return result_titles

    def get(self, custom: bool = False):
        """
        Convert Structure to csv writable structure
        :return:
        """
        return tuple(getattr(self, title) for title in self.get_titles(custom))

    def kernel_execute_failed(self):
        for typ in ("dyn", "cst", "bin"):
            perf_us = str(getattr(self, f"{typ}_perf_us")).split(',')
            for p in perf_us:
                if p in ("RTS_BINARY_FAILURE", "RTS_FUNCTION_FAILURE", "LAUNCH_FAILED",
                         "TRAP", "AIC_ERROR", "VEC_ERROR", "TIMEOUT",
                         "UNKNOWN_RTS_ERROR"):
                    return True
        return False

    def pick_data(self, titles: Tuple[str]) -> tuple:
        """ Pick result data via titles """
        data = []
        for t in titles:
            if hasattr(self, t):
                data.append(getattr(self, t))
            else:
                data.append('')
        return tuple(data)


class ComparisonResult:
    __slots__ = ("dyn_precision",
                 "cst_precision",
                 "bin_precision",
                 "passed")

    def __init__(self, default_value):
        self.dyn_precision = default_value
        self.cst_precision = default_value
        self.bin_precision = default_value
        self.passed = default_value

    def set(self, a, b, c, d):
        self.dyn_precision = a
        self.cst_precision = b
        self.bin_precision = c
        self.passed = d
        return self

    def get(self) -> tuple:
        return tuple(getattr(self, name) for name in self.__slots__)
