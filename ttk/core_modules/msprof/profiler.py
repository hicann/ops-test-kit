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
Interface for msprof
"""

__all__ = ["MsProfiler"]


# Standard Packages
import ctypes
import logging
import os
import shutil
import subprocess
import time
from typing import List, Optional

# Third-Party Packages
from .desc import (
    ACL_ERROR_DESC_DICT,
    AclProfType,
    AiCoreProfMetrics,
    MsprofCompactInfo,
    MsProfOpDfx,
    MsprofTaskType,
    TtkMsProfType,
)


class MsProfiler:
    """
    interfaces for msprof
    """

    def __init__(
        self,
        device_id: int,
        result_path: str,
        ttk_prof_type: TtkMsProfType,
        extra_acl_prof_type: Optional[List[AclProfType]] = None,
        start_step: int = 1,
        is_model: bool = False,
    ):
        self._dvc_id: int = device_id
        self._result_path: str = result_path
        self._start_step = start_step
        self._current_step = 0
        self._is_model: bool = is_model
        self._prof_cfg: Optional[ctypes.c_void_p] = None
        self._data_type_cfg = self._build_data_type_config(ttk_prof_type, extra_acl_prof_type)
        self._prof_inited: bool = False
        self._prof_started: bool = False
        if self._is_model or (ttk_prof_type == TtkMsProfType.NONE and not extra_acl_prof_type):
            self._msprof_dll = None
        else:
            if os.path.exists(result_path):
                shutil.rmtree(result_path)
            self._msprof_dll = ctypes.CDLL("libmsprofiler.so")
        self._prof_api_dll = None  # initialize when needed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._msprof_dll is not None:
            self._stop()
            self._analysis_profile_data()

    def step(self):
        if self._msprof_dll is None:
            return
        if self._current_step == self._start_step:
            self._start()
        self._current_step += 1

    def report_op_dfx(self, op_dfx: MsProfOpDfx):
        if self._is_model:
            return
        if self._msprof_dll is None:
            return
        if self._prof_api_dll is None:
            self._prof_api_dll = ctypes.CDLL("libprofapi.so")
        api = self._prof_api_dll.MsprofReportCompactInfo
        if api is None:
            logging.warning("MsProf API [MsprofReportCompactInfo] is not defined.")
            return

        timestamp = self._get_sys_cycle()
        kernel_name = self._get_hash_id(op_dfx.kernel_name)
        task_type = self._get_op_task_type(op_dfx.core_type, op_dfx.task_ration)
        compact_info = MsprofCompactInfo(timestamp, kernel_name, task_type, op_dfx.block_dim)
        start_time = time.time()

        api.restype = ctypes.c_int32
        api.argtypes = [ctypes.c_uint8, ctypes.c_void_p, ctypes.c_uint32]
        rt_error = api(1, ctypes.c_void_p(ctypes.addressof(compact_info)), ctypes.sizeof(compact_info))
        self.parse_error(rt_error, "MsprofReportCompactInfo", f"costs {round(time.time() - start_time, 3)} seconds")

    @staticmethod
    def parse_error(acl_ret: ctypes.c_uint64, api_name: str, extra_info: str):
        # Convert aclError to int if received ctypes object
        if isinstance(acl_ret, ctypes.c_uint64):
            acl_ret = acl_ret.value
        elif isinstance(acl_ret, int):
            pass
        else:
            raise TypeError(f"Invalid acl_ret type {type(acl_ret)}")
        # Success
        if acl_ret == 0x0:
            logging.debug(f"MsProf API Call {api_name}() Success, {extra_info}")
            return
        # Fail
        desc = ACL_ERROR_DESC_DICT.get(acl_ret, str(acl_ret))
        raise RuntimeError(f"MsProf API Call {api_name}() Failed: {desc}, {extra_info}")

    def _start(self):
        self._acl_prof_init()
        self._acl_prof_create_config(self._data_type_cfg)
        self._acl_prof_start()

    def _stop(self):
        self._acl_prof_stop()
        self._acl_prof_destroy_config()
        self._acl_prof_finalize()

    def _acl_prof_init(self):
        """
        aclError aclprofInit(const char *profilerResultPath, size_t length);
        """
        c_path = self._result_path.encode("UTF-8")
        c_size = ctypes.c_size_t(len(self._result_path))
        self._api_call("aclprofInit", None, c_path, c_size)
        self._prof_inited = True

    def _acl_prof_create_config(self, data_type_cfg: int):
        """
        aclprofConfig *aclprofCreateConfig(
            uint32_t *deviceIdList, uint32_t deviceNums,
            aclprofAicoreMetrics aicoreMetrics,
            aclprofAicoreEvents *aicoreEvents,  # always nullptr
            uint64_t dataTypeConfig);
        """
        c_device_id_lst = (ctypes.c_uint32 * 1)(self._dvc_id)
        c_device_nums = ctypes.c_uint32(1)
        c_aicore_metrics = ctypes.c_int(AiCoreProfMetrics.PIPE_UTILIZATION.value)
        c_data_type_config = ctypes.c_uint64(data_type_cfg)
        self._prof_cfg = self._api_call_with_ptr_return(
            "aclprofCreateConfig", None, c_device_id_lst, c_device_nums, c_aicore_metrics, None, c_data_type_config
        )

    def _acl_prof_start(self):
        """
        aclError aclprofStart(const aclprofConfig *profilerConfig);
        """
        self._api_call("aclprofStart", None, self._prof_cfg)
        self._prof_started = True

    def _acl_prof_stop(self):
        """
        aclError aclprofStop(const aclprofConfig *profilerConfig);
        """
        if self._prof_started:
            self._api_call("aclprofStop", None, self._prof_cfg)
            self._prof_started = False

    def _acl_prof_destroy_config(self):
        """
        aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig)
        """
        if self._prof_cfg is not None:
            self._api_call("aclprofDestroyConfig", None, self._prof_cfg)
            self._prof_cfg = None

    def _acl_prof_finalize(self):
        """
        aclError aclprofFinalize();
        """
        if self._prof_inited:
            self._api_call("aclprofFinalize", None)
            self._prof_inited = False

    def _get_hash_id(self, hash_info: str):
        api = self._prof_api_dll.MsprofGetHashId
        if api is None:
            raise RuntimeError("MsProf API [MsprofGetHashId] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        api.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        hash_info = hash_info.encode("utf-8")
        rt_val = api(hash_info, len(hash_info))
        logging.debug(
            f"MsProf API Call MsprofGetHashId() Success, hash_info: {hash_info}. "
            f"costs {round(time.time() - start_time, 3)} seconds"
        )
        return rt_val

    def _get_sys_cycle(self):
        api = self._prof_api_dll.MsprofSysCycleTime
        if api is None:
            raise RuntimeError("MsProf API [MsprofSysCycleTime] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        api.argtypes = []
        rt_val = api()
        logging.debug(
            f"MsProf API Call MsprofSysCycleTime() Success, costs {round(time.time() - start_time, 3)} seconds"
        )
        return rt_val

    def _analysis_profile_data(self):
        if not os.path.exists(self._result_path):
            return
        msprof = f"{os.getenv('ASCEND_OPP_PATH', '')}/../tools/profiler/profiler_tool/analysis/msprof/msprof.py"
        if not os.path.exists(msprof):
            logging.warning(f"File [{msprof}] is not found. Please install CANN-toolkit RUN package.")
            return
        for t in ("summary", "timeline"):
            child = subprocess.run(
                ["python3", msprof, "export", t, "-dir", self._result_path], shell=False, capture_output=True
            )
            if child.returncode != 0:
                logging.error(f"call msprof.py to export {t} failed: {child.stdout}, {child.stderr}")

    def _api_call(self, api_name: str, extra_log: Optional[str], *api_args):
        if extra_log is None:
            extra_log = ""
        api = getattr(self._msprof_dll, api_name)
        if not api:
            raise RuntimeError(f"MsProf API [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_uint64
        rt_error = api(*api_args)
        self.parse_error(rt_error, api_name, f"{extra_log} costs {round(time.time() - start_time, 3)} seconds")

    def _api_call_with_ptr_return(self, api_name: str, extra_log: Optional[str], *api_args) -> ctypes.c_void_p:
        if extra_log is None:
            extra_log = ""
        api = getattr(self._msprof_dll, api_name)
        if not api:
            raise RuntimeError(f"MsProf API [{api_name}] is not defined.")
        start_time = time.time()
        api.restype = ctypes.c_void_p
        rt_ptr = api(*api_args)
        if not rt_ptr:
            raise RuntimeError(f"MsProf API {api_name} return nullptr. {extra_log}")
        logging.debug(
            f"MsProf API Call {api_name}() Success, {extra_log} costs {round(time.time() - start_time, 3)} seconds"
        )
        return ctypes.c_void_p(rt_ptr)

    @staticmethod
    def _build_data_type_config(
        ttk_prof_type: TtkMsProfType, extra_acl_prof_type: Optional[List[AclProfType]] = None
    ) -> int:
        if not extra_acl_prof_type:
            return ttk_prof_type.value
        else:
            data_type_cfg = ttk_prof_type.value
            for acl_pt in extra_acl_prof_type:
                data_type_cfg |= acl_pt.value
            return data_type_cfg

    @staticmethod
    def _get_op_task_type(core_type, task_ration):
        if core_type in ("AiCore", "CubeCore"):
            return MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
        elif core_type in ("VectorCore",):
            return MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
        else:
            if all(task_ration):
                return MsprofTaskType.MSPROF_TASK_TYPE_MIX_AIC.value
            elif task_ration[1] == 0:
                return MsprofTaskType.MSPROF_TASK_TYPE_AI_CORE.value
            else:
                return MsprofTaskType.MSPROF_TASK_TYPE_AIV.value
