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
Main Sequence for npu profiling
"""


__all__ = ["NpuInstance"]


# Standard Packages
import os
import logging

# Third-Party Packages
from ..dsmi import DSMIInterface
from ..infra import InstanceBase
from ..operator import Opc
from ...utilities import cpu_count
from ...utilities.platform import get_npu_hw_info


class NpuInstance(InstanceBase):
    """
    Npu Profiling Instance
    """

    def __init__(self):
        super().__init__()

    def env_prepare(self):
        # check environment.
        opp_path = os.getenv('ASCEND_OPP_PATH', '')
        if not opp_path:
            raise RuntimeError(f'Install path of opp or compiler is not found. '
                               f'Please check Ascend installation.')

    def get_device_count(self):
        """Resolve worker count and persist it on ``switches.device_count``.

        Manual-data prepare has no device execution, but the normal scheduler
        still needs one logical worker. That mode deliberately sets the stored
        count to one and avoids querying DSMI hardware.
        """
        if getattr(self.switches, "manual_data_mode", None) == "prepare":
            self.switches.device_count = 1
        elif self.switches.mode.is_model():
            self.switches.device_count = 1
        elif self.switches.device_count <= 0:
            if self.switches.compile_only:
                count = min(max(cpu_count() - 1, 1), 8)
            else:
                count = DSMIInterface().get_device_count()
            self.switches.device_count = count
        else:
            pass

    def get_device_platform(self):
        if self.switches.dev_plat == "AUTO":
            if self.switches.mode.is_model():
                raise RuntimeError(f"Please specify your platform type "
                                   f"with --plat in {self.switches.mode.name} mode")
            else:
                try:
                    self.switches.dev_plat = DSMIInterface().get_chip_info(0).get_complete_platform()
                except:
                    if (self.switches.compile_only or self.switches.validate_only or
                            getattr(self.switches, "manual_data_mode", None) == "prepare"):
                        raise RuntimeError(f"Try to get Ascend platform failed. "
                                           f"Please specify it with option like: --plat=Ascend910A")
                    else:
                        raise
        hw_info = get_npu_hw_info(self.switches.dev_plat)
        self.switches.short_soc_version = hw_info.get("short_soc_version")
        os.environ["TTK_FULL_SOC_VERSION"] = self.switches.dev_plat
        os.environ["TTK_SHORT_SOC_VERSION"] = self.switches.short_soc_version

    def setup_profile_object(self):
        params = tuple([self.task_keeper, self.mp_context])
        if 'api_name' in self.case_original_headers:
            from .op_api import ApiProfileObject
            self.profile_object = ApiProfileObject(*params)
        else:
            from .op import OpProfileObject
            self.profile_object = OpProfileObject(*params)
        if self.switches.mode.is_model():
            os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
        if (not self.switches.compile_only and
                getattr(self.switches, "manual_data_mode", None) != "prepare"):
            self._compile_help_kernels()

    def device_info(self, dev_id: int) -> str:
        if getattr(self.switches, "manual_data_mode", None) == "prepare":
            return f"manual-data:{dev_id} {self.switches.dev_plat}"
        if not self.switches.mode.is_online_board() or self.switches.compile_only:
            phyid = str(dev_id)
            platform = self.switches.dev_plat
            chip_ver = "MODEL"
            health = "ERR"
            temperature = "???"

            aicore_util = "???"
            mem_util = "???"
            memband_util = "???"
            hbm_util = "???"
            ddr_util = "???"
            hbmband_util = "???"
            veccore_util = "???"

            mem_freq = "????"
            hbm_freq = "????"
            aicore_freq = "????"
            aicore_maxfreq = "????"
            veccore_freq = "????"
        else:
            dsmi = DSMIInterface()
            phyid = str(dsmi.get_physical_id_from_logical_id(dev_id) or dev_id)
            try:
                dsmi_platform = dsmi.get_chip_info(dev_id)
                platform = dsmi_platform.get_complete_platform()
                chip_ver = dsmi_platform.get_ver()
            except:
                platform = "ERR"
                chip_ver = "ERR"

            try:
                health = DSMIInterface().get_device_health_state(dev_id).name
            except:
                health = "ERR"

            temperature = str(DSMIInterface().get_device_temperature(dev_id) or "???")
            # Utils
            mem_util = str(DSMIInterface().get_device_util(dev_id, 1, platform)).replace("None", "ERR")
            aicore_util = str(dsmi.get_device_util(dev_id, 2, platform)).replace("None", "ERR")
            memband_util = str(dsmi.get_device_util(dev_id, 5, platform)).replace("None", "ERR")
            hbm_util = str(dsmi.get_device_util(dev_id, 6, platform)).replace("None", "ERR")
            ddr_util = str(dsmi.get_device_util(dev_id, 8, platform)).replace("None", "ERR")
            hbmband_util = str(dsmi.get_device_util(dev_id, 10, platform)).replace("None", "ERR")
            veccore_util = str(dsmi.get_device_util(dev_id, 12, platform)).replace("None", "ERR")

            mem_freq = str(dsmi.get_device_frequency(dev_id, 1, platform) or "????")
            hbm_freq = str(dsmi.get_device_frequency(dev_id, 6, platform) or "????")
            aicore_freq = str(dsmi.get_device_frequency(dev_id, 7, platform) or "????")
            aicore_maxfreq = str(dsmi.get_device_frequency(dev_id, 9, platform) or "????")
            veccore_freq = str(dsmi.get_device_frequency(dev_id, 12, platform) or "????")

        result = []

        def _add_info(info: str):
            if any(c.isdigit() for c in info):
                result.append(info)

        _add_info(f"{phyid.ljust(3)} {platform.ljust(13)} {chip_ver.ljust(2)} "
                  f"{temperature.ljust(2)}C {health.ljust(8)}")
        _add_info(f"AIC   {aicore_util.ljust(3)}% {aicore_freq.ljust(5)}Mhz / {aicore_maxfreq.ljust(5)}Mhz")
        _add_info(f"VEC   {veccore_util.ljust(3)}% {veccore_freq.ljust(5)}Mhz")
        _add_info(f"MEM   {mem_util.ljust(3)}% {mem_freq.ljust(5)}Mhz Bandwidth: {memband_util.ljust(3)}%")
        _add_info(f"DDR   {ddr_util.ljust(3)}%")
        _add_info(f"HBM   {hbm_util.ljust(3)}% {hbm_freq.ljust(5)}Mhz Bandwidth: {hbmband_util.ljust(3)}%")
        return '\n'.join(result)

    @staticmethod
    def _compile_help_kernels():
        Opc().compile_ub_clear()
        Opc().compile_l1_clear()
        Opc().compile_warmup_kernel()
