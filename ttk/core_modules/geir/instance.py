#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import logging
import os

from ttk.core_modules.dsmi import DSMIInterface
from ttk.core_modules.infra.instance_base import InstanceBase
from ttk.utilities import cpu_count
from ttk.utilities.platform import get_npu_hw_info


class GeirInstance(InstanceBase):
    def __init__(self):
        super().__init__()

    def env_prepare(self):
        from ttk._env import _find_ascend_root

        asc_path = _find_ascend_root()
        if not asc_path:
            opp_path = os.getenv("ASCEND_OPP_PATH", "")
            if not opp_path:
                raise RuntimeError("Install path of opp or compiler is not found. Please check Ascend installation.")

    def get_device_count(self):
        if self.switches.device_count <= 0:
            self.switches.device_count = DSMIInterface().get_device_count()
        if self.switches.device_count <= 0:
            self.switches.device_count = min(max(cpu_count() - 1, 1), 4)
            logging.warning(f"No NPU devices detected, using {self.switches.device_count} workers")

    def get_device_platform(self):
        if self.switches.dev_plat == "AUTO":
            try:
                self.switches.dev_plat = DSMIInterface().get_chip_info(0).get_complete_platform()
            except Exception:
                logging.warning("Cannot detect platform, using default.")
                self.switches.dev_plat = "Ascend910B"
        hw_info = get_npu_hw_info(self.switches.dev_plat)
        self.switches.short_soc_version = hw_info.get("short_soc_version", "")
        os.environ["TTK_FULL_SOC_VERSION"] = self.switches.dev_plat
        os.environ["TTK_SHORT_SOC_VERSION"] = self.switches.short_soc_version

    def setup_profile_object(self):
        from .object import GeirProfileObject

        self.profile_object = GeirProfileObject(self.task_keeper, self.mp_context)

    def device_info(self, dev_id: int) -> str:
        try:
            chip = DSMIInterface().get_chip_info(dev_id)
            platform = chip.get_complete_platform()
        except Exception:
            platform = "N/A"
        return f"NPU:{dev_id} {platform}"
