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
FrameworkApiInstance — InstanceBase implementation for framework_api tests.
"""

import logging
import os

from ttk.core_modules.infra.instance_base import InstanceBase
from ttk.utilities.container_utils import get_global_storage

from .backends import get_backend
from .object import FrameworkApiProfileObject


class FrameworkApiInstance(InstanceBase):
    """Entry instance for framework-level API testing."""

    def __init__(self):
        super().__init__()
        switches = get_global_storage()
        if switches.backend == "npusim":
            self._inject_camodel_env(switches)
        framework = getattr(switches, "framework", "torch")
        self.backend = get_backend(switches.force_cpu, framework=framework)
        if not self.backend.has_device():
            switches.proc_no_reuse = True
        logging.info(f"Framework API mode: backend={self.backend.device_type()}, framework={framework}")

    @staticmethod
    def _inject_camodel_env(switches):
        """Prepare the camodel simulation environment for the E2E profiling workers.

        E2E profiling workers are forked (forkserver) from this process and
        inherit its ``LD_LIBRARY_PATH``, so prepending the camodel lib dir here
        makes every worker load the camodel runtime instead of the real device
        runtime. Must run before ``get_backend()`` probes
        ``torch.npu.is_available()`` so the NPU backend is selected.
        """
        if switches.force_cpu:
            raise ValueError("--cpu cannot be combined with --backend npusim")
        from ttk.core_modules.simulator.config import resolve_camodel_lib_dir

        camodel = resolve_camodel_lib_dir(switches.sim_soc_version)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        camodel_str = str(camodel)
        # Exact entry check on the ":"-split list (mirrors _env._restore_ld_paths):
        # substring matching on the joined string would skip injection when an
        # unrelated entry merely contains the camodel path (e.g. ".../camodel_bak").
        if camodel_str not in existing.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{camodel_str}:{existing}" if existing else camodel_str
        log_dir = os.path.join(switches.sim_output_dir, "camodel_log")
        os.environ.setdefault("CAMODEL_LOG_PATH", log_dir)
        os.environ.setdefault("STARS_LOG_PATH", log_dir)
        logging.info("E2E npusim: camodel runtime injected from %s", camodel)

    def env_prepare(self):
        pass

    def get_device_count(self):
        switches = get_global_storage()
        if switches.device_count == -1:
            switches.device_count = self.backend.device_count()
        logging.info(f"Device count: {switches.device_count}")

    def get_device_platform(self):
        switches = get_global_storage()
        if switches.dev_plat == "AUTO":
            switches.dev_plat = self.backend.device_name()
        switches.short_soc_version = self.backend.soc_series()
        logging.info(f"Device platform: {switches.dev_plat}")

    def setup_profile_object(self):
        self.profile_object = FrameworkApiProfileObject(self.task_keeper, self.mp_context, self.backend)

    def device_info(self, dev_id: int) -> str:
        return f"{self.backend.device_type()}:{dev_id}"
