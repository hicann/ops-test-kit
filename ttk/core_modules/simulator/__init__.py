#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
"""NPUSim simulator backend for the kernel/aclnn test modes.

See ``docs/TTK_NPUSim仿真集成设计.md`` for the design.
"""
from .sim_profiling import run_aclnn_sim, run_kernel_sim

__all__ = ["run_kernel_sim", "run_aclnn_sim"]
