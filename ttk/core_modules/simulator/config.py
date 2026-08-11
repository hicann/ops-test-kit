#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
"""Shared constants for the NPUSim simulator backend."""

# NPUSim SoC version (``npusim record -s``) -> CANN platform ini name (the
# ``full_soc`` TTK uses for get_npu_hw_info / compilation). The ini lives under
# ``<opp>/../aarch64-linux/data/platform_config/<name>.ini`` and mirrors the
# same chip (NpuArch etc.) as the camodel NPUSim simulates.
SIM_PLATFORM_BY_SOC = {
    "Ascend950": "Ascend950PR_9589",
    "Ascend950DT": "Ascend950PR_9599",
}


def resolve_platform_soc(soc_version: str) -> str:
    """Map an NPUSim SoC version to the TTK platform name; passthrough if unknown."""
    return SIM_PLATFORM_BY_SOC.get(soc_version, soc_version)
