#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public result contract for an optional stage before the main NPU API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PreNpuResult:
    """Tell TTK whether the main NPU API should run after the custom stage."""

    stop: bool = False
    reason: str = ""

    def __post_init__(self):
        if not isinstance(self.stop, bool):
            raise TypeError("PreNpuResult.stop must be a boolean")
        if not isinstance(self.reason, str):
            raise TypeError("PreNpuResult.reason must be a string")

