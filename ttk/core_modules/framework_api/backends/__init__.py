#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from typing import Optional

from .base import Backend
from .npu_backend import NpuBackend
from .gpu_backend import GpuBackend
from .cpu_backend import CpuBackend

_BACKEND_MAP = {
    "npu": NpuBackend,
    "gpu": GpuBackend,
    "cpu": CpuBackend,
}


def get_backend(backend_name: Optional[str] = None) -> Backend:
    """Get backend by name or auto-detect (npu > gpu > cpu)."""
    if backend_name:
        cls = _BACKEND_MAP.get(backend_name)
        if cls is None:
            raise ValueError(f"Unknown backend: {backend_name}. Must be one of {list(_BACKEND_MAP.keys())}")
        return cls()

    for cls in (NpuBackend, GpuBackend, CpuBackend):
        b = cls()
        if b.is_available():
            return b
    return CpuBackend()