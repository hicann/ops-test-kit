#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

"""TorchBackend: torch-generic intermediate layer.

Holds the shared torch implementation (numpy<->torch conversion, device move via
``getattr(torch, self.torch_lib)``). Hardware-specific subclasses override
``is_npu`` only; alias() comes from Backend (config-driven _segment_name).

``device_name`` is inherited from Backend (returns the hardware model via
``torch.<torch_lib>.get_device_name``). CpuTorchBackend keeps its device_name
override (no ``torch.cpu.get_device_name`` exists).
"""

import torch

from .base import Backend
from ....utilities import numpy_to_torch_tensor, torch_to_numpy_tensor


class TorchBackend(Backend):
    """Shared torch implementation; subclass per hardware."""

    def is_available(self) -> bool:
        return bool(getattr(torch, self.torch_lib).is_available())

    def device_count(self) -> int:
        return getattr(torch, self.torch_lib).device_count()

    def to_device(self, tensor, dev_id: int = 0):
        t = self.from_numpy(tensor)
        return getattr(t, self.torch_lib)(dev_id)

    def synchronize(self, dev_id: int = 0):
        getattr(torch, self.torch_lib).synchronize(dev_id)

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def to_numpy(self, tensor):
        # .contiguous() guards torch_to_numpy_tensor against non-contiguous tensors
        # (matches existing cpu_backend behaviour).
        return torch_to_numpy_tensor(tensor.detach().cpu().contiguous())

    def use_device(self) -> bool:
        return True
