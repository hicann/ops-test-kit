#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch

from .base import Backend
from ....utilities import numpy_to_torch_tensor, torch_to_numpy_tensor


class CpuBackend(Backend):
    """CPU backend (for golden baseline or no-device testing)."""

    def device_name(self):
        return "cpu"

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def to_device(self, tensor, dev_id=0):
        return self.from_numpy(tensor)

    def synchronize(self, dev_id=0):
        pass

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def to_numpy(self, tensor):
        return torch_to_numpy_tensor(tensor.detach().contiguous())
    
    def use_device(self):
        return False