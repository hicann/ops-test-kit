#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import numpy as np
import torch

from .base import Backend
from ....utilities import (
    numpy_to_torch_tensor, torch_to_numpy_tensor, is_torch_native_dtype,
    get_npu_available_device_ids,
    get_ascend_full_soc_version,
    get_npu_hw_info
)


class NpuBackend(Backend):
    """NPU backend"""

    def device_name(self):
        return "npu"

    def is_available(self):
        return self.device_count() > 0

    def device_count(self):
        ids = get_npu_available_device_ids()
        return len(ids)

    def to_device(self, tensor, dev_id=0):
        import torch_npu
        str_dtype = tensor.dtype.name
        if is_torch_native_dtype(tensor.dtype.name):
            torch_tensor = self.from_numpy(tensor)
            return torch_tensor.npu(dev_id)
        else:
            if str_dtype == 'int4':
                raise RuntimeError(f"Dtype [{str_dtype}] is not supported yet.")
            else:
                np_fp32 = tensor.astype(np.float32)
                npu_torch_tensor = torch_npu.npu_dtype_cast(
                    self.from_numpy(np_fp32).npu(dev_id),
                    dtype=getattr(torch_npu, str_dtype)
                )
                return npu_torch_tensor

    def synchronize(self, dev_id=0):
        torch.npu.synchronize(dev_id)

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def to_numpy(self, tensor):
        return torch_to_numpy_tensor(tensor.detach().cpu())

    def soc_version(self):
        return get_ascend_full_soc_version()

    def soc_series(self):
        hw_info = get_npu_hw_info(self.soc_version())
        return hw_info['short_soc_version']

    def use_device(self):
        return True