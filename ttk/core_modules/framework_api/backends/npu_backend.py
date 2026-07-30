#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from __future__ import annotations

import numpy as np

from .torch_backend import TorchBackend
from ....utilities import (
    is_torch_native_dtype, get_npu_hw_info,
)


class NpuTorchBackend(TorchBackend):
    """NPU backend.

    Inherits the torch-generic device API (is_available/device_count) from
    TorchBackend — those go through ``torch.npu.*``.

    device_name is inherited from Backend (``torch.npu.get_device_name``
    returns the SoC model, e.g. 'Ascend910B3'). soc_series derives the short
    series from that model via get_npu_hw_info (no more asys dependency).

    alias() is inherited from Backend (config-driven _segment_name, injected by
    _build = the yaml segment key).
    """

    def is_npu(self) -> bool:
        return True

    def to_device(self, tensor, dev_id=0):
        import torch_npu  # NPU-only: keep import in method body (lazy)
        str_dtype = tensor.dtype.name
        if is_torch_native_dtype(tensor.dtype.name):
            torch_tensor = self.from_numpy(tensor)
            return torch_tensor.npu(dev_id)
        else:
            if str_dtype == 'int4':
                raise RuntimeError(f"Dtype [{str_dtype}] is not supported yet.")
            elif str_dtype == 'float8_e8m0':
                return self.from_numpy(tensor).npu(dev_id)
            else:
                np_fp32 = tensor.astype(np.float32)
                npu_torch_tensor = torch_npu.npu_dtype_cast(
                    self.from_numpy(np_fp32).npu(dev_id),
                    dtype=getattr(torch_npu, str_dtype)
                )
                return npu_torch_tensor

    # to_numpy/from_numpy inherited from TorchBackend (was a dead byte-identical
    # override that also dropped the .contiguous() guard the parent applies).

    def soc_series(self):
        try:
            hw_info = get_npu_hw_info(self.device_name())
            return hw_info['short_soc_version']
        except (FileNotFoundError, RuntimeError, KeyError):
            return self.device_name()
