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

"""CPU TF backend (for golden baseline or no-device testing).

When npu_device.open() has been called (NPU TF backend active), NPU becomes
the TF default device via _ContextWithDefaultDevice.  CPU golden generation
must explicitly place tensors and ops on CPU to avoid silently running on NPU.
"""

from .tf_backend import TfBackend


class CpuTfBackend(TfBackend):
    """CPU TF backend (for golden baseline or no-device testing)."""

    tf_device_type = "cpu"
    profile = {"profiler": {"activities": ["CPU"]}}
    _segment_name = "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def from_numpy(self, arr):
        import tensorflow as tf
        from ttk.utilities.dtypes import normalize_to_tf_dtype
        import numpy as np

        if arr is None:
            return None
        arr = np.ascontiguousarray(arr)
        arr = normalize_to_tf_dtype(arr)
        with tf.device("/CPU:0"):
            return tf.convert_to_tensor(arr)

    def to_device(self, tensor, dev_id=0, preserve_stride=False):
        return self.from_numpy(tensor)

    def device_scope(self, dev_id=0):
        import tensorflow as tf

        return tf.device("/CPU:0")

    def synchronize(self, dev_id=0):
        pass

    def has_device(self) -> bool:
        return False
