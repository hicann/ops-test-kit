#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TfBackend: TF-generic intermediate layer.

Holds the shared TF implementation (numpy<->tf conversion, device move via
``tf.device``). Hardware-specific subclasses override ``is_npu`` only.
"""

from __future__ import annotations

import numpy as np

from .base import Backend


class TfBackend(Backend):
    """Shared TF implementation; subclass per hardware."""

    tf_device_type: str = ""

    def is_available(self) -> bool:
        import tensorflow as tf

        return bool(tf.config.list_physical_devices(self.tf_device_type))

    def device_count(self) -> int:
        import tensorflow as tf

        return len(tf.config.list_physical_devices(self.tf_device_type))

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        import tensorflow as tf

        t = self.from_numpy(tensor)
        if self.tf_device_type and self.tf_device_type != "cpu":
            with tf.device(f"/{self.tf_device_type}:{dev_id}"):
                return tf.identity(t)
        return t

    def synchronize(self, dev_id: int = 0):
        pass

    def from_numpy(self, arr):
        import tensorflow as tf

        from ttk.utilities.dtypes import normalize_to_tf_dtype

        if arr is None:
            return None
        # 0-D 本就连续；ascontiguousarray 会把标量提升为 (1,)，破坏 TF 0-D 参数校验
        arr = np.ascontiguousarray(arr) if arr.ndim else arr
        arr = normalize_to_tf_dtype(arr)
        return tf.convert_to_tensor(arr)

    def to_numpy(self, tensor, safe: bool = False):
        if tensor is None:
            return None
        return tensor.numpy()

    def has_device(self) -> bool:
        return self.tf_device_type != "cpu"

    def device_name(self, dev_id: int = 0) -> str:
        return self.device_type()

    def is_npu(self) -> bool:
        return False

    def soc_series(self) -> str:
        return self.device_name()

    def clone(self, tensor):
        import tensorflow as tf

        return tf.identity(tensor)

    def restore_inplace(self, target, backup):
        pass

    def is_npu_only(self, api_name: str) -> bool:
        return api_name.startswith(("tf.npu_",))

    def supports_graph_mode(self) -> bool:
        return True

    def supports_format_cast(self) -> bool:
        return False

    def needs_numpy_fallback(self, testcase) -> bool:
        return not testcase.is_tf_dtype_support()

    def inputs_from_numpy(self, testcase, raw_inputs):
        from ..input_generation import np_to_tf_inputs

        return np_to_tf_inputs(testcase, raw_inputs)
