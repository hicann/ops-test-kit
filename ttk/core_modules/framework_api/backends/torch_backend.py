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

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        if preserve_stride:
            return self._to_device_preserving_stride(tensor, dev_id)
        t = self.from_numpy(tensor)
        return getattr(t, self.torch_lib)(dev_id)

    def _to_device_preserving_stride(self, tensor, dev_id):
        """Move a (possibly non-contiguous) tensor to device preserving stride."""
        if tensor is None:
            return None
        if self.torch_lib == "cpu":
            return tensor
        if tensor.is_contiguous():
            return getattr(tensor, self.torch_lib)(dev_id)
        return self._clone_full_storage(tensor, f"{self.torch_lib}:{dev_id}")

    def synchronize(self, dev_id: int = 0):
        getattr(torch, self.torch_lib).synchronize(dev_id)

    def from_numpy(self, arr):
        return numpy_to_torch_tensor(arr)

    def to_numpy(self, tensor, safe: bool = False):
        tensor = tensor.detach()
        if safe:
            tensor = tensor.clone()
        return torch_to_numpy_tensor(tensor.cpu().contiguous())

    def has_device(self) -> bool:
        return True

    def _clone_full_storage(self, tensor, dst_device=None):
        """Clone a non-contiguous tensor preserving stride AND stride-gap data.

        ``empty_strided`` + ``copy_`` only allocates storage for the visible
        elements of the view, leaving stride-gap memory uninitialized.  An
        operator that reads into those gaps sees garbage that differs between
        clones and the original tensor — which is why the last profiling round
        (using the original) can diverge from earlier rounds (using clones).

        To fix this we clone the *entire* underlying storage (gaps included)
        and rebuild the non-contiguous view via ``as_strided``.
        """
        src_storage = tensor.untyped_storage()
        elem_size = tensor.element_size()
        storage_numel = src_storage.size() // elem_size
        src_flat = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(src_storage, 0, (storage_numel,), (1,))
        new_flat = src_flat.to(dst_device) if dst_device else src_flat.clone()
        return torch.as_strided(new_flat, tensor.shape, tensor.stride(), tensor.storage_offset())

    def clone(self, tensor):
        """Clone a tensor preserving its non-contiguous stride and gap data.

        Unlike ``torch.Tensor.clone()``, which materializes a non-contiguous
        view into a contiguous tensor, this clones the full underlying storage
        (so stride-gap data is identical to the original) and rebuilds the
        non-contiguous view via ``as_strided``.
        """
        if tensor is None:
            return None
        if tensor.is_contiguous():
            return tensor.clone()
        return self._clone_full_storage(tensor)

    def restore_inplace(self, target, backup):
        target[:] = backup

    def is_npu_only(self, api_name: str) -> bool:
        return api_name.startswith(("torch_npu.", "torch.npu"))

    def supports_graph_mode(self) -> bool:
        return True

    def supports_format_cast(self) -> bool:
        return self.is_npu()

    def needs_numpy_fallback(self, testcase) -> bool:
        return not testcase.is_torch_dtype_support()

    def inputs_from_numpy(self, testcase, raw_inputs):
        if self.needs_numpy_fallback(testcase):
            return list(raw_inputs)
        from ..input_generation import np_to_torch_inputs

        return np_to_torch_inputs(testcase, raw_inputs)
