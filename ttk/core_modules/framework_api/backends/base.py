#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from abc import ABC, abstractmethod

import numpy as np


class Backend(ABC):
    """Hardware backend abstraction for NPU / XPU / CPU.

    Device-capability contract (6 first-principle methods):
      is_available / device_count / to_device / synchronize / from_numpy / to_numpy

    Identity contract:
      device_name()  -- hardware MODEL name (via torch.<lib>.get_device_name);
                        e.g. 'Ascend910B3'. CPU has no torch.cpu.get_device_name,
                        so CpuTorchBackend overrides device_name to alias() ('cpu').
      alias()        -- the config-segment name (config-driven). _build injects
                        _segment_name = the yaml segment key, so alias() returns
                        whatever the deployer named the segment ('npu'/'mlu'/...).
                        CpuTorchBackend (never goes through _build) sets the
                        class attribute _segment_name = 'cpu'.
      is_npu()       -- convenience predicate (default False; NpuTorchBackend overrides).
      has_device()   -- whether device resources are used.
      soc_series()   -- short SoC series; default degrades to device_name() (model);
                        NpuTorchBackend overrides via get_npu_hw_info(model).

    Injection points (set by config/loader):
      torch_lib  -- torch module attribute name; getattr(torch, torch_lib) yields the
                    device module (e.g. 'npu' → torch.npu, 'mlu' → torch.mlu).
      profile    -- profiler config dict.
    """

    torch_lib: str
    profile: dict
    # Config-driven segment name; injected by _build (name = yaml segment key).
    # CpuTorchBackend (never _build) sets the class attribute 'cpu' as a fallback.
    _segment_name: str = ""

    def device_name(self, dev_id: int = 0) -> str:
        """Return hardware MODEL name via torch.<torch_lib>.get_device_name."""
        import torch

        return getattr(torch, self.torch_lib).get_device_name(dev_id)

    def device_type(self) -> str:
        """Return the config-segment name (config-driven via _segment_name).

        No longer hardcoded per subclass: _build injects _segment_name = the yaml
        segment key, so a 'mlu:' segment yields alias() == 'mlu', 'xpu:' yields
        'xpu', etc. CpuTorchBackend (never _build) sets the class attribute 'cpu'.
        """
        return self._segment_name

    @abstractmethod
    def is_available(self) -> bool:
        """Check if device is available."""

    @abstractmethod
    def device_count(self) -> int:
        """Return number of available devices."""

    def to_device(self, tensor, dev_id: int = 0, preserve_stride: bool = False):
        """Move tensor to target device.

        preserve_stride=True: preserve non-contiguous stride (torch only).
        Default: framework's standard device move.
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self, dev_id: int = 0):
        """Synchronize device (for accurate timing)."""

    @abstractmethod
    def from_numpy(self, arr: np.ndarray):
        """Convert numpy array to framework tensor (zero-copy preferred)."""

    def to_numpy(self, tensor, safe: bool = False) -> np.ndarray:
        """Convert framework tensor to numpy array.

        safe=True: extra detach/clone for inplace-extracted tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def has_device(self) -> bool:
        """Whether use device resource."""

    def use_device(self) -> bool:
        """Compatibility alias used by the framework multi-device runner."""
        return self.has_device()

    def is_npu(self) -> bool:
        """Whether this is an NPU backend. Default False; NpuTorchBackend overrides."""
        return False

    def soc_series(self) -> str:
        """Default: degrade to device_name() (model). NpuTorchBackend overrides short."""
        return self.device_name()

    # ========== Framework tensor-lifecycle methods ==========

    def clone(self, tensor):
        """Make a copy of a framework tensor (for inplace backup)."""
        import copy

        return copy.copy(tensor)

    def restore_inplace(self, target, backup):
        """Restore a tensor from backup after an inplace operation. Default: no-op."""
        pass

    def is_npu_only(self, api_name: str) -> bool:
        """Whether the API can only run on device (not CPU). Default: False."""
        return False

    def supports_graph_mode(self) -> bool:
        """Whether this backend supports graph mode compilation."""
        return True

    def supports_format_cast(self) -> bool:
        """Whether this backend supports NPU format cast."""
        return False

    def set_deterministic_level(self, level):
        """Set deterministic computation level. Default: no-op."""
        pass

    def device_scope(self, dev_id=0):
        """Context manager for device-scoped execution. Default: nullcontext."""
        from contextlib import nullcontext

        return nullcontext()

    def wrap_eager_callable(self, resolved):
        """Wrap an API callable for eager execution. Default: no-op."""
        return resolved

    def needs_numpy_fallback(self, testcase) -> bool:
        """Whether plugins should receive numpy arrays (instead of framework tensors).

        True when the framework cannot natively represent the testcase's dtype
        (e.g. torch with float8/int4), so data is passed as numpy arrays instead.
        """
        return False

    def inputs_from_numpy(self, testcase, raw_inputs):
        """Convert flat numpy arrays to framework tensors.

        When needs_numpy_fallback() is True (framework cannot natively
        represent the dtype, e.g. torch int4/float8), raw numpy arrays
        are returned as-is so plugins receive numpy instead of crashing
        in the framework tensor conversion.
        """
        if self.needs_numpy_fallback(testcase):
            return list(raw_inputs)
        return [self.from_numpy(arr) if arr is not None else None for arr in raw_inputs]

    def result_to_numpy(self, result, copy=False):
        """Convert an API result (Tensor/tuple/scalar) to list of numpy arrays."""
        if result is None:
            return [None]
        if isinstance(result, (tuple, list)):
            nps = []
            for r in result:
                if r is None:
                    nps.append(None)
                elif hasattr(r, "numpy") and callable(getattr(r, "numpy")):
                    arr = self.to_numpy(r)
                    nps.append(arr.copy() if copy else arr)
                else:
                    nps.append(np.array(r))
            return nps
        if hasattr(result, "numpy") and callable(getattr(result, "numpy")):
            arr = self.to_numpy(result)
            return [arr.copy() if copy else arr]
        return [np.array(result)]
