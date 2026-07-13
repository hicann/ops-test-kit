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
import torch


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
      use_device()   -- whether device resources are used.
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
        return getattr(torch, self.torch_lib).get_device_name(dev_id)

    def alias(self) -> str:
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

    @abstractmethod
    def to_device(self, tensor, dev_id: int = 0):
        """Move tensor to target device."""

    @abstractmethod
    def synchronize(self, dev_id: int = 0):
        """Synchronize device (for accurate timing)."""

    @abstractmethod
    def from_numpy(self, arr: np.ndarray):
        """Convert numpy array to framework tensor (zero-copy preferred)."""

    @abstractmethod
    def to_numpy(self, tensor) -> np.ndarray:
        """Convert framework tensor to numpy array."""

    @abstractmethod
    def use_device(self) -> bool:
        """Whether use device resource."""

    def is_npu(self) -> bool:
        """Whether this is an NPU backend. Default False; NpuTorchBackend overrides."""
        return False

    def soc_series(self) -> str:
        """Default: degrade to device_name() (model). NpuTorchBackend overrides short."""
        return self.device_name()
