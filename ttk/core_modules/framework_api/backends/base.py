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
    """Hardware backend abstraction for NPU / GPU / CPU."""

    @abstractmethod
    def device_name(self) -> str:
        """Return device name: 'npu' / 'cuda' / 'cpu'"""

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

    def soc_version(self) -> str:
        return self.device_name()

    def soc_series(self) -> str:
        return self.soc_version()
