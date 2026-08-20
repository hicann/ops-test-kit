# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Backend ABC + TorchBackend + 三个硬件后端测试。

Task 7: device_name() 返回硬件 MODEL（torch.<lib>.get_device_name）；
alias() 携带段名；soc_version 已移除（合并入 device_name）。
"""
from __future__ import annotations

import pytest

from ttk.core_modules.framework_api.backends.base import Backend
from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend
from ttk.core_modules.framework_api.backends.npu_torch_backend import NpuTorchBackend
from ttk.core_modules.framework_api.backends.xpu_torch_backend import XpuTorchBackend


def test_backend_abc_has_new_methods():
    """Backend ABC 必须包含 device_name/device_type/has_device/is_npu 方法。"""
    assert all(hasattr(Backend, m) for m in ["device_name", "device_type", "has_device", "is_npu"])


def test_cpu_no_device(monkeypatch):
    """CpuTorchBackend: device_type()='cpu'，has_device=False。"""
    cb = CpuTorchBackend()
    cb.torch_lib = "cpu"
    cb.profile = {}
    # CPU never goes through _build; _segment_name = 'cpu' is a class attribute.
    assert cb.device_type() == "cpu"
    assert cb.has_device() is False


def test_npu_backend_is_npu_device_type():
    """NpuTorchBackend: is_npu()=True, device_type()='npu'（段名驱动）。"""
    nb = NpuTorchBackend()
    nb.torch_lib = "npu"
    nb.profile = {}
    nb._segment_name = "npu"
    assert nb.is_npu() is True
    assert nb.device_type() == "npu"


# --- device_type = config-driven segment name (not hardcoded per subclass) ---


@pytest.mark.parametrize(
    "segment, torch_lib, profiler_config, expected_cls, check_torch_lib",
    [
        pytest.param(
            "gpu", "cuda", {"activities": ["CPU", "CUDA"]},
            XpuTorchBackend, "cuda",
            id="segment_name_not_hardcoded",
        ),
        pytest.param(
            "ascend", "npu", "builtin",
            NpuTorchBackend, None,
            id="npu_torch_lib",
        ),
    ],
)
def test_build(segment, torch_lib, profiler_config, expected_cls, check_torch_lib):
    """_build 注入 _segment_name = yaml 段名；device_type() 返回段名（非硬编码）。"""
    from ttk.core_modules.framework_api.backends import _build

    profile = {"torch_lib": torch_lib, "profiler": profiler_config}
    b = _build("torch", segment, profile)
    assert isinstance(b, expected_cls)
    assert b.device_type() == segment
    if check_torch_lib is not None:
        assert b.torch_lib == check_torch_lib
