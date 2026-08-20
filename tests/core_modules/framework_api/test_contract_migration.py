# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Task 7: atomic contract migration.

After Task 7:
  - device_name() returns the hardware MODEL (via torch.get_device_name), not
    the torch_lib segment ('npu'/'cuda'/'cpu'). CPU has no model, so its
    device_name override stays == alias() ('cpu').
  - soc_version() is removed (merged into device_name).
  - All string-literal role comparisons (== 'npu'/'gpu'/'cpu') in
    framework_api are gone, replaced by is_npu()/alias()/use_device().
  - get_profiler uses is_npu() + profile.get('profiler') instead of
    device_name() string compares.
"""
from __future__ import annotations

import subprocess

from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend


def test_soc_version_method_removed():
    """soc_version 合并进 device_name，base/backend 不再暴露 soc_version。"""
    cb = CpuTorchBackend()
    cb.torch_lib = "cpu"
    cb.profile = {}
    assert not hasattr(cb, "soc_version"), "soc_version must be removed in Task 7 (merged into device_name)"


def test_no_string_comparison_on_role():
    """grep 确认无 =='npu'/'gpu'/'cpu' 角色字符串逻辑残留 in framework_api。

    Role comparisons (device_type()/device_name()/soc_series() == 'npu'/'gpu'/'cpu')
    are forbidden — routing goes through is_npu()/device_type()/has_device().
    torch_lib value matches (e.g. ``torch_lib == "npu"`` for class derivation in
    _build, ``torch_lib == "cpu"`` for the cpu-skip) are ALLOWED: torch_lib is
    the torch module attribute, not a role.

    Implementation note: _build derives the backend class from torch_lib (cuda/
    mlu/musa -> XpuTorchBackend, npu -> NpuTorchBackend, cpu -> CpuTorchBackend);
    the device_type is config-driven (_segment_name = yaml segment key), so no
    'xpu'/'gpu' role string is ever compared.
    """
    r = subprocess.run(
        ["grep", "-rnE", "--include=*.py", r"""== ?["'](npu|gpu|cpu)["']""", "ttk/core_modules/framework_api/"],
        capture_output=True,
        text=True,
    )
    # filter out allowed torch_lib value matches + docstring example text.
    residue = [ln for ln in r.stdout.splitlines() if ln and "torch_lib" not in ln and "yields alias" not in ln]
    assert not residue, "string comparison residue:\n" + "\n".join(residue)
