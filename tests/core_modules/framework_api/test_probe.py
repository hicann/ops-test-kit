# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Task 5: _probe (hardware detection D-scheme) + get_backend(force_cpu).

_probe: cuda lib skips import_module; non-cuda import_module + getattr
is_available; whole body wrapped in try/except Exception (covers
ImportError / RuntimeError / AttributeError) returning False + warning.
get_backend(force_cpu=True) -> CpuTorchBackend; else iterate _hw_profiles
in order, _probe each non-cpu profile, build first hit; cpu fallback.
"""

from __future__ import annotations

from ttk.core_modules.framework_api.backends import _probe, get_backend
from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend


def test_probe_missing_torch_lib_returns_false(caplog):
    """A profile missing torch_lib is rejected up front (not swallowed as
    "device not available"). Both empty {} and {profiler: ...} -> False +
    warning naming torch_lib."""
    for bad in ({}, {"profiler": "builtin"}):
        caplog.clear()
        assert _probe(bad) is False
        assert any("torch_lib" in rec.message for rec in caplog.records)


def test_force_cpu_returns_cpu_backend():
    assert isinstance(get_backend(force_cpu=True), CpuTorchBackend)
