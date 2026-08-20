# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Task 4: hardware config in default.yaml + get_hardware_config + profile validation.

- frameworks.torch.npu segment ships in default.yaml (torch_lib=npu, profiler=builtin).
- get_hardware_config() returns {} when config empty (only-cpu legal default;
  distinct from remote's None).
- _build(fw, name, profile) fail-fast validates torch_lib + profiler before
  instantiating/injecting the backend.
"""
from __future__ import annotations

from ttk.config.loader import get_hardware_config, load_config


def test_hardware_config_returns_frameworks():
    load_config()
    hw = get_hardware_config()
    assert hw["torch"]["npu"]["torch_lib"] == "npu"


def test_get_hardware_config_empty_returns_dict(monkeypatch):
    monkeypatch.setattr("ttk.config.loader.get_config", lambda: {})
    assert get_hardware_config() == {}
