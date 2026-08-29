# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for manual-data mode NpuInstance."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from ttk.core_modules.npu.instance_refactor import NpuInstance


def _instance():
    instance = NpuInstance.__new__(NpuInstance)
    instance.switches = SimpleNamespace(
        manual_data_mode="prepare",
        device_count=-1,
        dev_plat="Ascend950",
        short_soc_version=None,
        mode=SimpleNamespace(is_model=lambda: False, is_online_board=lambda: True),
        compile_only=False,
        validate_only=False,
    )
    return instance


def test_prepare_uses_one_logical_worker_without_querying_devices(monkeypatch):
    instance = _instance()
    dsmi = MagicMock(side_effect=AssertionError("DSMI must not be queried"))
    monkeypatch.setattr("ttk.core_modules.npu.instance_refactor.DSMIInterface", dsmi)

    instance.get_device_count()

    assert instance.switches.device_count == 1
    assert not dsmi.called
