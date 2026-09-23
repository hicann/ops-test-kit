#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# SPDX-License-Identifier: CANN-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from types import SimpleNamespace

import pytest

from ttk.cli.bridge import _apply_dump_config


def _config(calls):
    return SimpleNamespace(
        enable_input=lambda: calls.append("in"),
        enable_output=lambda: calls.append("out"),
        enable_golden=lambda: calls.append("golden"),
        enable_xpu=lambda: calls.append("xpu"),
        enable_all=lambda: calls.append("full"),
    )


def test_unknown_dump_mode_is_rejected_before_mutating_config():
    calls = []

    with pytest.raises(ValueError, match=r"Unsupported --dump mode\(s\): typo"):
        _apply_dump_config(_config(calls), "in, typo")

    assert calls == []


def test_dump_modes_are_trimmed_before_application():
    calls = []

    _apply_dump_config(_config(calls), " in, xpu ")

    assert calls == ["in", "xpu"]
