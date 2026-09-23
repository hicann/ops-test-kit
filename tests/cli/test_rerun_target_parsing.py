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

from argparse import Namespace
from types import SimpleNamespace

from ttk.cli.bridge import _apply_plugin_rerun_args


def test_rerun_targets_trim_spaces_and_ignore_empty_items():
    switches = SimpleNamespace()

    _apply_plugin_rerun_args(
        switches,
        Namespace(plugin=None, rerun="precision_status, timeout, "),
    )

    assert switches.rerun_targets == ["precision_status", "timeout"]
