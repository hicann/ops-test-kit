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

from ttk.cli.bridge import _apply_case_selection_args


def test_testcase_filter_trims_spaces_and_ignores_empty_items():
    switches = SimpleNamespace()

    _apply_case_selection_args(
        switches,
        Namespace(testcase="case_a, case_b,", testcase_index=None, testcase_count=None),
    )

    assert switches.selected_testcases == ["case_a", "case_b"]
