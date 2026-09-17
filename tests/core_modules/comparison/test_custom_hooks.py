#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""compare 钩子（pre_compare / compare）的哨兵处理：str 哨兵不进钩子；
None golden（抑制位置）交由插件钩子自行处理，不禁用钩子。"""

import numpy as np

from ttk.core_modules.comparison.custom import _can_customize, try_custom_compare


class _Case:
    testcase_name = "case_x"
    output_dist = None
    tensors = ()
    scalars = ()
    attributes = {}
    original_dict = {}


def _cmp(o0, o1, g0, g1):
    return [
        {"pass": bool(np.array_equal(o0, g0)), "precision": 100},
        {"pass": bool(np.array_equal(o1, g1)), "precision": 100},
    ]


def test_str_sentinel_not_customizable():
    """golden 含 str 哨兵（SUPPRESSED/UNSUPPORTED/GOLDEN_FAILURE 等）不进钩子。"""
    outputs = [np.array([1.0]), np.array([2.0])]
    assert _can_customize(outputs, [np.array([1.0]), "GOLDEN_FAILURE"]) is False
    assert _can_customize(outputs, [np.array([1.0]), np.array([2.0])]) is True


def test_custom_compare_runs_when_golden_none():
    """golden 含 None（抑制位置）时钩子仍接管——抑制位置由插件 compare 自行处理。"""
    outputs = [np.array([1.0]), np.array([2.0])]
    goldens = [np.array([1.0]), None]

    def cmp_handles_none(o0, o1, g0, g1):
        return [
            {"pass": bool(np.array_equal(o0, g0)), "precision": 100},
            {"pass": g1 is None, "precision": "SUPPRESSED"},
        ]

    precision, log, passed = try_custom_compare(_Case(), list(outputs), list(goldens), cmp_handles_none)
    assert precision == "100%,SUPPRESSED"
    assert passed is True


def test_custom_compare_runs_when_all_goldens_real():
    """golden 全为真实数组时钩子正常接管。"""
    outputs = [np.array([1.0]), np.array([2.0])]
    goldens = [np.array([1.0]), np.array([2.0])]
    precision, log, passed = try_custom_compare(_Case(), list(outputs), list(goldens), _cmp)
    assert precision == "100%,100%"
    assert passed is True
