#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.npu.op.comparison:
- comparing() function: 4-group comparison logic (dyn/golden, cst/golden, bin/golden, bin/dyn)
- ComparisonResult output
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ttk.core_modules.npu.op.comparison import comparing


@pytest.fixture(autouse=True)
def _mock_global_storage():
    mock_gs = MagicMock()
    mock_gs.compare_method = "close"
    with patch('ttk.core_modules.npu.op.comparison.get_global_storage', return_value=mock_gs):
        yield mock_gs


@pytest.fixture(autouse=True)
def _mock_outputs_to_numpy():
    with patch('ttk.core_modules.npu.op.comparison.__outputs_to_numpy_arrays',
               lambda outputs, dtypes: None):
        yield


def _call_comparing(mock_compare_side_effect):
    with patch('ttk.core_modules.npu.op.comparison.compare') as mock_compare:
        mock_compare.side_effect = mock_compare_side_effect
        result = comparing(
            "dyn_k", "cst_k", "bin_k",
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            ((1e-3, 1e-3),),
            ("float32",),
        )
        return result, mock_compare


class TestComparingPass:

    def test_all_pass(self):
        result, _ = _call_comparing([
            ("1.0", "", True),
            ("1.0", "", True),
            ("1.0", "", True),
            ("1.0", "", True),
        ])
        assert result.passed == "PASS"
        assert result.dyn_precision == "1.0"
        assert result.cst_precision == "1.0"
        assert result.bin_precision == "1.0"

    def test_dyn_fail_means_overall_fail(self):
        result, _ = _call_comparing([
            ("0.5", "", False),
            ("1.0", "", True),
            ("1.0", "", True),
            ("1.0", "", True),
        ])
        assert result.passed == "FAIL"
        assert result.dyn_precision == "0.5"

    def test_bin_fail_means_overall_fail(self):
        result, _ = _call_comparing([
            ("1.0", "", True),
            ("1.0", "", True),
            ("0.3", "", False),
            ("1.0", "", True),
        ])
        assert result.passed == "FAIL"
        assert result.bin_precision == "0.3"

    def test_3_comparisons_made(self):
        result, mock_compare = _call_comparing([
            ("1.0", "", True),
            ("1.0", "", True),
            ("1.0", "", True),
        ])
        assert mock_compare.call_count == 3

    def test_comparison_order(self):
        call_log = []
        def track_compare(outputs, goldens, output_dtypes, method, options):
            first = outputs[0]
            label = "dyn" if first is dyn_out else "cst" if first is cst_out else "bin"
            if goldens[0] is golden:
                label += "_vs_golden"
            elif goldens[0] is dyn_out:
                label += "_vs_dyn"
            call_log.append(label)
            return ("1.0", "", True)

        dyn_out = np.array([1.0])
        cst_out = np.array([1.0])
        bin_out = np.array([1.0])
        golden = np.array([1.0])

        with patch('ttk.core_modules.npu.op.comparison.compare', side_effect=track_compare):
            comparing("dyn_k", "cst_k", "bin_k",
                      (dyn_out,), (cst_out,), (bin_out,), (golden,),
                      ((1e-3, 1e-3),), ("float32",))

        assert call_log == [
            "dyn_vs_golden",
            "cst_vs_golden",
            "bin_vs_golden",
        ]

    def test_exception_returns_compare_failure(self):
        with patch('ttk.core_modules.npu.op.comparison.compare', side_effect=RuntimeError("boom")):
            result = comparing(
                "dyn_k", "cst_k", "bin_k",
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                ((1e-3, 1e-3),),
                ("float32",),
            )
        assert result.passed == "COMPARE_FAILURE"

    def test_none_thresholds(self):
        result, _ = _call_comparing([("1.0", "", True)] * 4)
        assert result.passed == "PASS"
