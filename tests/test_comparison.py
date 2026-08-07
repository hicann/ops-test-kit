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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.npu.op.comparison import comparing


@pytest.fixture(autouse=True)
def _mock_outputs_to_numpy():
    with patch('ttk.core_modules.npu.op.comparison.__outputs_to_numpy_arrays',
               lambda outputs, dtypes: None):
        yield


def _call_comparing(mock_compare_side_effect):
    with patch('ttk.core_modules.comparison.custom.compare') as mock_compare:
        mock_compare.side_effect = mock_compare_side_effect
        result = comparing(
            "dyn_k", "cst_k", "bin_k",
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            (np.array([1.0]),),
            ("float32",),
            standards=[MagicMock()],
        )
        return result, mock_compare


class TestComparingPass:

    def test_all_pass(self):
        result, _ = _call_comparing([
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
        ])
        assert result.passed == "PASS"
        assert result.dyn_precision == "1.0"
        assert result.cst_precision == "1.0"
        assert result.bin_precision == "1.0"

    def test_dyn_fail_means_overall_fail(self):
        result, _ = _call_comparing([
            ("0.5", "", False, {}),
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
        ])
        assert result.passed == "FAIL"
        assert result.dyn_precision == "0.5"

    def test_bin_fail_means_overall_fail(self):
        result, _ = _call_comparing([
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
            ("0.3", "", False, {}),
        ])
        assert result.passed == "FAIL"
        assert result.bin_precision == "0.3"

    def test_3_comparisons_made(self):
        result, mock_compare = _call_comparing([
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
            ("1.0", "", True, {}),
        ])
        assert mock_compare.call_count == 3

    def test_comparison_order(self):
        call_log = []
        def track_compare(outputs, goldens, output_dtypes, *, standards, third_parties=None):
            first = outputs[0]
            label = "dyn" if first is dyn_out else "cst" if first is cst_out else "bin"
            if goldens[0] is golden:
                label += "_vs_golden"
            elif goldens[0] is dyn_out:
                label += "_vs_dyn"
            call_log.append(label)
            return ("1.0", "", True, {})

        dyn_out = np.array([1.0])
        cst_out = np.array([1.0])
        bin_out = np.array([1.0])
        golden = np.array([1.0])

        with patch('ttk.core_modules.comparison.custom.compare', side_effect=track_compare):
            comparing("dyn_k", "cst_k", "bin_k",
                      (dyn_out,), (cst_out,), (bin_out,), (golden,),
                      ("float32",), standards=[MagicMock()])

        assert call_log == [
            "dyn_vs_golden",
            "cst_vs_golden",
            "bin_vs_golden",
        ]

    def test_exception_returns_compare_failure(self):
        with patch('ttk.core_modules.comparison.custom.compare', side_effect=RuntimeError("boom")):
            result = comparing(
                "dyn_k", "cst_k", "bin_k",
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                (np.array([1.0]),),
                ("float32",),
                standards=[MagicMock()],
            )
        assert result.passed == "COMPARE_FAILURE"

    def test_none_thresholds(self):
        result, _ = _call_comparing([("1.0", "", True, {})] * 3)
        assert result.passed == "PASS"


def _kernel_case(output_distribution=()):
    input_array = np.array([7.0], dtype=np.float32)
    return SimpleNamespace(
        op_name="custom_kernel",
        testcase_name="kernel_custom_compare",
        input_arrays=(input_array,),
        attributes={"axis": 1},
        original_dict={"remark": "kernel context"},
        output_distribution=output_distribution,
    )


def _custom_comparing(case, dyn_outputs, cst_outputs, bin_outputs, goldens,
                      *, pre_compare=None, custom_compare=None):
    return comparing(
        "dyn_k", "cst_k", "bin_k",
        dyn_outputs, cst_outputs, bin_outputs, goldens,
        ("float32",), standards=[MagicMock()], testcase=case,
        pre_compare=pre_compare, custom_compare=custom_compare,
    )


class TestKernelCustomComparing:

    def test_pre_compare_runs_before_custom_compare_for_each_mode(self):
        case = _kernel_case()
        calls = []

        def pre_compare(output, golden):
            calls.append(("pre", float(output[0]), float(golden[0])))
            return [output + 1, golden + 1]

        def custom_compare(output, golden, *, compare_context):
            calls.append(("compare", float(output[0]), float(golden[0])))
            assert compare_context.api_name == "custom_kernel"
            assert compare_context.testcase_name == "kernel_custom_compare"
            assert compare_context.input_tensors is case.input_arrays
            assert compare_context.input_scalars == ()
            assert compare_context.attributes["axis"] == 1
            assert compare_context.csv_fields["remark"] == "kernel context"
            return {"pass": True, "precision": 98.5}

        with patch("ttk.core_modules.comparison.custom.compare") as builtin:
            result = _custom_comparing(
                case,
                [np.array([1.0])],
                [np.array([2.0])],
                [np.array([3.0])],
                [np.array([4.0])],
                pre_compare=pre_compare,
                custom_compare=custom_compare,
            )

        assert not builtin.called
        assert calls == [
            ("pre", 1.0, 4.0), ("compare", 2.0, 5.0),
            ("pre", 2.0, 4.0), ("compare", 3.0, 5.0),
            ("pre", 3.0, 4.0), ("compare", 4.0, 5.0),
        ]
        assert result.dyn_precision == "98.5%"
        assert result.cst_precision == "98.5%"
        assert result.bin_precision == "98.5%"
        assert result.passed == "PASS"
        assert result.metrics == {"dyn": {}, "cst": {}, "bin": {}}

    def test_disabled_mode_keeps_builtin_sentinel_path(self):
        case = _kernel_case()
        custom_calls = []

        def custom_compare(output, golden):
            custom_calls.append(float(output[0]))
            return {"pass": True, "precision": 100.0}

        with patch("ttk.core_modules.comparison.custom.compare") as builtin:
            builtin.return_value = ("DYN_OFF", "", True, {"standard": "sentinel"})
            result = _custom_comparing(
                case,
                ["DYN_OFF"],
                [np.array([2.0])],
                [np.array([3.0])],
                [np.array([4.0])],
                custom_compare=custom_compare,
            )

        assert builtin.call_count == 1
        assert custom_calls == [2.0, 3.0]
        assert result.dyn_precision == "DYN_OFF"
        assert result.cst_precision == "100.0%"
        assert result.bin_precision == "100.0%"
        assert result.metrics["dyn"] == {"standard": "sentinel"}

    def test_pre_compare_without_custom_compare_uses_builtin(self):
        case = _kernel_case()
        seen = []

        def pre_compare(output, golden):
            return [output * 2, golden * 3]

        def builtin(outputs, goldens, _dtypes, *, standards, third_parties=None):
            del standards, third_parties
            seen.append((float(outputs[0][0]), float(goldens[0][0])))
            return "100.0%", "", True, {0: {"standard": "builtin"}}

        with patch("ttk.core_modules.comparison.custom.compare", side_effect=builtin):
            result = _custom_comparing(
                case,
                [np.array([1.0])],
                [np.array([2.0])],
                [np.array([3.0])],
                [np.array([4.0])],
                pre_compare=pre_compare,
            )

        assert seen == [(2.0, 12.0), (4.0, 12.0), (6.0, 12.0)]
        assert result.passed == "PASS"
        assert result.metrics["bin"] == {0: {"standard": "builtin"}}

    def test_inplace_pre_compare_gets_fresh_golden_for_each_mode(self):
        case = _kernel_case()
        golden_values = []

        def pre_compare(_output, golden):
            golden[:] += 1

        def custom_compare(_output, golden):
            golden_values.append(float(golden[0]))
            return {"pass": True, "precision": 100.0}

        original_golden = np.array([4.0])
        _custom_comparing(
            case,
            [np.array([1.0])],
            [np.array([2.0])],
            [np.array([3.0])],
            [original_golden],
            pre_compare=pre_compare,
            custom_compare=custom_compare,
        )

        assert golden_values == [5.0, 5.0, 5.0]
        np.testing.assert_array_equal(original_golden, np.array([4.0]))

    def test_custom_compare_gets_fresh_golden_for_each_mode(self):
        case = _kernel_case()
        golden_values = []

        def custom_compare(_output, golden):
            golden_values.append(float(golden[0]))
            golden[:] += 1
            return {"pass": True, "precision": 100.0}

        original_golden = np.array([4.0])
        _custom_comparing(
            case,
            [np.array([1.0])],
            [np.array([2.0])],
            [np.array([3.0])],
            [original_golden],
            custom_compare=custom_compare,
        )

        assert golden_values == [4.0, 4.0, 4.0]
        np.testing.assert_array_equal(original_golden, np.array([4.0]))

    def test_kernel_output_distribution_is_folded(self):
        case = _kernel_case(output_distribution=(2,))
        folded_lengths = []

        def custom_compare(output_list, golden_list):
            folded_lengths.append((len(output_list), len(golden_list)))
            return {"pass": True, "precision": 100.0}

        outputs = [np.array([1.0]), np.array([2.0])]
        goldens = [np.array([1.0]), np.array([2.0])]
        result = comparing(
            "dyn_k", "cst_k", "bin_k",
            list(outputs), list(outputs), list(outputs), goldens,
            ("float32", "float32"), standards=[MagicMock(), MagicMock()],
            testcase=case, custom_compare=custom_compare,
        )

        assert folded_lengths == [(2, 2), (2, 2), (2, 2)]
        assert result.passed == "PASS"

    def test_flat_kernel_bytes_are_reshaped_for_custom_compare(self):
        case = _kernel_case()
        seen_shapes = []

        def custom_compare(output, golden):
            seen_shapes.append((output.shape, golden.shape))
            return {"pass": output.shape == golden.shape, "precision": 100.0}

        flat_output = np.arange(6, dtype=np.float32)
        golden = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = _custom_comparing(
            case,
            [flat_output.copy()],
            [flat_output.copy()],
            [flat_output.copy()],
            [golden],
            custom_compare=custom_compare,
        )

        assert seen_shapes == [((2, 3), (2, 3))] * 3
        assert result.passed == "PASS"

    def test_custom_compare_exception_returns_compare_failure(self):
        case = _kernel_case()

        def custom_compare(_output, _golden):
            raise RuntimeError("kernel compare failed")

        result = _custom_comparing(
            case,
            [np.array([1.0])],
            [np.array([2.0])],
            [np.array([3.0])],
            [np.array([4.0])],
            custom_compare=custom_compare,
        )

        assert result.passed == "COMPARE_FAILURE"
