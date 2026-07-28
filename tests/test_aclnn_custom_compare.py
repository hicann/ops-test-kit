#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ACLNN TestSpec pre_compare and custom compare integration tests."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ttk.core_modules.npu.op_api.comparison import Comparator


def _context(outputs, goldens, output_dist=()):
    input_tensor = np.array([7.0], dtype=np.float32)
    return SimpleNamespace(
        api_name="aclnnTestCustomCompare",
        testcase_name="custom_compare_case",
        tensors=(input_tensor,),
        scalars=(np.array(0.5, dtype=np.float32),),
        attributes={"axis": -1},
        original_dict={"remark": "context-test"},
        output_dist=output_dist,
        prof_result=SimpleNamespace(
            output_bytes=list(outputs),
            output_view_shapes=[value.shape for value in outputs],
        ),
        golden_tensors=list(goldens),
        flat_output_dtypes=("float32",) * len(outputs),
        flat_precision_tolerances=None,
        flat_absolute_precision=None,
    )


def _run(context, spec_attrs, builtin_compare=None):
    def get_attr(_api_name, attr_name, plugin_path):
        assert plugin_path == "/tmp/test-plugin"
        return spec_attrs.get(attr_name)

    storage = SimpleNamespace(compare_method=None, plugin_path="/tmp/test-plugin")
    with patch("ttk.core_modules.npu.op_api.comparison.get_global_storage", return_value=storage), \
            patch("ttk.core_modules.npu.op_api.comparison.get_spec_attr", side_effect=get_attr), \
            patch.object(Comparator, "_output_bytes_to_tensors"), \
            patch("ttk.core_modules.npu.op_api.comparison.compare",
                  side_effect=builtin_compare) as fallback:
        result = Comparator(context).compare()
    return result, fallback


def test_aclnn_pre_compare_runs_before_custom_compare():
    context = _context(
        [np.array([2.0, 1.0], dtype=np.float32)],
        [np.array([1.0, 2.0], dtype=np.float32)],
    )
    calls = []

    def pre_compare(output, golden):
        calls.append("pre_compare")
        output[:] = np.sort(output)
        golden[:] = np.sort(golden)

    def custom_compare(output, golden):
        calls.append("compare")
        assert np.array_equal(output, golden)
        return {"pass": True, "precision": 100.0}

    result, fallback = _run(
        context,
        {"pre_compare": pre_compare, "compare": custom_compare},
    )

    assert calls == ["pre_compare", "compare"]
    assert result.precision == "100.0%"
    assert result.passed == "PASS"
    assert result.metrics == {}
    fallback.assert_not_called()


def test_aclnn_custom_compare_gets_context_only_by_explicit_opt_in():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.0], dtype=np.float32)],
    )
    captured = {}

    def custom_compare(output, golden, *, compare_context):
        captured["context"] = compare_context
        return {"pass": np.array_equal(output, golden), "precision": 100.0}

    result, fallback = _run(context, {"compare": custom_compare})

    compare_context = captured["context"]
    assert compare_context.api_name == context.api_name
    assert compare_context.testcase_name == context.testcase_name
    assert compare_context.input_tensors is context.tensors
    assert compare_context.input_scalars is context.scalars
    assert compare_context.attributes == context.attributes
    assert compare_context.csv_fields == context.original_dict
    assert result.passed == "PASS"
    fallback.assert_not_called()


def test_aclnn_custom_compare_var_kwargs_do_not_receive_context():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.0], dtype=np.float32)],
    )
    captured = {}

    def custom_compare(output, golden, **kwargs):
        captured.update(kwargs)
        return {"pass": np.array_equal(output, golden), "precision": 100.0}

    result, fallback = _run(context, {"compare": custom_compare})

    assert "compare_context" not in captured
    assert result.passed == "PASS"
    fallback.assert_not_called()


def test_aclnn_none_batch_slice_info_is_not_forwarded():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.0], dtype=np.float32)],
    )
    context.batch_slice_info = None
    captured = {}

    def custom_compare(output, golden, **kwargs):
        captured.update(kwargs)
        return {"pass": np.array_equal(output, golden), "precision": 100.0}

    result, fallback = _run(context, {"compare": custom_compare})

    assert "batch_slice_info" not in captured
    assert result.passed == "PASS"
    fallback.assert_not_called()


def test_aclnn_custom_compare_preserves_tensor_list_nesting():
    context = _context(
        [np.array([1.0]), np.array([2.0]), np.array([3.0])],
        [np.array([1.0]), np.array([2.0]), np.array([3.0])],
        output_dist=(2, 0),
    )

    def pre_compare(output_list, output, golden_list, golden):
        assert len(output_list) == len(golden_list) == 2
        return [
            [value + 1 for value in output_list],
            output + 1,
            [value + 1 for value in golden_list],
            golden + 1,
        ]

    def custom_compare(output_list, output, golden_list, golden):
        assert all(np.array_equal(actual, expected)
                   for actual, expected in zip(output_list, golden_list))
        assert np.array_equal(output, golden)
        return [
            {"pass": True, "precision": "PAIR_PASS"},
            {"pass": True, "precision": "SINGLE_PASS"},
        ]

    result, fallback = _run(
        context,
        {"pre_compare": pre_compare, "compare": custom_compare},
    )

    assert result.precision == "PAIR_PASS,SINGLE_PASS"
    assert result.passed == "PASS"
    fallback.assert_not_called()


def test_aclnn_pre_compare_then_builtin_fallback_keeps_metrics():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([2.0], dtype=np.float32)],
    )

    def pre_compare(output, golden):
        output[:] += 1

    def builtin_compare(outputs, goldens, _dtypes, **_kwargs):
        assert np.array_equal(outputs[0], goldens[0])
        return "100.0%", "", True, {0: {"standard": "stat_rel_err"}}

    result, fallback = _run(
        context,
        {"pre_compare": pre_compare},
        builtin_compare=builtin_compare,
    )

    assert result.passed == "PASS"
    assert result.metrics == {0: {"standard": "stat_rel_err"}}
    fallback.assert_called_once()


def test_aclnn_without_hooks_preserves_builtin_compare_path():
    outputs = [np.array([1.0], dtype=np.float32)]
    goldens = [np.array([1.0], dtype=np.float32)]
    context = _context(outputs, goldens)

    def builtin_compare(actual, expected, dtypes, **kwargs):
        assert actual is context.prof_result.output_bytes
        assert expected is context.golden_tensors
        assert tuple(str(dtype) for dtype in dtypes) == ("float32",)
        assert kwargs["third_parties"] is None
        return "100.0%", "", True, {0: {"standard": "stat_rel_err"}}

    result, fallback = _run(context, {}, builtin_compare=builtin_compare)

    assert result.precision == "100.0%"
    assert result.passed == "PASS"
    assert result.metrics == {0: {"standard": "stat_rel_err"}}
    fallback.assert_called_once()


def test_aclnn_custom_compare_failure_is_reported():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.0], dtype=np.float32)],
    )

    def custom_compare(_output, _golden):
        raise RuntimeError("custom compare failed")

    result, fallback = _run(context, {"compare": custom_compare})

    assert result.precision == "COMPARE_FAILURE"
    assert result.passed == "COMPARE_FAILURE"
    fallback.assert_not_called()


def test_aclnn_invalid_pre_compare_return_is_reported():
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([1.0], dtype=np.float32)],
    )

    def pre_compare(_output, _golden):
        return [np.array([1.0], dtype=np.float32)]

    result, fallback = _run(context, {"pre_compare": pre_compare})

    assert result.precision == "COMPARE_FAILURE"
    assert result.passed == "COMPARE_FAILURE"
    fallback.assert_not_called()


def test_aclnn_custom_compare_loads_from_exact_spec_registration(tmp_path):
    spec_file = tmp_path / "spec.py"
    spec_file.write_text(
        """
class AclnnLoaderSpec:
    @staticmethod
    def pre_compare(output, golden):
        output[:] += 1

    @staticmethod
    def compare(output, golden):
        return {
            "pass": bool((output == golden).all()),
            "precision": "SPEC_LOADED",
        }


__spec__ = {
    "aclnnTestCustomCompare": "AclnnLoaderSpec",
}
""".lstrip(),
        encoding="utf-8",
    )
    context = _context(
        [np.array([1.0], dtype=np.float32)],
        [np.array([2.0], dtype=np.float32)],
    )
    storage = SimpleNamespace(compare_method=None, plugin_path=str(tmp_path))

    with patch("ttk.core_modules.npu.op_api.comparison.get_global_storage", return_value=storage), \
            patch.object(Comparator, "_output_bytes_to_tensors"), \
            patch("ttk.core_modules.npu.op_api.comparison.compare") as fallback:
        result = Comparator(context).compare()

    assert result.precision == "SPEC_LOADED"
    assert result.passed == "PASS"
    fallback.assert_not_called()
