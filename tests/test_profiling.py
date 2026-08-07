#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.framework_api.profiling:
_prepare_nested_tensors, _to_non_contiguous_view,
_default_generate_inputs.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

def _prepare_nested_tensors(all_tensors, output_tensor_indexes):
    """Split nested tensors into inputs and out tensors based on top-level indexes."""
    out_indices = set(output_tensor_indexes or ())
    inputs = [t for i, t in enumerate(all_tensors) if i not in out_indices]
    out_tensors = [all_tensors[i] for i in sorted(out_indices)]
    return inputs, out_tensors


from ttk.core_modules.framework_api.input_generation import (
    to_non_contiguous_view, default_generate_inputs,
)


def _make_switches(**overrides):
    from ttk.utilities.classes import SWITCHES
    sw = SWITCHES()
    sw.input_distribution = "uniform"
    for k, v in overrides.items():
        setattr(sw, k, v)
    return sw


def _make_testcase(shapes, dtypes, attrs=None, output_tensor_indexes=(),
                   pure_output_indexes=None, storage_shapes=(),
                   view_strides=(), view_offsets=(),
                   input_data_ranges=None):
    case = MagicMock()
    case.flat_tensor_view_shapes = shapes
    case.flat_tensor_dtypes = dtypes
    case.flat_input_data_ranges = input_data_ranges or ()
    case.pure_output_indexes = pure_output_indexes or []
    case.flat_tensor_storage_shapes = storage_shapes
    case.flat_tensor_view_strides = view_strides
    case.flat_tensor_view_offsets = view_offsets

    def flat_storage_side_effect(idx):
        if storage_shapes and idx < len(storage_shapes):
            val = storage_shapes[idx]
            if val is not None:
                return val
        return shapes[idx] if idx < len(shapes) else None

    def flat_stride_side_effect(idx):
        if view_strides and idx < len(view_strides):
            s = view_strides[idx]
            if s is not None and s != ():
                return s
        from ttk.utilities.container_utils import shape_stride
        v = shapes[idx] if idx < len(shapes) else None
        return shape_stride(v) if v is not None else None

    def flat_offset_side_effect(idx):
        if view_offsets and idx < len(view_offsets):
            val = view_offsets[idx]
            if val is not None:
                return val
        return 0

    case.flat_storage_shape.side_effect = flat_storage_side_effect
    case.flat_view_stride.side_effect = flat_stride_side_effect
    case.flat_view_offset.side_effect = flat_offset_side_effect
    return case


class TestPrepareNestedTensors:

    def test_no_output_indexes(self):
        tensors = [1, 2, 3]
        inputs, out = _prepare_nested_tensors(tensors, ())
        assert inputs == [1, 2, 3]
        assert out == []


class TestToNonContiguousView:

    def test_stride_with_offset(self):
        storage = np.arange(12, dtype=np.float32)
        view_shape = (3, 2)
        view_stride = (4, 2)
        view_offset = 1
        result = to_non_contiguous_view(storage, view_shape, view_stride, view_offset)
        assert result.shape == (3, 2)
        expected = np.array([[1, 3], [5, 7], [9, 11]], dtype=np.float32)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_stride_no_offset(self):
        storage = np.arange(6, dtype=np.float32).reshape(2, 3)
        view_shape = (3,)
        view_stride = (2,)
        view_offset = 0
        result = to_non_contiguous_view(storage, view_shape, view_stride, view_offset)
        assert result.shape == (3,)

    def test_stride_only_row(self):
        storage = np.arange(12, dtype=np.float32).reshape(3, 4)
        view_shape = (3, 2)
        view_stride = (4, 1)
        view_offset = 0
        result = to_non_contiguous_view(storage, view_shape, view_stride, view_offset)
        assert result.shape == (3, 2)


class TestDefaultGenerateInputs:

    def test_basic_input_generation(self):
        case = _make_testcase(
            shapes=((4,), (4,)),
            dtypes=('float32', 'float32'))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 2
        assert inputs[0].shape == (4,)
        assert inputs[0].dtype == np.float32
        assert inputs[1].shape == (4,)
        assert inputs[1].dtype == np.float32

    def test_none_placeholder(self):
        case = _make_testcase(
            shapes=((4,), None),
            dtypes=('float32', None))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 2
        assert inputs[0] is not None
        assert inputs[1] is None

    def test_pure_output_indexes_get_ones(self):
        case = _make_testcase(
            shapes=((3,), (3,)),
            dtypes=('float32', 'float32'),
            pure_output_indexes=[1])
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 2
        np.testing.assert_array_equal(inputs[1], np.ones(3, dtype=np.float32))

    def test_int_dtype_generation(self):
        case = _make_testcase(
            shapes=((5,),),
            dtypes=('int64',))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert inputs[0].dtype == np.int64

    def test_multiple_tensors_mixed(self):
        case = _make_testcase(
            shapes=((2, 3), (6,), (2,)),
            dtypes=('float16', 'float32', 'int32'),
            pure_output_indexes=[2])
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 3
        assert inputs[0].dtype == np.float16
        assert inputs[1].dtype == np.float32
        assert inputs[2].dtype == np.int32
        np.testing.assert_array_equal(inputs[2], np.ones(2, dtype=np.int32))

    def test_non_contiguous_view_generated(self):
        case = _make_testcase(
            shapes=((3, 4),),
            dtypes=('float32',),
            view_strides=((8, 1),),
            view_offsets=(2,))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 1
        assert inputs[0].shape == (3, 4)

    def test_multiple_output_indexes(self):
        tensors = ['a', 'b', 'c', 'd']
        inputs, out = _prepare_nested_tensors(tensors, (1, 3))
        assert inputs == ['a', 'c']
        assert out == ['b', 'd']

    def test_none_output_indexes(self):
        tensors = [1, 2]
        inputs, out = _prepare_nested_tensors(tensors, None)
        assert inputs == [1, 2]
        assert out == []


class TestUnfoldOutputs:

    def test_all_single(self):
        from ttk.utilities.container_utils import deep_flatten
        arr0 = np.array([1, 2])
        arr1 = np.array([3, 4])
        result = deep_flatten([arr0, arr1])
        assert len(result) == 2
        assert result[0] is arr0
        assert result[1] is arr1

    def test_with_tensor_list(self):
        from ttk.utilities.container_utils import deep_flatten
        arr0a = np.array([1])
        arr0b = np.array([2])
        arr1 = np.array([3])
        result = deep_flatten([[arr0a, arr0b], arr1])
        assert len(result) == 3
        assert result[0] is arr0a
        assert result[1] is arr0b
        assert result[2] is arr1

    def test_empty(self):
        from ttk.utilities.container_utils import deep_flatten
        assert deep_flatten([]) == ()

    def test_tuple_also_unfolded(self):
        from ttk.utilities.container_utils import deep_flatten
        arr0a = np.array([1])
        arr0b = np.array([2])
        arr1 = np.array([3])
        result = deep_flatten([(arr0a, arr0b), arr1])
        assert len(result) == 3
        assert result[0] is arr0a
        assert result[1] is arr0b
        assert result[2] is arr1


class TestApplyPreCompareSkip:

    def _make_switches_with_path(self, plugin_path=None):
        sw = _make_switches()
        sw.plugin_path = plugin_path
        return sw

    def test_no_plugin_path_returns_unchanged(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "softmax_v2"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path=None)
        result = [np.array([1, 2])]
        golden = [np.array([3, 4])]
        _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(golden[0], [3, 4])

    def test_no_spec_found_returns_unchanged(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "nonexistent_op"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        result = [np.array([1, 2])]
        golden = [np.array([3, 4])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=None):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(golden[0], [3, 4])

    def test_spec_has_no_pre_compare_returns_unchanged(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        result = [np.array([1, 2])]
        golden = [np.array([3, 4])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=None):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(golden[0], [3, 4])

    def test_empty_golden_skips(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        result = [np.array([1, 2])]
        golden = []
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1, 2])

    def test_sentinel_golden_skips(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        result = [np.array([1, 2])]
        golden = ["SUPPRESSED"]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1, 2])

    def test_empty_result_nps_skips(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        result = []
        golden = [np.array([3.0, 4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(golden[0], [3.0, 4.0])


class TestApplyPreCompareInplace:

    def test_inplace_no_tensor_list(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            for arr in outputs:
                arr[:] = arr * 2
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])

    def test_inplace_with_tensor_list(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            for output in outputs:
                if isinstance(output, list):
                    for arr in output:
                        arr[:] = np.sort(arr)
                else:
                    output[:] = np.sort(output)
        case = MagicMock()
        case.api_name = "topk"
        case.testcase_name = "test_0"
        case.output_dist = (2,)
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([3.0, 1.0]), np.array([4.0, 2.0])]
        golden = [np.array([3.0, 1.0]), np.array([4.0, 2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [1.0, 3.0])
        np.testing.assert_array_equal(result[1], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [1.0, 3.0])
        np.testing.assert_array_equal(golden[1], [2.0, 4.0])


class TestApplyPreCompareReturnMode:

    def test_return_mode_no_tensor_list(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            return [arr * 2 for arr in outputs]
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])

    def test_return_mode_mixed_tensor_list_and_single(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            npu_tl, npu_single, golden_tl, golden_single = outputs
            return [
                [a * 2 for a in npu_tl], npu_single * 2,
                [a * 2 for a in golden_tl], golden_single * 2,
            ]
        case = MagicMock()
        case.api_name = "mixed_op"
        case.testcase_name = "test_0"
        case.output_dist = (2, 0)
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        golden = [np.array([4.0]), np.array([5.0]), np.array([6.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [2.0])
        np.testing.assert_array_equal(result[1], [4.0])
        np.testing.assert_array_equal(result[2], [6.0])
        np.testing.assert_array_equal(golden[0], [8.0])
        np.testing.assert_array_equal(golden[1], [10.0])
        np.testing.assert_array_equal(golden[2], [12.0])

    def test_return_mode_nested_length_mismatch_raises(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            return [outputs[0]]
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0])]
        golden = [np.array([2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            with pytest.raises(ValueError, match="pre_compare returned len="):
                _apply_pre_compare(case, result, golden, sw)

    def test_return_mode_unfold_length_mismatch_raises(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            npu_tl, npu_single, golden_tl, golden_single = outputs
            return [
                [npu_tl[0], npu_tl[1], npu_single], npu_single,
                [golden_tl[0], golden_tl[1], golden_single], golden_single,
            ]
        case = MagicMock()
        case.api_name = "mixed_op"
        case.testcase_name = "test_0"
        case.output_dist = (2, 0)
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        golden = [np.array([4.0]), np.array([5.0]), np.array([6.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            with pytest.raises(ValueError, match="unfolded len mismatch"):
                _apply_pre_compare(case, result, golden, sw)

    def test_exception_propagates(self):
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare
        def pre_compare(*outputs):
            raise RuntimeError("boom")
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0])]
        golden = [np.array([2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            with pytest.raises(RuntimeError, match="boom"):
                _apply_pre_compare(case, result, golden, sw)


class TestPreCompareEndToEnd:

    def test_do_profile_calls_pre_compare_before_compare(self):
        """Verify _do_profile calls _apply_pre_compare and passes transformed data to _evaluate_eager_precision."""
        from ttk.core_modules.framework_api import profiling as prof_module

        def pre_compare(*outputs):
            return [arr * 2 for arr in outputs]

        captured = {}

        def fake_evaluate_eager_precision(testcase, raw_inputs, result_nps, golden_nps,
                                     switches, perf, return_struct, third_parties=None):
            captured["result"] = list(result_nps)
            captured["golden"] = list(golden_nps)

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.get_param_plan.return_value = MagicMock()
        case.is_torch_dtype_support.return_value = True

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        mock_backend = MagicMock()
        mock_backend.use_device.return_value = False
        # Task 7 contract: graph mode guards on is_npu() (CPU-style -> False,
        # graph skipped). MagicMock auto-returns truthy, so pin explicitly.
        mock_backend.is_npu.return_value = False

        mock_return_struct = MagicMock()

        fake_result = [np.array([1.0, 2.0])]
        fake_golden = [np.array([10.0, 20.0])]

        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr",
                   side_effect=lambda api, attr, path: pre_compare if attr == "pre_compare" else None), \
             patch.object(prof_module, "get_process_context"), \
             patch.object(prof_module, "resolve_api", return_value=(MagicMock(__name__="add"), False)), \
             patch.object(prof_module, "generate_inputs", return_value=[np.array([1.0])]), \
             patch.object(prof_module, "_dump_inputs"), \
             patch.object(prof_module, "DeviceLock"), \
             patch.object(prof_module, "_profiling_print"), \
             patch.object(prof_module, "_execute_eager", return_value=(fake_result, MagicMock())), \
             patch.object(prof_module, "_dump_outputs"), \
             patch.object(prof_module, "_generate_golden_data", return_value=fake_golden), \
             patch.object(prof_module, "_evaluate_eager_precision", side_effect=fake_evaluate_eager_precision):
            prof_module._do_profile(case, mock_backend, {}, {}, 0, sw, mock_return_struct)

        np.testing.assert_array_equal(captured["result"][0], [2.0, 4.0])
        np.testing.assert_array_equal(captured["golden"][0], [20.0, 40.0])


class TestTryCustomCompareSkip:

    def _make_switches_with_path(self, plugin_path=None):
        sw = _make_switches()
        sw.plugin_path = plugin_path
        return sw

    def test_no_plugin_path_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path=None)
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        assert _try_custom_compare(case, result, golden, sw) is None

    def test_no_spec_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "nonexistent"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=None):
            result = [np.array([1.0])]
            golden = [np.array([2.0])]
            assert _try_custom_compare(case, result, golden, sw) is None

    def test_spec_has_no_compare_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=None):
            result = [np.array([1.0])]
            golden = [np.array([2.0])]
            assert _try_custom_compare(case, result, golden, sw) is None

    def test_empty_result_nps_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            result = []
            golden = [np.array([2.0])]
            assert _try_custom_compare(case, result, golden, sw) is None

    def test_empty_golden_nps_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            result = [np.array([1.0])]
            golden = []
            assert _try_custom_compare(case, result, golden, sw) is None

    def test_sentinel_golden_returns_none(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path="/fake/path")
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=MagicMock()):
            result = [np.array([1.0])]
            golden = ["SUPPRESSED"]
            assert _try_custom_compare(case, result, golden, sw) is None


class TestTryCustomCompareAdapt:

    def _make_case(self, api_name="add", output_dist=()):
        case = MagicMock()
        case.api_name = api_name
        case.testcase_name = "test_0"
        case.output_dist = output_dist
        return case

    def _make_switches(self):
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        return sw

    def test_single_dict_return(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return {"pass": True, "precision": 99.5}
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            p, log, is_pass = _try_custom_compare(case, result, golden, sw)
        assert p == "99.5%"
        assert is_pass is True

    def test_flat_list_dict_return(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return [
                {"pass": True, "precision": 100.0},
                {"pass": False, "precision": "mismatch"},
            ]
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0]), np.array([2.0])]
        golden = [np.array([1.0]), np.array([3.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            p, log, is_pass = _try_custom_compare(case, result, golden, sw)
        assert p == "100.0%,mismatch"
        assert is_pass is False

    def test_nested_list_dict_return(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return [
                [{"pass": True, "precision": 100.0}, {"pass": True, "precision": 95.0}],
                {"pass": False, "precision": "fail"},
            ]
        case = self._make_case(output_dist=(2, 0))
        sw = self._make_switches()
        result = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        golden = [np.array([1.0]), np.array([2.0]), np.array([9.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            p, log, is_pass = _try_custom_compare(case, result, golden, sw)
        assert p == "100.0%,95.0%,fail"
        assert is_pass is False

    def test_precision_float_formatted(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return {"pass": True, "precision": 99.98}
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            p, _, _ = _try_custom_compare(case, result, golden, sw)
        assert p == "99.98%"

    def test_precision_str_passthrough(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return {"pass": True, "precision": "cosine:0.9998"}
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            p, _, _ = _try_custom_compare(case, result, golden, sw)
        assert p == "cosine:0.9998"

    def test_error_info_in_log(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return {"pass": False, "precision": 50.0, "error_info": "3 mismatches"}
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            _, log, _ = _try_custom_compare(case, result, golden, sw)
        assert log == "Output 0: 3 mismatches\n"

    def test_multiple_error_info_lines(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return [
                {"pass": False, "precision": 50.0, "error_info": "first"},
                {"pass": False, "precision": 25.0, "error_info": "second"},
            ]
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0]), np.array([2.0])]
        golden = [np.array([3.0]), np.array([4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            _, log, _ = _try_custom_compare(case, result, golden, sw)
        assert log == "Output 0: first\nOutput 1: second\n"

    def test_missing_pass_raises(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return {"precision": 99.0}
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(ValueError, match=r"missing required key\(s\): 'pass'"):
                _try_custom_compare(case, result, golden, sw)

    def test_non_dict_item_raises_clear_error(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return [{"pass": True, "precision": 100.0}, "invalid"]
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0]), np.array([2.0])]
        golden = [np.array([1.0]), np.array([2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(ValueError, match=r"compare output\[1\] is not a dict"):
                _try_custom_compare(case, result, golden, sw)

    def test_non_dict_non_list_raises(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return 42
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(ValueError, match="must return dict or list"):
                _try_custom_compare(case, result, golden, sw)

    def test_empty_list_raises(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            return []
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(ValueError, match="empty list"):
                _try_custom_compare(case, result, golden, sw)

    def test_exception_propagates(self):
        from ttk.core_modules.framework_api.profiling import _try_custom_compare
        def compare(*outputs):
            raise RuntimeError("boom")
        case = self._make_case()
        sw = self._make_switches()
        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(RuntimeError, match="boom"):
                _try_custom_compare(case, result, golden, sw)


class TestCustomCompareEndToEnd:

    def test_evaluate_precision_uses_custom_compare(self):
        """Verify _evaluate_eager_precision uses custom compare when available."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            return {"pass": True, "precision": 100.0}

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr",
                   side_effect=lambda *a, **k: custom_compare if a[-2] == "compare" else None):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "100.0%"
        assert return_struct.precision_status == "PASS"

    def test_evaluate_precision_falls_back_to_builtin(self):
        """Verify _evaluate_eager_precision uses built-in compare when no custom compare."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = None  # no plugin path → no custom compare

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        captured = {}

        def fake_compare(*args, **kwargs):
            captured["called"] = True
            return "100.0%", "log", True, {}

        with patch.object(prof_module, "compare", side_effect=fake_compare):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert captured.get("called") is True
        assert return_struct.eager_precision == "100.0%"

    def test_evaluate_precision_custom_compare_failure(self):
        """Verify custom compare exception → COMPARE_FAILURE."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            raise RuntimeError("custom boom")

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr",
                   side_effect=lambda *a, **k: custom_compare if a[-2] == "compare" else None):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "COMPARE_FAILURE"
        assert return_struct.precision_status == "FAIL"
