#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.framework_api.profiling:
_build_tol_options, _prepare_nested_tensors, _to_non_contiguous_view,
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


from ttk.core_modules.framework_api.profiling import (
    _build_tol_options,
)
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


class TestBuildTolOptions:

    def test_none_precision_tolerances(self):
        case = MagicMock()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None
        opts = _build_tol_options(case)
        assert opts['rtol'] is None
        assert opts['ptol'] is None
        assert opts['atol'] is None

    def test_with_precision_tolerances(self):
        case = MagicMock()
        ptols = [(0.01, 0.02), (0.03, 0.04)]
        case.precision_tolerances = ptols
        case.flat_precision_tolerances = ptols
        case.flat_absolute_precision = 1e-5
        opts = _build_tol_options(case)
        assert opts['rtol'] == [0.01, 0.03]
        assert opts['ptol'] == [0.02, 0.04]
        assert opts['atol'] == 1e-5

    def test_empty_precision_tolerances(self):
        case = MagicMock()
        case.precision_tolerances = []
        case.flat_precision_tolerances = []
        case.flat_absolute_precision = None
        opts = _build_tol_options(case)
        assert opts['rtol'] == []
        assert opts['ptol'] == []


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


