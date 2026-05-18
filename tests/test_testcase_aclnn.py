#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.testcase_manager.testcase_aclnn: nested tensor structure,
flat properties, compression, and parameter validation.
"""

import pytest
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.utilities.container_utils import flatten_nested_sequence


NESTED_SHAPES = (((3, 3), (3, 2)), (3, 5))
FLAT_SHAPES = ((3, 3), (3, 5))


class TestFlattenByDistribution:
    """Tests for _flatten_by_distribution static method."""

    def test_flat_values_flat_dist(self):
        assert TestcaseAclnn._flatten_by_distribution(("a", "b"), (0, 0)) == ("a", "b")

    def test_expand_tuple_list(self):
        result = TestcaseAclnn._flatten_by_distribution(("a", ("b", "c")), (0, 2))
        assert result == ("a", "b", "c")

    def test_broadcast_single_in_list(self):
        result = TestcaseAclnn._flatten_by_distribution(("a", ("b",)), (0, 2))
        assert result == ("a", "b", "b")

    def test_broadcast_scalar_to_list(self):
        result = TestcaseAclnn._flatten_by_distribution(("a", "b"), (2, 0))
        assert result == ("a", "a", "b")

    def test_mixed(self):
        result = TestcaseAclnn._flatten_by_distribution(
            (("x", "y"), "z"), (2, 0))
        assert result == ("x", "y", "z")


class TestGetTensorListDistribution:
    """Tests for _get_tensor_list_distribution."""

    def test_flat_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES)
        assert case._get_tensor_list_distribution() == (0, 0)

    def test_nested_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES)
        assert case._get_tensor_list_distribution() == (2, 0)

    def test_all_nested(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(((1,), (2,)), ((3,),)))
        assert case._get_tensor_list_distribution() == (2, 1)

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None)
        assert case._get_tensor_list_distribution() == ()


class TestFlatTensorViewShapes:
    """Tests for flat_tensor_view_shapes property."""

    def test_flat(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES)
        assert case.flat_tensor_view_shapes == FLAT_SHAPES

    def test_nested(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES)
        assert case.flat_tensor_view_shapes == ((3, 3), (3, 2), (3, 5))

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None)
        assert case.flat_tensor_view_shapes is None

    def test_empty(self, make_testcase):
        case = make_testcase(tensor_view_shapes=())
        assert case.flat_tensor_view_shapes == ()


class TestFlatTensorDtypesCompression:
    """Tests for flat_tensor_dtypes: flatten nested TensorList structure.

    In production, _normalize_compressed_fields runs first, expanding
    compressed forms like ('float32',) to (('float32','float32'),'float32').
    The flat property then just flattens nesting.
    """

    def test_already_flat(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=("float32", "float32", "float32"))
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_top_level_compression_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=("float32",))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_per_param_compression_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=("float32", "float32"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_per_tensor_list_explicit(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float32"))
        assert case.flat_tensor_dtypes == ("float32", "int8", "float32")

    def test_per_tensor_list_broadcast_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32",), "float32"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ("float32", "float32", "float32")

    def test_flat_no_nesting(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=FLAT_SHAPES,
            tensor_dtypes=("float32", "float32"))
        assert case.flat_tensor_dtypes == ("float32", "float32")

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_dtypes=None)
        assert case.flat_tensor_dtypes is None


class TestFlatTensorFormatsCompression:
    """Tests for flat_tensor_formats: flatten nested TensorList structure."""

    def test_top_level_compression_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_formats=("ND",))
        case._normalize_compressed_fields()
        assert case.flat_tensor_formats == ("ND", "ND", "ND")

    def test_per_param_compression_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_formats=("ND", "NZ"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_formats == ("ND", "ND", "NZ")

    def test_per_tensor_list_broadcast_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_formats=(("NZ",), "ND"))
        case._normalize_compressed_fields()
        assert case.flat_tensor_formats == ("NZ", "NZ", "ND")


class TestFlatTensorViewOffsetsCompression:
    """Tests for flat_tensor_view_offsets: flatten nested TensorList structure.

    After _normalize_compressed_fields, offsets are fully expanded; the flat
    property just needs to flatten nesting.  Without normalization the raw
    compressed form is preserved (broadcast is normalize's job).
    """

    def test_nested_offsets_flattened(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_view_offsets=((5, 10), 3))
        assert case.flat_tensor_view_offsets == (5, 10, 3)

    def test_flat_offsets_passthrough(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=FLAT_SHAPES,
            tensor_view_offsets=(0, 5))
        assert case.flat_tensor_view_offsets == (0, 5)

    def test_compressed_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_view_offsets=(0,))
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_offsets == (0, 0, 0)

    def test_per_param_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_view_offsets=(5, 10))
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_offsets == (5, 5, 10)

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_view_offsets=None)
        assert case.flat_tensor_view_offsets is None

    def test_empty(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), tensor_view_offsets=())
        assert case.flat_tensor_view_offsets == ()


class TestIsTensorListElement:
    """Tests for _is_tensor_list_element static method."""

    def test_tensor_list(self):
        assert TestcaseAclnn._is_tensor_list_element(((3, 3), (3, 2))) is True

    def test_single_tensor(self):
        assert TestcaseAclnn._is_tensor_list_element((3, 5)) is False

    def test_none(self):
        assert TestcaseAclnn._is_tensor_list_element(None) is False

    def test_empty(self):
        assert TestcaseAclnn._is_tensor_list_element(()) is False


class TestOutputProperties:
    """Tests for output_* and flat_output_* properties."""

    def test_output_dtypes_single_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "float32"), "float32"),
            output_tensor_indexes=(1,))
        assert case.output_dtypes == ("float32",)

    def test_output_dtypes_tensor_list_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float32"),
            output_tensor_indexes=(0,))
        assert case.output_dtypes == (("float32", "int8"),)

    def test_flat_output_dtypes_single_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "float32"), "float32"),
            output_tensor_indexes=(1,))
        assert case.flat_output_dtypes == ("float32",)

    def test_flat_output_dtypes_tensor_list_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float32"),
            output_tensor_indexes=(0,))
        assert case.flat_output_dtypes == ("float32", "int8")

    def test_flat_output_dtypes_multiple_outputs(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(("float32", "int8"), "float32"),
            output_tensor_indexes=(0, 1))
        assert case.flat_output_dtypes == ("float32", "int8", "float32")

    def test_output_view_shapes_single(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(1,))
        assert case.output_view_shapes == ((3, 5),)

    def test_output_view_shapes_tensor_list(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(0,))
        assert case.output_view_shapes == (((3, 3), (3, 2)),)

    def test_flat_output_view_shapes_tensor_list(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(0,))
        assert case.flat_output_view_shapes == ((3, 3), (3, 2))

    def test_flat_output_view_shapes_all(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(0, 1))
        assert case.flat_output_view_shapes == ((3, 3), (3, 2), (3, 5))

    def test_flat_output_storage_shapes_with_defaults(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(1,))
        assert case.flat_output_storage_shapes == ((3, 5),)

    def test_flat_output_storage_shapes_tensor_list(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_storage_shapes=(((10, 10), (10, 10)), (3, 5)),
            output_tensor_indexes=(0,))
        assert case.flat_output_storage_shapes == ((10, 10), (10, 10))

    def test_output_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None, output_tensor_indexes=())
        assert case.output_dtypes == ()
        assert case.flat_output_dtypes == ()


class TestFlatInputDataRanges:
    """Tests for flat_input_data_ranges with nested/compressed support."""

    def test_flat_ranges(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=FLAT_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0))

    def test_nested_expanded(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0), (0.0, 5.0)))
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_top_level_compression(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((-1.0, 1.0),))
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_per_param_compression(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_nested_tensor_list(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)))
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=((3, 3),), input_data_ranges=None)
        assert case.input_data_ranges is None
        assert case.flat_input_data_ranges is None


class TestFlatScalarDataRanges:
    """Tests for flat_scalar_data_ranges with nested/compressed support."""

    def test_flat_ranges(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=('float32', 'int64'),
            scalar_data_ranges=((None, 1.0), (-1.0, 1.0)))
        assert case.flat_scalar_data_ranges == ((None, 1.0), (-1.0, 1.0))

    def test_top_level_compression(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=(('int64', 'int64'), 'float32'),
            scalar_data_ranges=((-1.0, 1.0),))
        case._inferred_scalar_list_dist = (2, 0)
        assert case.flat_scalar_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_per_param_compression(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=(('int64', 'int64'), 'float32'),
            scalar_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._inferred_scalar_list_dist = (2, 0)
        assert case.flat_scalar_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_nested_explicit(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=(('int64', 'int64'), 'float32'),
            scalar_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)))
        case._inferred_scalar_list_dist = (2, 0)
        assert case.flat_scalar_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_none(self, make_testcase):
        case = make_testcase(scalar_dtypes=(), scalar_data_ranges=None)
        assert case.flat_scalar_data_ranges is None


class TestIsScalarListElement:
    """Tests for _is_scalar_list_element static method."""

    def test_scalar_list_is_not_nested(self):
        assert TestcaseAclnn._is_scalar_list_element(("float32", "int8")) is False

    def test_scalar_list_nested(self):
        assert TestcaseAclnn._is_scalar_list_element((("float32", "int8"),)) is True

    def test_single_scalar(self):
        assert TestcaseAclnn._is_scalar_list_element("float32") is False

    def test_none(self):
        assert TestcaseAclnn._is_scalar_list_element(None) is False

    def test_empty(self):
        assert TestcaseAclnn._is_scalar_list_element(()) is False


class TestIsFieldAlreadyNested:
    """Tests for _is_field_already_nested — prevents double-normalization."""

    def test_fully_nested_matches_dist(self):
        field = (('float32', 'float32'), 'float32')
        dist = (2, 0)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is True

    def test_compressed_not_nested(self):
        field = ('float32',)
        dist = (2, 0)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is False

    def test_len_mismatch(self):
        field = ('float32', 'float32')
        dist = (3,)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is False

    def test_inner_len_mismatch(self):
        field = (('float32',), 'float32')
        dist = (2, 0)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is False

    def test_all_single_tensors(self):
        field = ('float32', 'float32')
        dist = (0, 0)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is True

    def test_single_tensorlist(self):
        field = (('a', 'b', 'c'),)
        dist = (3,)
        assert TestcaseAclnn._is_field_already_nested(field, dist) is True


class TestNormalizeSkipsAlreadyNested:
    """Tests that normalize skips fields already matching dist structure."""

    def test_fully_nested_dtypes_not_double_expanded(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=(('float32', 'float32'), 'float32'))
        case._normalize_compressed_fields()
        assert case.tensor_dtypes == (('float32', 'float32'), 'float32')
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'float32')

    def test_single_tensorlist_fully_nested(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5), (4, 1)),),
            tensor_dtypes=(('float32', 'float32', 'float32'),))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'float32')

    def test_fully_nested_ranges_not_double_expanded(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))


class TestFlatPropertiesAfterNormalize:
    """Tests for flat_* properties after normalize — branch priority guards."""

    def test_compressed_dtypes_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5), (4, 1)),),
            tensor_dtypes=('float32',))
        case._normalize_compressed_fields()
        flat = case.flat_tensor_dtypes
        assert flat == ('float32', 'float32', 'float32')
        assert all(isinstance(x, str) for x in flat)

    def test_compressed_formats_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5), (4, 1)),),
            tensor_formats=('ND',))
        case._normalize_compressed_fields()
        flat = case.flat_tensor_formats
        assert flat == ('ND', 'ND', 'ND')
        assert all(isinstance(x, str) for x in flat)

    def test_compressed_ranges_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5), (4, 1)),),
            input_data_ranges=((-1.0, 1.0),))
        case._normalize_compressed_fields()
        flat = case.flat_input_data_ranges
        assert flat == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in flat)

    def test_per_param_dtypes_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5)), (4, 5)),
            tensor_dtypes=('float32', 'int8'))
        case._normalize_compressed_fields()
        flat = case.flat_tensor_dtypes
        assert flat == ('float32', 'float32', 'int8')
        assert all(isinstance(x, str) for x in flat)

    def test_per_param_ranges_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((2, 3), (3, 5)), (4, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._normalize_compressed_fields()
        flat = case.flat_input_data_ranges
        assert flat == ((None, 1.0), (None, 1.0), (-1.0, 1.0))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in flat)


class TestNormalizeInputDataRanges:
    """Tests for _normalize_input_data_ranges: TensorList broadcast/pad/truncate."""

    def _make_case(self, shapes, ranges=None):
        case = TestcaseAclnn()
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = ('float32',) * len(flatten_nested_sequence(shapes))
        if ranges is not None:
            case.input_data_ranges = ranges
        case._normalize_compressed_fields()
        return case

    def test_empty_ranges_unchanged(self):
        case = TestcaseAclnn()
        case.tensor_view_shapes = ((3, 4), (5, 6))
        case.tensor_dtypes = ('float32', 'float32')
        case.input_data_ranges = None
        case._normalize_compressed_fields()
        assert case.input_data_ranges is None

    def test_single_range_broadcast_to_all_params(self):
        case = self._make_case(
            shapes=((3, 4), ((5, 6), (7, 8))),
            ranges=((0.0, 1.0),))
        assert case.input_data_ranges == ((0.0, 1.0), ((0.0, 1.0), (0.0, 1.0)))

    def test_missing_params_padded_with_none(self):
        case = self._make_case(
            shapes=((3, 4), ((5, 6), (7, 8)), (9, 10)),
            ranges=((0.0, 1.0), (-1.0, 1.0)))
        # 2 ranges provided, 3 params → last padded with (None, None)
        assert case.input_data_ranges == (
            (0.0, 1.0), ((-1.0, 1.0), (-1.0, 1.0)), (None, None))

    def test_tensorlist_single_range_broadcast(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6)), (7, 8)),
            ranges=((None, 1.0), (-1.0, 1.0)))
        # TensorList(2) with single range → broadcast 2 copies
        assert case.input_data_ranges == (
            ((None, 1.0), (None, 1.0)), (-1.0, 1.0))

    def test_tensorlist_range_list_exact_match(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6)), (7, 8)),
            ranges=(((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0)))
        assert case.input_data_ranges == (
            ((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0))

    def test_tensorlist_range_list_short_padded(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6), (7, 8)), (9, 10)),
            ranges=(((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0)))
        # TensorList(3) but only 2 ranges → pad (None, None)
        assert case.input_data_ranges == (
            ((0.0, 0.5), (0.5, 1.0), (None, None)), (-1.0, 1.0))

    def test_tensorlist_range_list_extra_truncated(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6)), (7, 8)),
            ranges=(((0.0, 0.5), (0.5, 1.0), (1.0, 2.0)), (-1.0, 1.0)))
        # TensorList(2) but 3 ranges → truncate to 2
        assert case.input_data_ranges == (
            ((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0))

    def test_tensorlist_single_element_list_broadcast(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6), (7, 8)), (9, 10)),
            ranges=(((0.0, 1.0),), (-1.0, 1.0)))
        # (r,) → broadcast to 3 copies
        assert case.input_data_ranges == (
            ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)), (-1.0, 1.0))

    def test_lstm_scenario_all_none(self):
        """LSTM: Tensor + TensorList(2) + TensorList(4) with no ranges."""
        case = self._make_case(
            shapes=((3, 2, 4), ((1, 2, 4), (1, 2, 4)),
                    ((16, 4), (16, 4), (16,), (16,))),
            ranges=None)
        assert case.input_data_ranges is None

    def test_already_nested_unchanged(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6)), (7, 8)),
            ranges=(((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0)))
        # Already properly nested → no change
        assert case.input_data_ranges == (
            ((0.0, 0.5), (0.5, 1.0)), (-1.0, 1.0))


class TestNonePlaceholderInStrideOffsetStorage:
    """Tests for None placeholder in tensor_view_strides, tensor_view_offsets, tensor_storage_shapes.

    When a field value is None, the accessor derives from tensor_view_shapes (contiguous).
    """

    def _make_case(self, strides=None, offsets=None, storage_shapes=None):
        case = TestcaseAclnn()
        case.tensor_view_shapes = ((3, 4), (5, 6), (7, 8))
        case.tensor_dtypes = ('float32', 'float32', 'float32')
        if strides is not None:
            case.tensor_view_strides = strides
        if offsets is not None:
            case.tensor_view_offsets = offsets
        if storage_shapes is not None:
            case.tensor_storage_shapes = storage_shapes
        return case

    def test_stride_none_falls_back_to_contiguous(self):
        case = self._make_case(strides=(None, (12, 2), None))
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_stride(1) == (12, 2)
        assert case.flat_view_stride(2) == (8, 1)

    def test_offset_none_falls_back_to_zero(self):
        case = self._make_case(offsets=(None, 10, None))
        assert case.flat_view_offset(0) == 0
        assert case.flat_view_offset(1) == 10
        assert case.flat_view_offset(2) == 0

    def test_storage_shape_none_falls_back_to_view(self):
        case = self._make_case(storage_shapes=((6, 8), None, (10, 12)))
        assert case.flat_storage_shape(0) == (6, 8)
        assert case.flat_storage_shape(1) == (5, 6)
        assert case.flat_storage_shape(2) == (10, 12)

    def test_all_none_uses_defaults(self):
        case = self._make_case(
            strides=(None, None, None),
            offsets=(None, None, None),
            storage_shapes=(None, None, None))
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_offset(1) == 0
        assert case.flat_storage_shape(2) == (7, 8)

    def test_no_fields_set_defaults(self):
        case = self._make_case()
        from ttk.utilities.container_utils import shape_stride
        for i in range(3):
            assert case.flat_view_stride(i) == shape_stride(case.flat_tensor_view_shapes[i])
            assert case.flat_view_offset(i) == 0
            assert case.flat_storage_shape(i) == case.flat_tensor_view_shapes[i]

    def test_mixed_none_and_values(self):
        case = self._make_case(
            strides=(None, (12, 2), None),
            offsets=(5, None, 20),
            storage_shapes=((4, 6), None, None))
        assert case.flat_view_stride(0) == (4, 1)
        assert case.flat_view_stride(1) == (12, 2)
        assert case.flat_view_stride(2) == (8, 1)
        assert case.flat_view_offset(0) == 5
        assert case.flat_view_offset(1) == 0
        assert case.flat_view_offset(2) == 20
        assert case.flat_storage_shape(0) == (4, 6)
        assert case.flat_storage_shape(1) == (5, 6)
        assert case.flat_storage_shape(2) == (7, 8)


class TestFlatInputDataRangesPadding:
    """Tests for flat_input_data_ranges: flatten/broadcast, then pad (None,None) at end if short."""

    def test_short_by_one_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        assert case.pure_output_indexes == [2]
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (None, None))

    def test_short_broadcast_last_ignores_pure_output_position(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(1,))
        assert case.pure_output_indexes == [1]
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (None, None))

    def test_short_by_two_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (3, 4), (4, 5), (5, 6)),
            input_data_ranges=((None, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(1, 3))
        assert case.pure_output_indexes == [1, 3]
        assert case.flat_input_data_ranges == ((None, 1.0), (0.0, 5.0), (None, None), (None, None))

    def test_short_pads_none_with_pure_output_at_start(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((-1.0, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(0,))
        assert case.pure_output_indexes == [0]
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (0.0, 5.0), (None, None))

    def test_exact_count_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(2,))
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_no_short_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0))

    def test_broadcast_fills_all_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((-1.0, 1.0),),
            output_tensor_indexes=(2,))
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_broadcast_fills_all_even_with_gap(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (3, 4), (4, 5), (5, 6)),
            input_data_ranges=((None, 1.0),),
            output_tensor_indexes=())
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (None, 1.0), (None, 1.0))

    def test_expand_by_dist_fills_all_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(1,))
        # Must normalize first — flat_input_data_ranges assumes normalized structure.
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_expand_by_dist_short_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0), (None, None))

    def test_nested_ranges_flatten_short_then_pad(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=(((0.0, 5.0), (1.0, 2.0)),),
            output_tensor_indexes=(2,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        # Single range-list broadcast: TensorList(2) gets exact match,
        # non-TensorList params also get the broadcast range-list as-is.
        assert case.flat_input_data_ranges == ((0.0, 5.0), (1.0, 2.0), ((0.0, 5.0), (1.0, 2.0)), ((0.0, 5.0), (1.0, 2.0)))


class TestNormalizeInputDataRangesPadding:
    """Tests for _normalize_input_data_ranges: expand, pad at end if short, then re-nest."""

    def test_normalize_short_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (None, None))

    def test_normalize_broadcast_fills_all(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5)),
            input_data_ranges=((0.0, 5.0),),
            output_tensor_indexes=(0,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.flat_input_data_ranges == ((0.0, 5.0), (0.0, 5.0), (0.0, 5.0))

    def test_normalize_expand_by_dist_fills_all(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2), (3, 4)), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(1,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.input_data_ranges == (((None, 1.0), (None, 1.0), (None, 1.0)), (-1.0, 1.0))
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_normalize_expand_by_dist_short_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        dist = case._get_tensor_list_distribution()
        case._normalize_input_data_ranges(dist)
        assert case.input_data_ranges == (((None, 1.0), (None, 1.0)), (-1.0, 1.0), (None, None))
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0), (None, None))


# ========== TensorList flattening edge-case tests ==========

# Two TensorList params, each with 1 tensor inside.
_TL2_SHAPES = (((3, 2, 4),), ((3, 2, 4),))


class TestTensorListFlatDtypes:
    """flat_tensor_dtypes must correctly flatten nested TensorList dtypes."""

    def test_nested_tuple_of_str(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_dtypes=(('float32',), ('float32',)))
        flat = case.flat_tensor_dtypes
        assert flat == ('float32', 'float32')
        assert all(isinstance(d, str) for d in flat)

    def test_flat_passthrough(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (4, 5)),
            tensor_dtypes=('float32', 'float16'))
        assert case.flat_tensor_dtypes == ('float32', 'float16')


class TestTensorListFlatFormats:
    """flat_tensor_formats must correctly flatten nested TensorList formats."""

    def test_nested_tuple_of_str(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_formats=(('ND',), ('ND',)))
        flat = case.flat_tensor_formats
        assert flat == ('ND', 'ND')
        assert all(isinstance(f, str) for f in flat)


class TestTensorListFlatOffsets:
    """flat_tensor_view_offsets must correctly flatten nested TensorList offsets."""

    def test_nested_tuple_of_int(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_view_offsets=((0,), (0,)))
        flat = case.flat_tensor_view_offsets
        assert flat == (0, 0)
        assert all(isinstance(o, int) for o in flat)

    def test_mixed_nested_and_flat(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            tensor_view_offsets=((0, 10), 5))
        assert case.flat_tensor_view_offsets == (0, 10, 5)


class TestTensorListFlatStrides:
    """flat_tensor_view_strides must use _flatten_by_distribution for shape-tuple leaves."""

    def test_nested_tuple_of_shape_tuples(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_view_strides=(((1, 2, 3),), ((4, 5, 6),)))
        flat = case.flat_tensor_view_strides
        assert flat == ((1, 2, 3), (4, 5, 6))
        assert all(isinstance(s, tuple) for s in flat)

    def test_compressed_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            tensor_view_strides=((4, 1),))
        case._normalize_compressed_fields()
        assert case.flat_tensor_view_strides == ((4, 1), (4, 1), (4, 1))


class TestTensorListFlatStorageShapes:
    """flat_tensor_storage_shapes must use _flatten_by_distribution for shape-tuple leaves."""

    def test_nested_tuple_of_shape_tuples(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=_TL2_SHAPES,
            tensor_storage_shapes=(((3, 2, 4),), ((3, 2, 4),)))
        flat = case.flat_tensor_storage_shapes
        assert flat == ((3, 2, 4), (3, 2, 4))
        assert all(isinstance(s, tuple) for s in flat)

    def test_compressed_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            tensor_storage_shapes=((10, 10),))
        case._normalize_compressed_fields()
        assert case.flat_tensor_storage_shapes == ((10, 10), (10, 10), (10, 10))
