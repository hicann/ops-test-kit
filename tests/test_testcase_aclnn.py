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

from collections import OrderedDict

import pytest
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfo
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
        assert case.tensor_list_dist == (0, 0)

    def test_nested_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES)
        assert case.tensor_list_dist == (2, 0)

    def test_all_nested(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(((1,), (2,)), ((3,),)))
        assert case.tensor_list_dist == (2, 1)

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None)
        assert case.tensor_list_dist == ()


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
        """Flat input (len==flat_count > len(dist)) → invalid."""
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0), (0.0, 5.0)))
        case._normalize_compressed_fields()
        assert case.is_valid is False

    def test_top_level_compression(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((-1.0, 1.0),))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_per_param_compression(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

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
        case._scalar_list_dist = (2, 0)
        case._normalize_compressed_fields()
        assert case.flat_scalar_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_per_param_compression(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=(('int64', 'int64'), 'float32'),
            scalar_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._scalar_list_dist = (2, 0)
        case._normalize_compressed_fields()
        assert case.flat_scalar_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_nested_explicit(self, make_testcase):
        case = make_testcase(
            scalar_dtypes=(('int64', 'int64'), 'float32'),
            scalar_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)))
        case._scalar_list_dist = (2, 0)
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
    """Tests for already-nested detection via _normalize_field_by_dist with _is_scalar_group."""

    def _assert_already_nested(self, field, dist):
        """Verify scalar field is detected as already-nested (no modification)."""
        case = TestcaseAclnn()
        case.is_valid = True
        case.tensor_dtypes = field
        case._normalize_field_by_dist("tensor_dtypes", dist, TestcaseAclnn._is_scalar_group)
        assert case.is_valid is True
        assert case.tensor_dtypes == field

    def _assert_not_already_nested(self, field, dist):
        """Verify scalar field is NOT already-nested (gets normalized or rejected)."""
        case = TestcaseAclnn()
        case.is_valid = True
        case.tensor_dtypes = field
        case._normalize_field_by_dist("tensor_dtypes", dist, TestcaseAclnn._is_scalar_group)
        assert case.tensor_dtypes != field or case.is_valid is False

    def test_fully_nested_matches_dist(self):
        self._assert_already_nested((('float32', 'float32'), 'float32'), (2, 0))

    def test_compressed_not_nested(self):
        self._assert_not_already_nested(('float32',), (2, 0))

    def test_len_mismatch(self):
        self._assert_not_already_nested(('float32', 'float32'), (3,))

    def test_inner_len_mismatch(self):
        self._assert_not_already_nested((('float32',), 'float32'), (2, 0))

    def test_all_single_tensors(self):
        self._assert_already_nested(('float32', 'float32'), (0, 0))

    def test_single_tensorlist(self):
        self._assert_already_nested((('a', 'b', 'c'),), (3,))


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
        case.tensor_dtypes = tuple(
            tuple('float32' for _ in s) if isinstance(s, (tuple, list)) and s and isinstance(s[0], (tuple, list)) else 'float32'
            for s in shapes
        )
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
        # TensorList(3) but range-list has only 2 entries — ambiguous, mark invalid.
        assert case.is_valid is False

    def test_tensorlist_range_list_extra_truncated(self):
        case = self._make_case(
            shapes=(((3, 4), (5, 6)), (7, 8)),
            ranges=(((0.0, 0.5), (0.5, 1.0), (1.0, 2.0)), (-1.0, 1.0)))
        # TensorList(2) but range-list has 3 entries — ambiguous, mark invalid.
        assert case.is_valid is False

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
    """Tests for flat_input_data_ranges: normalize then flatten — pad (None,None) at end if short."""

    def test_short_by_one_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        case._normalize_compressed_fields()
        assert case.pure_output_indexes == [2]
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (None, None))

    def test_short_broadcast_last_ignores_pure_output_position(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(1,))
        case._normalize_compressed_fields()
        assert case.pure_output_indexes == [1]
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (None, None))

    def test_short_by_two_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (3, 4), (4, 5), (5, 6)),
            input_data_ranges=((None, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(1, 3))
        case._normalize_compressed_fields()
        assert case.pure_output_indexes == [1, 3]
        assert case.flat_input_data_ranges == ((None, 1.0), (0.0, 5.0), (None, None), (None, None))

    def test_short_pads_none_with_pure_output_at_start(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((-1.0, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(0,))
        case._normalize_compressed_fields()
        assert case.pure_output_indexes == [0]
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (0.0, 5.0), (None, None))

    def test_exact_count_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),
            output_tensor_indexes=(2,))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_no_short_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0))

    def test_broadcast_fills_all_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 3), (3, 2), (3, 5)),
            input_data_ranges=((-1.0, 1.0),),
            output_tensor_indexes=(2,))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_broadcast_fills_all_even_with_gap(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (3, 4), (4, 5), (5, 6)),
            input_data_ranges=((None, 1.0),),
            output_tensor_indexes=())
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (None, 1.0), (None, 1.0))

    def test_expand_by_dist_fills_all_no_padding(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(1,))
        # Must normalize first — flat_input_data_ranges assumes normalized structure.
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_expand_by_dist_short_pads_none(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=((None, 1.0), (-1.0, 1.0)),
            output_tensor_indexes=(2,))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0), (None, None))

    def test_nested_ranges_single_expr_broadcast_and_pad(self, make_testcase):
        """len==1 with single range expression: broadcast to all positions, then
        TensorList internal broadcast, missing positions padded with pad_value."""
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=((0.0, 5.0),),
            output_tensor_indexes=(2,))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        # len==1: single range expression (0.0,5.0) broadcast to all 3 positions,
        # then TensorList position 0 internally broadcasts to 2 copies.
        assert case.flat_input_data_ranges == ((0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0))

    def test_nested_ranges_single_range_list_invalid(self, make_testcase):
        """len==1 with range-list value: cannot broadcast, mark invalid."""
        case = make_testcase(
            tensor_view_shapes=(((3, 3), (3, 2)), (3, 5), (4, 5)),
            input_data_ranges=(((0.0, 5.0), (1.0, 2.0)),),
            output_tensor_indexes=(2,))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False


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


# ========== Comprehensive tests for _normalize_range_field_by_dist paths ==========


class TestIsRangeFieldAlreadyNested:
    """Tests for already-nested detection via _normalize_field_by_dist with _is_range_group."""

    def _assert_already_nested(self, field, dist):
        """Verify range field is detected as already-nested (no modification)."""
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", dist, TestcaseAclnn._is_range_group)
        assert case.is_valid is True
        assert case.input_data_ranges == field

    def _assert_not_already_nested(self, field, dist):
        """Verify range field is NOT already-nested (gets normalized or rejected)."""
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = field
        case._normalize_field_by_dist("input_data_ranges", dist, TestcaseAclnn._is_range_group)
        assert case.input_data_ranges != field or case.is_valid is False

    def test_fully_nested_range_list(self):
        self._assert_already_nested(
            (((0.0, 1.0), (-1.0, 1.0)), (0.5, 2.0)), (2, 0))

    def test_len_mismatch_returns_false(self):
        self._assert_not_already_nested(
            ((0.0, 1.0), (-1.0, 1.0)), (2, 0))

    def test_single_tensorlist_fully_nested(self):
        self._assert_already_nested(
            (((0.0, 1.0), (-1.0, 1.0), (0.5, 2.0)),), (3,))

    def test_single_element_tensorlist_nested(self):
        self._assert_already_nested(
            (((0.0, 1.0),),), (1,))

    def test_inner_len_mismatch_returns_false(self):
        self._assert_not_already_nested(
            (((0.0, 1.0),), (0.5, 2.0)), (2, 0))

    def test_flat_values_in_tensorlist_position_returns_false(self):
        # (0.0, 1.0) is a range expression, not a range list → not already nested
        self._assert_not_already_nested(
            ((0.0, 1.0), (-1.0, 1.0)), (2,))

    def test_all_single_tensors_returns_true(self):
        self._assert_already_nested(
            ((0.0, 1.0), (-1.0, 1.0)), (0, 0))

    def test_empty_field_returns_false(self):
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = ()
        case._normalize_field_by_dist("input_data_ranges", (0,), TestcaseAclnn._is_range_group)
        # Empty field is a no-op (early return), not a True/False result
        assert case.input_data_ranges == ()

    def test_none_field_returns_false(self):
        case = TestcaseAclnn()
        case.is_valid = True
        case.input_data_ranges = None
        case._normalize_field_by_dist("input_data_ranges", (0,), TestcaseAclnn._is_range_group)
        # None field is a no-op (early return)
        assert case.input_data_ranges is None


class TestResolvePadValue:
    """Direct unit tests for _resolve_pad_value class method."""

    def test_explicit_pad_overrides_default(self):
        # explicit_pad takes precedence
        assert TestcaseAclnn._resolve_pad_value("input_data_ranges", explicit_pad=(0, 0)) == (0, 0)

    def test_auto_resolve_from_header_none_default(self):
        # input_data_ranges default is ((None, None),) → pad is (None, None)
        result = TestcaseAclnn._resolve_pad_value("input_data_ranges")
        assert result == (None, None)

    def test_auto_resolve_from_header_scalar_default(self):
        # absolute_precision default is 1e-8 (scalar float) → pad is 1e-8
        result = TestcaseAclnn._resolve_pad_value("absolute_precision")
        assert result == 1e-8

    def test_auto_resolve_unknown_field_returns_no_pad(self):
        # field not in complete_headers → _NO_PAD
        from ttk.core_modules.testcase_manager.testcase_base import _NO_PAD
        result = TestcaseAclnn._resolve_pad_value("nonexistent_field")
        assert result is _NO_PAD


class TestWriteBackNormalized:
    """Tests for _write_back_normalized: field write-back and flat cache clearing."""

    def test_writes_back_as_tuple(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=((0.0, 1.0),))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert isinstance(case.input_data_ranges, tuple)

    def test_clears_flat_cache(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        # Manually populate cache (simulating prior access after normalize)
        case._flat_input_data_ranges = ((0.0, 1.0), (-1.0, 1.0))
        # Normalize should clear cache via _write_back_normalized
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case._flat_input_data_ranges is None


class TestNormalizeRangeFieldNonTupleField:
    """Tests for non-tuple/non-list field input → early return."""

    def test_non_tuple_field_unchanged(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 4), (5, 6)),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        # Set to a non-tuple value
        case.input_data_ranges = "not_a_tuple"
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == "not_a_tuple"

    def test_empty_tuple_field_unchanged(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 4), (5, 6)),
            input_data_ranges=())
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == ()

    def test_none_field_unchanged(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((3, 4), (5, 6)),
            input_data_ranges=None)
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges is None


class TestNormalizeRangeFieldLenGtLenDist:
    """len(field) > len(dist) → mark invalid (CASE_FIELD_AMBIGUOUS)."""

    def test_flat_count_eq_len_dist_invalid(self, make_testcase):
        # dist=(2,0), len(dist)=2, len(field)=3 > len(dist) → invalid
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)))
        dist = case.tensor_list_dist
        assert dist == (2, 0)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_len_gt_flat_count_invalid(self, make_testcase):
        # dist=(2,0), len(dist)=2, len(field)=4 > len(dist) → invalid
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=((0, 1), (2, 3), (4, 5), (6, 7)))
        dist = case.tensor_list_dist
        assert dist == (2, 0)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_all_tensorlist_flat_count_invalid(self, make_testcase):
        # dist=(2,3), len(dist)=2, len(field)=5 > len(dist) → invalid
        case = make_testcase(
            tensor_view_shapes=(((1,), (2,)), ((3,), (4,), (5,))),
            input_data_ranges=((0, 1), (2, 3), (4, 5), (6, 7), (8, 9)))
        dist = case.tensor_list_dist
        assert dist == (2, 3)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"


class TestNormalizeRangeFieldLenEqLenDistNotNested:
    """len(field)==len(dist) but not nested → per-param expand path.

    Range expressions at TensorList positions get broadcast internally.
    """

    def test_range_expr_at_tensorlist_position(self, make_testcase):
        # dist=(2,), len(field)=1==len(dist)=1
        # Single range expr → broadcast to all, then TensorList expand
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)),),
            input_data_ranges=((0.0, 1.0),))
        dist = case.tensor_list_dist
        assert dist == (2,)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == (((0.0, 1.0), (0.0, 1.0)),)

    def test_len_eq_dist_with_range_expr_at_tensorlist(self, make_testcase):
        # dist=(2,0), len(field)=2==len(dist)=2
        # field[0] is range expression (not range list) → per-param expand
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        assert dist == (2, 0)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        # Position 0 (TensorList(2)): single range expr → broadcast to 2 copies
        # Position 1 (single): keep as-is
        assert case.input_data_ranges == (((0.0, 1.0), (0.0, 1.0)), (-1.0, 1.0))


class TestNormalizeRangeFieldPadValueExplicit:
    """Tests for explicit pad_value overriding header default."""

    def test_explicit_pad_value_used(self, make_testcase):
        # dist=(0,0,0), len(field)=2 → pad 3rd position with explicit value
        case = make_testcase(
            tensor_view_shapes=((3, 4), (5, 6), (7, 8)),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        assert dist == (0, 0, 0)
        case._normalize_range_field_by_dist("input_data_ranges", dist, pad_value=(5.0, 10.0))
        assert case.input_data_ranges == ((0.0, 1.0), (-1.0, 1.0), (5.0, 10.0))

    def test_auto_resolve_pad_from_header_default(self, make_testcase):
        # No explicit pad_value, 2 ranges for 3 params → pad 3rd with auto-resolved default
        case = make_testcase(
            tensor_view_shapes=((3, 4), (5, 6), (7, 8)),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        # auto-resolved pad for input_data_ranges is (None, None)
        assert case.input_data_ranges == ((0.0, 1.0), (-1.0, 1.0), (None, None))


class TestNormalizeRangeFieldMixedTensorList:
    """Tests for mixed TensorList + single tensor scenarios."""

    def test_all_tensorlist_positions(self, make_testcase):
        # dist=(2,3): two TensorList params
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), ((7, 8), (9, 10), (11, 12))),
            input_data_ranges=((0.0, 1.0), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        assert dist == (2, 3)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == (
            ((0.0, 1.0), (0.0, 1.0)),
            ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)))

    def test_partial_range_list_at_tensorlist_invalid(self, make_testcase):
        # dist=(2,0), len(field)=2==len(dist)=2
        # field[0] is range-list with len 3 != num 2 → AMBIGUOUS
        case = make_testcase(
            tensor_view_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_data_ranges=(((0.0, 1.0), (2.0, 3.0), (4.0, 5.0)), (-1.0, 1.0)))
        dist = case.tensor_list_dist
        assert dist == (2, 0)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.is_valid is False
        assert case.fail_reason == "CASE_FIELD_AMBIGUOUS"

    def test_single_tensor_no_tensorlist(self, make_testcase):
        # dist=(0,): single tensor, no TensorList
        case = make_testcase(
            tensor_view_shapes=((3, 4),),
            input_data_ranges=((0.0, 1.0),))
        dist = case.tensor_list_dist
        assert dist == (0,)
        case._normalize_range_field_by_dist("input_data_ranges", dist)
        assert case.input_data_ranges == ((0.0, 1.0),)


# ========== Tests for _auto_fill_output_tensor_indexes ==========

def _make_op_api_info(tensor_names, tensor_types=None):
    """Helper to create a mock OpApiInfo with given tensor param names."""
    if tensor_types is None:
        tensor_types = ["aclTensor*"] * len(tensor_names)
    params = OrderedDict()
    for name, typ in zip(tensor_names, tensor_types):
        params[name] = {"type": typ}
    info = OpApiInfo(params=params)
    return info


class TestAutoFillOutputTensorIndexes:
    """Tests for _auto_fill_output_tensor_indexes naming-convention-based detection."""

    def _make_case(self, api_name="aclnnDummy", tensor_names=None,
                   tensor_types=None, tensor_view_shapes=None):
        """Create a TestcaseAclnn with mocked OpApiInfoKeeper.info_of."""
        if tensor_names is None:
            tensor_names = ["self", "other", "out"]
        if tensor_view_shapes is None:
            tensor_view_shapes = tuple((2, 3) for _ in tensor_names)
        info = _make_op_api_info(tensor_names, tensor_types)
        case = TestcaseAclnn()
        case.api_name = api_name
        case.is_valid = True
        case.fail_reason = None
        case.tensor_view_shapes = tensor_view_shapes
        case.tensor_dtypes = tuple("float32" for _ in tensor_names)
        case.attributes = {}
        case.output_tensor_indexes = ()
        case.output_inplace_indexes = ()
        # Patch at module level where _auto_fill_output_tensor_indexes calls it
        self._patcher = patch(
            "ttk.core_modules.testcase_manager.testcase_aclnn.OpApiInfoKeeper")
        MockCls = self._patcher.start()
        MockCls.return_value.info_of.return_value = info
        return case

    def teardown_method(self):
        if hasattr(self, '_patcher'):
            self._patcher.stop()

    # --- Basic output detection by *Out suffix ---

    def test_out_suffix_detected(self):
        case = self._make_case(tensor_names=["self", "other", "yOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_output_suffix_detected(self):
        case = self._make_case(tensor_names=["input", "gradOutput"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1,)

    def test_output_literal_detected(self):
        case = self._make_case(tensor_names=["self", "other", "output"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_out_optional_detected(self):
        case = self._make_case(tensor_names=["input", "meanOutOptional", "rstdOutOptional"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1, 2)

    def test_output_optional_detected(self):
        case = self._make_case(tensor_names=["input", "activationFeatureOutputOptional"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1,)

    # --- Ref suffix (inplace) ---

    def test_ref_suffix_detected(self):
        case = self._make_case(tensor_names=["selfRef", "other"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (0,)

    def test_mixed_ref_and_out(self):
        case = self._make_case(tensor_names=["selfRef", "other", "yOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (0, 2)

    # --- Fallback to last tensor ---

    def test_fallback_no_matching_names(self):
        case = self._make_case(tensor_names=["self", "weight", "bias"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (-1,)

    # --- Already set by user ---

    def test_user_set_not_overridden(self):
        case = self._make_case(tensor_names=["self", "other", "yOut"])
        case.output_tensor_indexes = (0,)
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (0,)

    # --- Not valid ---

    def test_invalid_skipped(self):
        case = self._make_case(tensor_names=["self", "out"])
        case.is_valid = False
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == ()

    # --- Backward/Grad exclusions ---

    def test_backward_excludes_grad_output(self):
        case = self._make_case(
            api_name="aclnnSoftmaxBackward",
            tensor_names=["gradOutput", "output", "yOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_backward_excludes_grad_out(self):
        case = self._make_case(
            api_name="aclnnLayerNormBackward",
            tensor_names=["input", "gradOut", "gradInputOut", "gradWeightOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2, 3)

    def test_backward_excludes_grad_output_underscore(self):
        """grad_output doesn't match *Out/*Output suffix, excluded by name."""
        case = self._make_case(
            api_name="aclnnModulateBackward",
            tensor_names=["grad_output", "yOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1,)

    def test_backward_excludes_attention_out(self):
        case = self._make_case(
            api_name="aclnnNsaSelectedAttentionGrad",
            tensor_names=["query", "attentionOut", "dqOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_backward_excludes_d_out(self):
        case = self._make_case(
            api_name="aclnnSparseFlashAttentionGrad",
            tensor_names=["query", "dOut", "dqOut", "dkOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2, 3)

    def test_backward_output_not_last_excluded(self):
        case = self._make_case(
            api_name="aclnnLogSoftmaxBackward",
            tensor_names=["gradOutput", "output", "yOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_backward_output_as_last_included(self):
        case = self._make_case(
            api_name="aclnnAvgPool3dBackward",
            tensor_names=["gradOutput", "self", "output"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (2,)

    def test_grad_api_detected(self):
        case = self._make_case(
            api_name="aclnnFlashAttentionScoreGrad",
            tensor_names=["query", "dkOut", "dvOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1, 2)

    # --- Non-backward: no exclusions ---

    def test_non_backward_out_fallback(self):
        """'out' not matched by *Out/*Output, falls back to last tensor."""
        case = self._make_case(
            api_name="aclnnAdd",
            tensor_names=["self", "other", "out"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (-1,)

    def test_non_backward_all_out(self):
        case = self._make_case(
            api_name="aclnnAminmax",
            tensor_names=["self", "minOut", "maxOut"])
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1, 2)

    # --- None (nullptr) tensor handling ---

    def test_none_out_optional_in_output_indexes(self):
        case = self._make_case(
            tensor_names=["input", "resultOut", "reserveOutOptional"],
            tensor_view_shapes=((3, 4), (3, 4), None))
        case._auto_fill_output_tensor_indexes()
        assert case.output_tensor_indexes == (1, 2)

    # --- None (nullptr) inplace handling ---

    def test_none_ref_skipped_in_inplace(self):
        case = self._make_case(
            tensor_names=["varRef", "mRef", "maxGradNormOptionalRef", "grad"],
            tensor_view_shapes=((512,), (512,), None, (512,)))
        case._auto_fill_output_inplace_indices()
        assert case.output_inplace_indexes == (0, 1)
