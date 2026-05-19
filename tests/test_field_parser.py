#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.testcase_manager.field_parser: nested parsers.
"""

import pytest
from ttk.core_modules.testcase_manager.field_parser import (
    shapelike_stc_nested,
    shapelike_float_signed_nested,
    scalar_nested,
)
from functools import partial

_int_container_nested = partial(scalar_nested, allowed_type=int)


class TestShapelikeStcNested:
    """Tests for shapelike_stc_nested parser."""

    def test_flat_two_tensors(self):
        result = shapelike_stc_nested("((3,3),(3,5))")
        assert result == ((3, 3), (3, 5))

    def test_tensor_list_plus_tensor(self):
        result = shapelike_stc_nested("(((3,3),(3,2)),(3,5))")
        assert result == (((3, 3), (3, 2)), (3, 5))

    def test_single_tensor(self):
        result = shapelike_stc_nested("((3,5),)")
        assert result == ((3, 5),)

    def test_with_none(self):
        result = shapelike_stc_nested("(((3,3),(3,2)),None,(3,5))")
        assert result == (((3, 3), (3, 2)), None, (3, 5))

    def test_empty_tuple(self):
        result = shapelike_stc_nested("()")
        assert result == ()

    def test_scalar_shape(self):
        result = shapelike_stc_nested("((3,),)")
        assert result == ((3,),)

    def test_1d_shape(self):
        result = shapelike_stc_nested("((5,),)")
        assert result == ((5,),)

    def test_invalid_shape_float(self):
        with pytest.raises(TypeError):
            shapelike_stc_nested("((3.5,),)")

    def test_invalid_shape_string(self):
        with pytest.raises(TypeError):
            shapelike_stc_nested("('abc',)")


class TestScalarNested:
    """Tests for scalar_nested parser."""

    def test_flat_dtypes(self):
        result = scalar_nested("('float32','float32')")
        assert result == ("float32", "float32")

    def test_nested_dtypes(self):
        result = scalar_nested("(('float32','float32'),'float32')")
        assert result == (("float32", "float32"), "float32")

    def test_single_dtype(self):
        result = scalar_nested("('float32',)")
        assert result == ("float32",)

    def test_with_none(self):
        result = scalar_nested("(('float32','float32'),None)")
        assert result == (("float32", "float32"), None)

    def test_formats(self):
        result = scalar_nested("('ND','ND','ND')")
        assert result == ("ND", "ND", "ND")

    def test_compressed_tensor_list_format(self):
        result = scalar_nested("(('ND',),'ND')")
        assert result == (("ND",), "ND")

    def test_float_tuple_no_double_wrap(self):
        result = scalar_nested("(1e-08, 1e-08, 1e-08)")
        assert result == (1e-08, 1e-08, 1e-08)
        for val in result:
            assert isinstance(val, float)

    def test_single_float(self):
        result = scalar_nested("1e-08")
        assert result == (1e-08,)


class TestIntContainerNested:
    """Tests for int_container_nested (scalar_nested with allowed_type=int)."""

    def test_flat_offsets(self):
        result = _int_container_nested("(0, 1)")
        assert result == (0, 1)

    def test_nested_offsets(self):
        result = _int_container_nested("((0, 1), 2)")
        assert result == ((0, 1), 2)

    def test_single_value(self):
        result = _int_container_nested("0")
        assert result == (0,)

    def test_empty(self):
        result = _int_container_nested("")
        assert result == ()

    def test_with_none(self):
        result = _int_container_nested("(0, None, 2)")
        assert result == (0, None, 2)


class TestShapelikeFloatSignedNested:
    """Tests for shapelike_float_signed_nested parser."""

    def test_flat_ranges(self):
        result = shapelike_float_signed_nested("((None, 1.0), (-1.0, 1.0))")
        assert result == ((None, 1.0), (-1.0, 1.0))

    def test_nested_tensor_list(self):
        result = shapelike_float_signed_nested("(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))")
        assert result == (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))

    def test_single_range(self):
        result = shapelike_float_signed_nested("((-1.0, 1.0),)")
        assert result == ((-1.0, 1.0),)

    def test_with_none_element(self):
        result = shapelike_float_signed_nested("(((0.0, 1.0), None), (0.0, 1.0))")
        assert result == (((0.0, 1.0), None), (0.0, 1.0))

    def test_empty_tuple(self):
        result = shapelike_float_signed_nested("()")
        assert result == ()

    def test_int_values(self):
        result = shapelike_float_signed_nested("((0, 1),)")
        assert result == ((0, 1),)

    def test_compressed_top_level(self):
        result = shapelike_float_signed_nested("((-1.0, 1.0),)")
        assert result == ((-1.0, 1.0),)

    def test_invalid_string_value(self):
        with pytest.raises(TypeError):
            shapelike_float_signed_nested("('abc',)")

    def test_invalid_nested_string(self):
        with pytest.raises(TypeError):
            shapelike_float_signed_nested("(('abc',),)")
