#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.testcase_manager.field_parser: nested parsers.
"""

from functools import partial

import pytest

from ttk.core_modules.testcase_manager.field_parser import (
    scalar_nested,
    shapelike_float_signed_nested,
    shapelike_stc_nested,
)

_int_container_nested = partial(scalar_nested, allowed_type=int)


class TestShapelikeStcNested:
    """Tests for shapelike_stc_nested parser.

    分两组：
    - test_parse: 输入字符串 → 期望解析结果
    - test_parse_raises: 非法 shape（float / string）→ TypeError
    """

    @pytest.mark.parametrize("text, expected", [
        ("((3,3),(3,5))", ((3, 3), (3, 5))),
        ("(((3,3),(3,2)),(3,5))", (((3, 3), (3, 2)), (3, 5))),
        ("((3,5),)", ((3, 5),)),
        ("(((3,3),(3,2)),None,(3,5))", (((3, 3), (3, 2)), None, (3, 5))),
        ("()", ()),
        ("((3,),)", ((3,),)),
        ("((5,),)", ((5,),)),
    ], ids=["flat-two-tensors", "tensor-list-plus-tensor", "single-tensor",
            "with-none", "empty-tuple", "scalar-shape", "1d-shape"])
    def test_parse(self, text, expected):
        """验证 shapelike_stc_nested 对各类合法输入的解析结果。"""
        assert shapelike_stc_nested(text) == expected

    @pytest.mark.parametrize("text", [
        "((3.5,),)",
        "('abc',)",
    ], ids=["invalid-shape-float", "invalid-shape-string"])
    def test_parse_raises(self, text):
        """验证非法 shape（float / string）→ TypeError。"""
        with pytest.raises(TypeError):
            shapelike_stc_nested(text)


class TestScalarNested:
    """Tests for scalar_nested parser.

    每行参数：输入字符串与期望解析结果。
    """

    @pytest.mark.parametrize("text, expected", [
        ("('float32','float32')", ("float32", "float32")),
        ("(('float32','float32'),'float32')", (("float32", "float32"), "float32")),
        ("('float32',)", ("float32",)),
        ("(('float32','float32'),None)", (("float32", "float32"), None)),
        ("('ND','ND','ND')", ("ND", "ND", "ND")),
        ("(('ND',),'ND')", (("ND",), "ND")),
        ("(1e-08, 1e-08, 1e-08)", (1e-08, 1e-08, 1e-08)),
        ("1e-08", (1e-08,)),
    ], ids=["flat-dtypes", "nested-dtypes", "single-dtype", "with-none",
            "formats", "compressed-tensor-list-format",
            "float-tuple-no-double-wrap", "single-float"])
    def test_parse(self, text, expected):
        """验证 scalar_nested 对各类输入的解析结果。"""
        result = scalar_nested(text)
        assert result == expected
        # float tuple 不应被二次包装为嵌套
        if text == "(1e-08, 1e-08, 1e-08)":
            for val in result:
                assert isinstance(val, float)


class TestIntContainerNested:
    """Tests for int_container_nested (scalar_nested with allowed_type=int).

    每行参数：输入字符串与期望解析结果。
    """

    @pytest.mark.parametrize("text, expected", [
        ("(0, 1)", (0, 1)),
        ("((0, 1), 2)", ((0, 1), 2)),
        ("0", (0,)),
        ("", ()),
        ("(0, None, 2)", (0, None, 2)),
    ], ids=["flat-offsets", "nested-offsets", "single-value",
            "empty", "with-none"])
    def test_parse(self, text, expected):
        """验证 int_container_nested 对各类输入的解析结果。"""
        assert _int_container_nested(text) == expected


class TestShapelikeFloatSignedNested:
    """Tests for shapelike_float_signed_nested parser.

    分两组：
    - test_parse: 输入字符串 → 期望解析结果
    - test_parse_raises: 非法字符串值 → TypeError
    """

    @pytest.mark.parametrize("text, expected", [
        ("((None, 1.0), (-1.0, 1.0))", ((None, 1.0), (-1.0, 1.0))),
        ("(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))",
         (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))),
        ("((-1.0, 1.0),)", ((-1.0, 1.0),)),
        ("(((0.0, 1.0), None), (0.0, 1.0))",
         (((0.0, 1.0), None), (0.0, 1.0))),
        ("()", ()),
        ("((0, 1),)", ((0, 1),)),
        ("((-1.0, 1.0),)", ((-1.0, 1.0),)),
    ], ids=["flat-ranges", "nested-tensor-list", "single-range",
            "with-none-element", "empty-tuple", "int-values",
            "compressed-top-level"])
    def test_parse(self, text, expected):
        """验证 shapelike_float_signed_nested 对各类合法输入的解析结果。"""
        assert shapelike_float_signed_nested(text) == expected

    @pytest.mark.parametrize("text", [
        "('abc',)",
        "(('abc',),)",
    ], ids=["invalid-string-value", "invalid-nested-string"])
    def test_parse_raises(self, text):
        """验证非法字符串值 → TypeError。"""
        with pytest.raises(TypeError):
            shapelike_float_signed_nested(text)
