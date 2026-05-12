#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.utilities.container_utils: nested tensor list utilities.
"""

import pytest
from ttk.utilities.container_utils import (
    infer_list_distribution_from_nesting,
    flatten_nested_sequence,
)


class TestInferListDistributionFromNesting:
    """Tests for infer_list_distribution_from_nesting."""

    def test_flat_tensors(self):
        assert infer_list_distribution_from_nesting(((3, 3), (3, 5))) == (0, 0)

    def test_tensor_list_plus_tensor(self):
        assert infer_list_distribution_from_nesting((((3, 3), (3, 2)), (3, 5))) == (2, 0)

    def test_all_tensor_lists(self):
        assert infer_list_distribution_from_nesting((((1,), (2,)), ((3,), (4,), (5,)))) == (2, 3)

    def test_with_none(self):
        assert infer_list_distribution_from_nesting((((3, 3), (3, 2)), None, (3, 5))) == (2, 0, 0)

    def test_single_tensor(self):
        assert infer_list_distribution_from_nesting(((3, 5),)) == (0,)

    def test_single_tensor_list(self):
        assert infer_list_distribution_from_nesting((((1,), (2,), (3,)),)) == (3,)

    def test_empty_tuple(self):
        assert infer_list_distribution_from_nesting(()) == ()

    def test_scalar_like_flat(self):
        assert infer_list_distribution_from_nesting(("float32", "int64")) == (0, 0)


class TestFlattenNestedSequence:
    """Tests for flatten_nested_sequence."""

    def test_flat_shapes(self):
        assert flatten_nested_sequence(((3, 3), (3, 5))) == ((3, 3), (3, 5))

    def test_nested_shapes(self):
        assert flatten_nested_sequence((((3, 3), (3, 2)), (3, 5))) == ((3, 3), (3, 2), (3, 5))

    def test_all_nested(self):
        assert flatten_nested_sequence((((1,), (2,)), ((3,), (4,)))) == ((1,), (2,), (3,), (4,))

    def test_with_none(self):
        assert flatten_nested_sequence((((3, 3), (3, 2)), None, (3, 5))) == ((3, 3), (3, 2), None, (3, 5))

    def test_flat_strings(self):
        assert flatten_nested_sequence(("ND", "ND")) == ("ND", "ND")

    def test_nested_strings(self):
        assert flatten_nested_sequence(
            (("ND",), "ND")) == (("ND",), "ND")

    def test_empty(self):
        assert flatten_nested_sequence(()) == ()

    def test_single_element(self):
        assert flatten_nested_sequence(((3, 5),)) == ((3, 5),)
