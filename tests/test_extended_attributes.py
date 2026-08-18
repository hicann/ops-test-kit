#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tests for attributes1~attributes9 extension columns merged into attributes.
"""
import pytest

from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp
from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e

CLS_LIST = [TestcaseOp, TestcaseAclnn, TestcaseE2e]


def _make(cls, **attrs):
    case = cls()
    case.is_valid = True
    case.fail_reason = None
    case.testcase_name = "ext_attr_case"
    case.attributes = attrs.pop("attributes", None)
    for k, v in attrs.items():
        setattr(case, k, v)
    return case


@pytest.mark.parametrize("cls", CLS_LIST)
def test_headers_declared(cls):
    hdrs = cls.complete_headers
    assert "attributes" in hdrs
    for i in range(1, 10):
        assert f"attributes{i}" in hdrs


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_basic(cls):
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2})
    case._merge_extended_attributes()
    assert case.attributes == {"a": 1, "b": 2}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_override_with_base(cls):
    case = _make(cls, attributes={"a": 1}, attributes1={"a": 2})
    case._merge_extended_attributes()
    assert case.attributes == {"a": 2}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_override_across_extended_columns(cls):
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2}, attributes2={"b": 3})
    case._merge_extended_attributes()
    assert case.attributes == {"a": 1, "b": 3}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_override_chain_last_wins(cls):
    case = _make(cls, attributes={"k": 0}, attributes1={"k": 1},
                 attributes5={"k": 5}, attributes9={"k": 9})
    case._merge_extended_attributes()
    assert case.attributes == {"k": 9}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_multi_column_merge(cls):
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2},
                 attributes2={"c": 3}, attributes3={"d": 4})
    case._merge_extended_attributes()
    assert case.attributes == {"a": 1, "b": 2, "c": 3, "d": 4}


@pytest.mark.parametrize("cls", CLS_LIST)
def test_no_extension_leaves_attributes(cls):
    case = _make(cls, attributes={"a": 1})
    case._merge_extended_attributes()
    assert case.attributes == {"a": 1}
    assert case.is_valid is True


@pytest.mark.parametrize("cls", CLS_LIST)
def test_merge_when_attributes_none(cls):
    case = _make(cls, attributes=None, attributes1={"b": 2})
    case._merge_extended_attributes()
    assert case.attributes == {"b": 2}


@pytest.mark.parametrize("cls", CLS_LIST)
def test_skip_when_invalid(cls):
    case = _make(cls, attributes={"a": 1}, attributes1={"b": 2})
    case.is_valid = False
    case._merge_extended_attributes()
    assert case.attributes == {"a": 1}


def test_validate_invokes_merge():
    from ttk.core_modules.testcase_manager.testcase_base import TestcaseBase
    case = TestcaseOp()
    case.is_valid = True
    case.attributes = {"a": 1}
    case.attributes1 = {"b": 2}
    TestcaseBase.validate(case)
    assert case.attributes == {"a": 1, "b": 2}
