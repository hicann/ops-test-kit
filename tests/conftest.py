#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Pytest configuration and shared fixtures for ttk tests.

Provides:
  - Device-free test isolation (autouse)
  - Shared helpers for constructing mock testcase structures
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_ttk_environment(monkeypatch):
    """Ensure tests do not accidentally pick up real device or config."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    monkeypatch.delenv("ASCEND_ROOT", raising=False)


@pytest.fixture
def make_testcase():
    """Factory fixture to create TestcaseAclnn instances for testing.

    Usage:
        case = make_testcase(
            api_name="aclnnDummy",
            tensor_view_shapes=(((3,3),(3,2)), (3,5)),
            tensor_dtypes=("float32",),
        )
    """

    def _make(api_name="aclnnDummy", **kwargs):
        from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
        case = TestcaseAclnn()
        case.api_name = api_name
        case.is_valid = True
        case.fail_reason = None
        case.attributes = kwargs.pop("attributes", {})
        for k, v in kwargs.items():
            setattr(case, k, v)
        return case

    return _make
