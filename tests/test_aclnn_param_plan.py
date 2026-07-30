#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for AclnnParamPlan and TestcaseAclnn.get_param_plan().
"""

import pytest
from collections import OrderedDict
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_aclnn import AclnnParamPlan
from ttk.core_modules.aclnn.op_api_info_keeper import OpApiInfo


def _make_op_api_info(params_dict):
    return OpApiInfo(params=OrderedDict(params_dict))


class TestAclnnParamPlanInit:

    def test_tensor_only(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("y", {"type": "aclTensor*", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnAdd", info)
        assert plan.tensor_count == 2
        assert plan.scalar_count == 0
        assert len(plan.param_layout) == 2
        assert plan.param_layout[0][0] == 'tensor'
        assert plan.param_layout[1][0] == 'tensor'

    def test_tensor_and_scalar(self):
        info = _make_op_api_info([
            ("self", {"type": "aclTensor*", "default": None}),
            ("dim", {"type": "aclScalar*", "default": None}),
            ("out", {"type": "aclTensor*", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnSoftmax", info)
        assert plan.tensor_count == 2
        assert plan.scalar_count == 1
        assert plan.param_layout[0] == ('tensor', 'self', 'aclTensor*', None)
        assert plan.param_layout[1] == ('scalar', 'dim', 'aclScalar*', None)
        assert plan.param_layout[2] == ('tensor', 'out', 'aclTensor*', None)

    def test_with_attributes(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("alpha", {"type": "float", "default": "1.0"}),
            ("dtype", {"type": "aclDataType", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnCast", info)
        assert plan.tensor_count == 1
        assert plan.scalar_count == 0
        assert len(plan.param_layout) == 3
        assert plan.param_layout[1][0] == 'other'
        assert plan.param_layout[1][1] == 'alpha'
        assert plan.param_layout[2][0] == 'other'
        assert plan.param_layout[2][1] == 'dtype'

    def test_tensor_list(self):
        info = _make_op_api_info([
            ("tensors", {"type": "aclTensorList*", "default": None}),
            ("dim", {"type": "int64_t", "default": "0"}),
        ])
        plan = AclnnParamPlan("aclnnCat", info)
        assert plan.tensor_count == 1
        assert plan.param_layout[0] == ('tensor', 'tensors', 'aclTensorList*', None)

    def test_scalar_list(self):
        info = _make_op_api_info([
            ("self", {"type": "aclTensor*", "default": None}),
            ("scalarList", {"type": "aclScalarList*", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnAddScalarList", info)
        assert plan.tensor_count == 1
        assert plan.scalar_count == 1
        assert plan.param_layout[1][0] == 'scalar'

    def test_full_signature(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("weight", {"type": "aclTensor*", "default": None}),
            ("bias", {"type": "aclTensor*", "default": None}),
            ("scale", {"type": "aclScalar*", "default": None}),
            ("groups", {"type": "int64_t", "default": "1"}),
            ("format", {"type": "aclDataType", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnConv2d", info)
        assert plan.tensor_count == 3
        assert plan.scalar_count == 1
        assert len(plan.param_layout) == 6


class TestAclnnParamPlanBuildArgs:

    def test_tensor_only_args(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("y", {"type": "aclTensor*", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnAdd", info)
        tensors = ["T_X", "T_Y"]
        args, extra = plan.build_args(tensors, [], {})
        assert args == ["T_X", "T_Y"]
        assert extra == {}

    def test_interleaved_tensor_scalar_attr(self):
        info = _make_op_api_info([
            ("self", {"type": "aclTensor*", "default": None}),
            ("dim", {"type": "aclScalar*", "default": None}),
            ("out", {"type": "aclTensor*", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnSoftmax", info)
        tensors = ["T_SELF", "T_OUT"]
        scalars = ["S_DIM"]
        args, extra = plan.build_args(tensors, scalars, {})
        assert args == ["T_SELF", "S_DIM", "T_OUT"]
        assert extra == {}

    def test_attribute_from_dict(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("alpha", {"type": "float", "default": "1.0"}),
        ])
        plan = AclnnParamPlan("aclnnAddAlpha", info)
        args, extra = plan.build_args(["T_X"], [], {"alpha": 2.5})
        assert args == ["T_X", 2.5]
        assert extra == {}

    def test_attribute_default_fallback(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("alpha", {"type": "float", "default": "1.0"}),
        ])
        plan = AclnnParamPlan("aclnnAddAlpha", info)
        args, extra = plan.build_args(["T_X"], [], {})
        assert args == ["T_X", "1.0"]
        assert extra == {}

    def test_attribute_missing_no_default(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("dtype", {"type": "aclDataType", "default": None}),
        ])
        plan = AclnnParamPlan("aclnnCast", info)
        args, extra = plan.build_args(["T_X"], [], {})
        assert args == ["T_X", None]
        assert extra == {}

    def test_full_conv2d_like(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("weight", {"type": "aclTensor*", "default": None}),
            ("bias", {"type": "aclTensor*", "default": None}),
            ("scale", {"type": "aclScalar*", "default": None}),
            ("groups", {"type": "int64_t", "default": "1"}),
        ])
        plan = AclnnParamPlan("aclnnConv2d", info)
        tensors = ["T_X", "T_W", "T_B"]
        scalars = ["S_SCALE"]
        attrs = {"groups": 4}
        args, extra = plan.build_args(tensors, scalars, attrs)
        assert args == ["T_X", "T_W", "T_B", "S_SCALE", 4]
        assert extra == {}

    def test_tensor_list_preserved(self):
        info = _make_op_api_info([
            ("tensors", {"type": "aclTensorList*", "default": None}),
            ("dim", {"type": "int64_t", "default": "0"}),
        ])
        plan = AclnnParamPlan("aclnnCat", info)
        tensor_list = [["T_A", "T_B", "T_C"]]
        args, extra = plan.build_args(tensor_list, [], {"dim": 1})
        assert args[0] == ["T_A", "T_B", "T_C"]
        assert args[1] == 1
        assert extra == {}


class TestGetParamPlan:

    def test_plan_cached(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
        ])
        from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
        case = TestcaseAclnn()
        case.api_name = "aclnnAdd"
        plan = AclnnParamPlan("aclnnAdd", info)
        case._param_plan_cache = plan
        assert case.get_param_plan() is plan
        assert case.get_param_plan() is plan

    def test_plan_none_for_no_api(self):
        from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
        case = TestcaseAclnn()
        case.api_name = None
        assert case.get_param_plan() is None

    def test_plan_build_args_integration(self):
        info = _make_op_api_info([
            ("x", {"type": "aclTensor*", "default": None}),
            ("alpha", {"type": "float", "default": "1.0"}),
        ])
        from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
        case = TestcaseAclnn()
        case.api_name = "aclnnAddAlpha"
        plan = AclnnParamPlan("aclnnAddAlpha", info)
        case._param_plan_cache = plan
        args, extra = plan.build_args(["T_X"], [], {"alpha": 3.0})
        assert args == ["T_X", 3.0]
