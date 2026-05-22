#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.framework_api.execution:
_match_overload, _coerce_value, build_positional_args, ParamPlan.build_args,
and ttk.core_modules.testcase_manager.testcase_e2e: get_param_plan.
"""

import pytest
from ttk.core_modules.testcase_manager.param_plan import (
    build_positional_args as _real_build_positional_args, match_overload, coerce_value,
)
from ttk.utilities.simple_param_extractor import ParamInfo
from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper
from ttk.utilities.simple_param_extractor import _MANUAL_OVERRIDES, OverloadInfo, APIParamInfo


@pytest.fixture(autouse=True)
def _clean_keeper():
    FrameworkApiInfoKeeper().clear_cache()
    _MANUAL_OVERRIDES.clear()
    yield
    FrameworkApiInfoKeeper().clear_cache()
    _MANUAL_OVERRIDES.clear()


def _register(api_name, params, source="test"):
    FrameworkApiInfoKeeper().register(api_name, params, source=source)


def _match_overload(api_name, input_tensor_count, attributes=None, tensor_distribution=None):
    api_info = FrameworkApiInfoKeeper().get(api_name)
    return match_overload(api_name, input_tensor_count, attributes, tensor_distribution, api_info)


def _build_positional_args(api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution=None):
    api_info = FrameworkApiInfoKeeper().get(api_name)
    return _real_build_positional_args(api_name, nested_tensors, attributes, output_tensor_indexes, tensor_distribution, api_info)


class TestMatchOverload:

    def test_single_overload_match(self):
        _register("torch.t1", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int"),
        ])
        params, oidx = _match_overload("torch.t1", 1, {"dim": "0"})
        assert params is not None
        assert oidx == 0

    def test_multi_overload_select_by_tensor_count(self):
        _register("torch.t2", [
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            [ParamInfo(name="input", type="Tensor")],
        ])
        params1, oidx1 = _match_overload("torch.t2", 1)
        assert oidx1 == 1
        params2, oidx2 = _match_overload("torch.t2", 2)
        assert oidx2 == 0

    def test_multi_overload_select_by_attribute_score(self):
        _register("torch.t3", [
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="min", type="Tensor")],
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="min", type="Number")],
        ])
        params, oidx = _match_overload("torch.t3", 1, {"min": "0.5"})
        assert oidx == 1

    def test_no_match_returns_none(self):
        _register("torch.t4", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        params, oidx = _match_overload("torch.t4", 3)
        assert params is None
        assert oidx == -1

    def test_tensor_distribution_filter(self):
        _register("torch.t5", [
            [ParamInfo(name="tensors", type="tuple of Tensors")],
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
        ])
        params_nested, oidx_nested = _match_overload("torch.t5", 1, tensor_distribution=[True])
        assert oidx_nested == 0
        params_flat, oidx_flat = _match_overload("torch.t5", 2, tensor_distribution=[False, False])
        assert oidx_flat == 1

    def test_unknown_api_returns_none(self):
        params, oidx = _match_overload("nonexistent.api", 1)
        assert params is None


class TestScoreAttrTypeCompatibility:
    """Tests for _score_attr_type_compatibility and its role as overload tiebreaker."""

    def test_str_attr_prefers_str_overload_over_number(self):
        _register("torch.matrix_norm_like", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="ord", type="Number"),
             ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
             ParamInfo(name="keepdim", type="bool", default="False", is_optional=True)],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="ord", type="str", default="fro", is_optional=True),
             ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
             ParamInfo(name="keepdim", type="bool", default="False", is_optional=True)],
        ])
        params, oidx = _match_overload(
            "torch.matrix_norm_like", 1,
            {"ord": "fro", "dim": (-2, -1), "keepdim": False})
        assert oidx == 1

    def test_int_attr_prefers_number_overload_over_str(self):
        _register("torch.matrix_norm_like2", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="ord", type="Number"),
             ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
             ParamInfo(name="keepdim", type="bool", default="False", is_optional=True)],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="ord", type="str", default="fro", is_optional=True),
             ParamInfo(name="dim", type="tuple of ints", default="(-2,-1)", is_optional=True),
             ParamInfo(name="keepdim", type="bool", default="False", is_optional=True)],
        ])
        params, oidx = _match_overload(
            "torch.matrix_norm_like2", 1,
            {"ord": 2, "dim": (-2, -1), "keepdim": False})
        assert oidx == 0

    def test_bool_attr_prefers_bool_overload(self):
        _register("torch.bool_overload_test", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="flag", type="str"),
             ParamInfo(name="dim", type="int")],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="flag", type="bool"),
             ParamInfo(name="dim", type="int")],
        ])
        params, oidx = _match_overload(
            "torch.bool_overload_test", 1,
            {"flag": True, "dim": 0})
        assert oidx == 1

    def test_tuple_attr_prefers_tuple_overload(self):
        _register("torch.tuple_overload_test", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="size", type="int"),
             ParamInfo(name="mode", type="str")],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="size", type="tuple of ints"),
             ParamInfo(name="mode", type="str")],
        ])
        params, oidx = _match_overload(
            "torch.tuple_overload_test", 1,
            {"size": (2, 3), "mode": "bilinear"})
        assert oidx == 1

    def test_key_score_takes_priority_over_value_score(self):
        _register("torch.key_priority_test", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="alpha", type="Number")],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="alpha", type="Number"),
             ParamInfo(name="extra", type="str")],
        ])
        params, oidx = _match_overload(
            "torch.key_priority_test", 1,
            {"alpha": 0.5})
        assert oidx == 0

    def test_live_matrix_norm_overload_selection(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        params, oidx = _match_overload(
            "torch.linalg.matrix_norm", 1,
            {"ord": "fro", "dim": (-2, -1), "keepdim": False})
        assert oidx == 1
        ord_param = next(p for p in params if p.name == "ord")
        assert ord_param.type == "str"


class TestCoerceValue:

    def test_none_passthrough(self):
        assert coerce_value(None, "int") is None

    def test_string_none(self):
        assert coerce_value("None", "int") is None

    def test_bool_string(self):
        assert coerce_value("true", "bool") is True
        assert coerce_value("false", "bool") is False
        assert coerce_value("1", "bool") is True

    def test_bool_native(self):
        assert coerce_value(True, "bool") is True
        assert coerce_value(False, "bool") is False

    def test_int_string(self):
        assert coerce_value("3", "int") == 3

    def test_float_string(self):
        assert coerce_value("3.5", "float") == 3.5

    def test_number_coerces_float(self):
        assert coerce_value("2.0", "Number") == 2.0

    def test_str_passthrough(self):
        assert coerce_value("hello", "str") == "hello"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unsupported type"):
            coerce_value(42, "UnknownType")

    def test_int_tuple_fallback(self):
        assert coerce_value((2, 2), "int") == (2, 2)

    def test_int_list_fallback(self):
        assert coerce_value([2, 3], "int") == (2, 3)

    def test_float_tuple_fallback(self):
        assert coerce_value((1.0, 2.0), "float") == (1.0, 2.0)

    def test_float_list_fallback(self):
        assert coerce_value([1, 2], "float") == (1.0, 2.0)

    def test_int_still_works_for_scalar(self):
        assert coerce_value(2, "int") == 2
        assert coerce_value("2", "int") == 2

    def test_float_still_works_for_scalar(self):
        assert coerce_value(1.0, "float") == 1.0
        assert coerce_value("1.5", "float") == 1.5


class TestBuildPositionalArgs:

    def test_simple_two_tensors(self):
        """torch.add(input, other, *, alpha=None, out=None) with 2 tensors."""
        _register("torch.add", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="alpha", type="Number", default="1",
                      is_optional=True, is_keyword_only=True),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ])
        t1, t2 = "T1", "T2"
        args, kwargs, oidx = _build_positional_args("torch.add", [t1, t2], {}, ())
        assert args == ["T1", "T2"]
        assert kwargs == {"alpha": 1.0}
        assert oidx == 0

    def test_with_attribute_kwargs(self):
        """torch.add with alpha=2.0 attribute goes to kwargs."""
        _register("torch.add_a", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="alpha", type="Number", default="1",
                      is_optional=True, is_keyword_only=True),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ])
        t1, t2 = "T1", "T2"
        args, kwargs, oidx = _build_positional_args(
            "torch.add_a", [t1, t2], {"alpha": "2.0"}, ())
        assert args == ["T1", "T2"]
        assert kwargs == {"alpha": 2.0}

    def test_interleaved_tensor_scalar(self):
        """torch.gather(input, dim, index) -- tensor, int, tensor interleaving."""
        _register("torch.gather", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int"),
            ParamInfo(name="index", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
            ParamInfo(name="sparse_grad", type="bool", default="False",
                      is_optional=True, is_keyword_only=True),
        ])
        t1, t2 = "T_INPUT", "T_INDEX"
        args, kwargs, oidx = _build_positional_args(
            "torch.gather", [t1, t2], {"dim": "1"}, ())
        assert args == ["T_INPUT", 1, "T_INDEX"]
        assert kwargs == {"sparse_grad": False}

    def test_out_tensor_from_output_indexes(self):
        """torch.abs(input, *, out=None) -- out comes from output_tensor_indexes."""
        _register("torch.abs_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ])
        t_in, t_out = "T_IN", "T_OUT"
        args, kwargs, oidx = _build_positional_args(
            "torch.abs_out", [t_in, t_out], {}, (1,))
        assert args == ["T_IN"]
        assert kwargs == {"out": "T_OUT"}

    def test_out_not_in_kwargs_when_none(self):
        """If output_tensor_indexes is empty, out=None should NOT appear in kwargs."""
        _register("torch.abs_no_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ])
        t_in = "T_IN"
        args, kwargs, oidx = _build_positional_args(
            "torch.abs_no_out", [t_in], {}, ())
        assert args == ["T_IN"]
        assert "out" not in kwargs

    def test_tensorlist_param(self):
        """torch.cat(tensors, dim) -- TensorList as single positional."""
        _register("torch.cat", [
            ParamInfo(name="tensors", type="tuple of Tensors"),
            ParamInfo(name="dim", type="int", default="0"),
        ])
        tl = ["T_A", "T_B"]
        args, kwargs, oidx = _build_positional_args(
            "torch.cat", [tl], {"dim": "0"}, ())
        assert args == [["T_A", "T_B"], 0]
        assert kwargs == {}

    def test_multi_overload_tensor_scalar(self):
        """torch.div with 2 overloads: (T, T) and (T, Number).
        With 1 tensor + 'other' attr should match scalar overload."""
        _register("torch.div", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="other", type="Tensor"),
             ParamInfo(name="rounding_mode", type="str", default="None",
                       is_optional=True, is_keyword_only=True),
             ParamInfo(name="out", type="Tensor", default="None",
                       is_optional=True, is_keyword_only=True)],
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="other", type="Number"),
             ParamInfo(name="rounding_mode", type="str", default="None",
                       is_optional=True, is_keyword_only=True),
             ParamInfo(name="out", type="Tensor", default="None",
                       is_optional=True, is_keyword_only=True)],
        ])
        t1 = "T_IN"
        args, kwargs, oidx = _build_positional_args(
            "torch.div", [t1], {"other": "3.0"}, ())
        assert oidx == 1
        assert args == ["T_IN", 3.0]

    def test_default_value_used_for_missing_attr(self):
        """Param with default but no matching attr, default is used."""
        _register("torch.def_test", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int", default="0"),
        ])
        t1 = "T_IN"
        args, kwargs, oidx = _build_positional_args(
            "torch.def_test", [t1], {}, ())
        assert args == ["T_IN", 0]
        assert kwargs == {}
    def test_no_tensor_no_default_gets_none(self):
        """Non-tensor param with no attr and no default gets None (no auto-inference)."""
        _register("torch.none_test", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Number"),
        ])
        t1 = "T_IN"
        args, kwargs, oidx = _build_positional_args(
            "torch.none_test", [t1], {}, ())
        assert args[0] == "T_IN"
        assert args[1] is None  # no smart default — strict mode

    def test_raises_when_overload_not_matched(self):
        """Should raise ValueError when no overload matches tensor count."""
        _register("torch.raise_test", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int", default="0"),
            ParamInfo(name="index", type="Tensor"),
        ])
        with pytest.raises(ValueError, match="Cannot match overload"):
            _build_positional_args("torch.raise_test", ["T1"], {}, ())

    def test_optional_tensor_gets_default_when_queue_empty(self):
        """Optional tensor param gets smart default when tensor queue empties."""
        _register("torch.raise_queue", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor", is_optional=True),
        ])
        args, kwargs, oidx = _build_positional_args(
            "torch.raise_queue", ["T1"], {}, ())
        assert args[0] == "T1"
        assert len(args) == 2  # optional tensor filled with None
        assert args[1] is None
        assert kwargs == {}


class TestVarPositional:
    """Tests for *args tensor parameter support (is_var_positional=True)."""

    def test_var_positional_consumes_all_remaining_tensors(self):
        _register("torch.block_diag", [
            ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
        ])
        args, kwargs, oidx = _build_positional_args(
            "torch.block_diag", ["T1", "T2", "T3"], {}, ())
        assert args == ["T1", "T2", "T3"]
        assert kwargs == {}
        assert oidx == 0

    def test_var_positional_with_two_tensors(self):
        _register("torch.block_diag_2", [
            ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
        ])
        args, kwargs, oidx = _build_positional_args(
            "torch.block_diag_2", ["T1", "T2"], {}, ())
        assert args == ["T1", "T2"]
        assert kwargs == {}

    def test_match_overload_var_positional_accepts_any_count(self):
        _register("torch.vp_match", [
            [ParamInfo(name="tensors", type="Tensor", is_var_positional=True)],
        ])
        params1, oidx1 = _match_overload("torch.vp_match", 2)
        assert params1 is not None
        assert oidx1 == 0
        params2, oidx2 = _match_overload("torch.vp_match", 5)
        assert params2 is not None
        assert oidx2 == 0

    def test_match_overload_var_positional_min_zero(self):
        _register("torch.vp_min", [
            [ParamInfo(name="tensors", type="Tensor", is_var_positional=True)],
        ])
        params, oidx = _match_overload("torch.vp_min", 0)
        assert params is not None

    def test_match_overload_non_var_still_enforces_upper_bound(self):
        _register("torch.vp_no_var", [
            [ParamInfo(name="input", type="Tensor"),
             ParamInfo(name="other", type="Tensor")],
        ])
        params, oidx = _match_overload("torch.vp_no_var", 3)
        assert params is None

    def test_param_plan_var_positional(self):
        _register("torch.vp_plan", [
            ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
        ])
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params, oidx = _match_overload("torch.vp_plan", 3)
        plan = ParamPlan("torch.vp_plan", params, oidx, (), {})
        args, kwargs, _ = plan.build_args(["A", "B", "C"])
        assert args == ["A", "B", "C"]
        assert kwargs == {}

    def test_live_torch_block_diag(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        args, kwargs, oidx = _build_positional_args(
            "torch.block_diag", ["T1", "T2"], {}, ())
        assert args == ["T1", "T2"]
        assert kwargs == {}
        assert oidx == 0


class TestParamPlanBuildArgs:
    """Tests for ParamPlan.build_args (reusable plan)."""

    def test_plan_reused_produces_same_result(self):
        _register("torch.plan_reuse", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int", default="0"),
        ])
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params, oidx = _match_overload("torch.plan_reuse", 1)
        plan = ParamPlan("torch.plan_reuse", params, oidx, (), {})

        args1, kwargs1, _ = plan.build_args(["T_A"])
        args2, kwargs2, _ = plan.build_args(["T_B"])
        assert args1 == ["T_A", 0]
        assert args2 == ["T_B", 0]
        assert kwargs1 == {}
        assert kwargs2 == {}

    def test_plan_with_out_tensor(self):
        _register("torch.plan_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_keyword_only=True, is_optional=True),
        ])
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params, oidx = _match_overload("torch.plan_out", 1)
        plan = ParamPlan("torch.plan_out", params, oidx, (0,), {})

        t_in, t_out = "T_IN", "T_OUT"
        args, kwargs, _ = plan.build_args([t_in, t_out])
        assert args == ["T_OUT"]  # T_IN filtered to out_tensors, T_OUT is input
        assert kwargs == {"out": "T_IN"}  # T_IN placed as out kwarg


class TestGetParamPlan:
    """Tests for testcase.get_param_plan() -- cached on testcase."""

    def _make_testcase(self, api_name, shapes, dtypes, attrs=None, output_indexes=()):
        from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
        case = TestcaseE2e()
        case.api_name = api_name
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.output_tensor_indexes = output_indexes
        case.validate()
        return case

    def test_plan_cached(self):
        case = self._make_testcase(
            'torch.abs',
            ((2, 3), (2, 3)),
            ('float32', 'float32'),
            output_indexes=(1,))
        plan1 = case.get_param_plan()
        plan2 = case.get_param_plan()
        assert plan1 is plan2

    def test_plan_none_for_no_api(self):
        case = self._make_testcase(
            'nonexistent.api',
            ((2, 3),),
            ('float32',))
        plan = case.get_param_plan()
        assert plan is None

    def test_plan_build_args_integration(self):
        import numpy as np
        _register("torch.np_test", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = self._make_testcase(
            'torch.np_test',
            ((2, 3), (2, 3)),
            ('float32', 'float32'))
        plan = case.get_param_plan()
        assert plan is not None
        tensors = [np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)]
        args, kwargs, _ = plan.build_args(tensors)
        assert len(args) == 2
        assert isinstance(args[0], np.ndarray)


class TestNpuWeightQuantBatchmatmul:
    """Tests for torch_npu.npu_weight_quant_batchmatmul with optional tensors."""

    @staticmethod
    def _make_case(variant, shapes, dtypes, attrs=None):
        from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
        case = TestcaseE2e()
        case.api_name = 'torch_npu.npu_weight_quant_batchmatmul'
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.validate()
        return case

    def test_per_channel_4_tensors_3_none(self):
        import numpy as np
        case = self._make_case('per_channel',
            ((4, 16), (16, 8), (1, 8), (1, 8), None, None, None),
            ('float16', 'int8', 'float16', 'float16', None, None, None))
        assert case.is_valid

        plan = case.get_param_plan()
        assert plan is not None
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((1, 8), np.float16),
            np.zeros((1, 8), np.float16),
            None, None, None,
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (1, 8)
        assert args[3].shape == (1, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6] is None
        assert args[7] == 0
        assert args[8] == 0
        assert kwargs == {}

    def test_with_bias_5_tensors_2_none(self):
        import numpy as np
        case = self._make_case('with_bias',
            ((4, 16), (16, 8), (1, 8), (1, 8), None, None, (4, 8)),
            ('float16', 'int8', 'float16', 'float16', None, None, 'float16'))
        assert case.is_valid

        plan = case.get_param_plan()
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((1, 8), np.float16),
            np.zeros((1, 8), np.float16),
            None, None,
            np.zeros((4, 8), np.float16),
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (1, 8)
        assert args[3].shape == (1, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6].shape == (4, 8)
        assert args[7] == 0
        assert args[8] == 0
        assert kwargs == {}

    def test_per_group_with_attr(self):
        import numpy as np
        case = self._make_case('per_group',
            ((4, 16), (16, 8), (2, 8), (2, 8), None, None, None),
            ('float16', 'int8', 'float16', 'float16', None, None, None),
            attrs={'antiquant_group_size': 8})
        assert case.is_valid

        plan = case.get_param_plan()
        tensors = [
            np.zeros((4, 16), np.float16),
            np.zeros((16, 8), np.int8),
            np.zeros((2, 8), np.float16),
            np.zeros((2, 8), np.float16),
            None, None, None,
        ]
        args, kwargs, _ = plan.build_args(tensors)

        assert args[0].shape == (4, 16)
        assert args[1].shape == (16, 8)
        assert args[2].shape == (2, 8)
        assert args[3].shape == (2, 8)
        assert args[4] is None
        assert args[5] is None
        assert args[6] is None
        assert args[7] == 8
        assert args[8] == 0
        assert kwargs == {}


class TestInplaceTensorMethodBuildArgs:

    def test_inplace_build_args_includes_self(self):
        import numpy as np
        _register("torch.Tensor.fake_add_", [
            ParamInfo(name="self", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="alpha", type="Number", default="1", is_optional=True,
                      is_keyword_only=True),
        ])
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        plan = ParamPlan(
            api_name="torch.Tensor.fake_add_",
            overload_params=_MANUAL_OVERRIDES["torch.Tensor.fake_add_"].params,
            overload_index=0,
            output_tensor_indexes=(0,),
            attributes={"alpha": "2"})
        tensors = [
            np.zeros((2, 3), np.float32),
            np.ones((2, 3), np.float32),
        ]
        args, kwargs, _ = plan.build_args(tensors)
        assert len(args) == 2
        assert args[0].shape == (2, 3)
        assert args[1].shape == (2, 3)
        assert kwargs.get("alpha") == 2.0


class TestCheckInputCountExceededVarPos:
    """Tests for _check_input_count_exceeded skipping VAR_POSITIONAL overloads."""

    def _make_testcase(self, api_name, shapes, dtypes, attrs=None, output_indexes=()):
        from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
        case = TestcaseE2e()
        case.api_name = api_name
        case.tensor_view_shapes = shapes
        case.tensor_dtypes = dtypes
        case.attributes = attrs or {}
        case.output_tensor_indexes = output_indexes
        case.validate()
        return case

    def test_var_pos_overload_skips_input_count_check(self):
        _register("torch.block_diag", [
            ParamInfo(name="tensors", type="Tensor", is_var_positional=True),
        ])
        case = self._make_testcase(
            "torch.block_diag",
            ((4, 4), (4, 4)),
            ("float32", "float32"))
        assert case.is_valid
        assert case.fail_reason != "INPUT_COUNT_EXCEEDED"

    def test_non_var_pos_overload_still_checked(self):
        _register("torch.strict_two_tensors", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = self._make_testcase(
            "torch.strict_two_tensors",
            ((4, 4), (4, 4), (4, 4)),
            ("float32", "float32", "float32"))
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_live_block_diag_passes(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = self._make_testcase(
            "torch.block_diag",
            ((4, 4), (4, 4)),
            ("float32", "float32"))
        assert case.is_valid


class TestTensorListOutBuildArgs:

    def test_tensor_list_out_assembles_tuple(self):
        import numpy as np
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ]
        t0 = np.zeros((2, 3))
        t1 = np.ones((2, 3))
        t2 = np.full((2, 3), 2.0)
        t3 = np.full((2, 3), 3.0)
        t4 = np.full((2, 3), 4.0)
        plan = ParamPlan(
            api_name="torch_npu.test_encode",
            overload_params=params,
            overload_index=0,
            output_tensor_indexes=(1, 2, 3, 4),
            attributes={},
        )
        args, kwargs, extra = plan.build_args([t0, t1, t2, t3, t4])
        assert len(args) == 1
        assert 'out' in kwargs
        assert len(kwargs['out']) == 4
        assert kwargs['out'][0] is t1
        assert kwargs['out'][3] is t4

    def test_single_out_still_works(self):
        import numpy as np
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_keyword_only=True),
        ]
        t0 = np.zeros((2, 3))
        t1 = np.ones((2, 3))
        plan = ParamPlan(
            api_name="torch.abs",
            overload_params=params,
            overload_index=0,
            output_tensor_indexes=(1,),
            attributes={},
        )
        args, kwargs, extra = plan.build_args([t0, t1])
        assert len(args) == 1
        assert kwargs['out'] is t1

    def test_tensor_list_out_with_attrs(self):
        import numpy as np
        from ttk.core_modules.testcase_manager.param_plan import ParamPlan
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="statistic", type="bool", default=False, is_keyword_only=True),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ]
        tensors = [np.zeros((2, 3))] + [np.ones((2, 3)) for _ in range(4)]
        plan = ParamPlan(
            api_name="torch_npu.npu_hans_encode",
            overload_params=params,
            overload_index=0,
            output_tensor_indexes=(1, 2, 3, 4),
            attributes={"statistic": True},
        )
        args, kwargs, extra = plan.build_args(tensors)
        assert len(args) == 1
        assert kwargs['statistic'] is True
        assert 'out' in kwargs
        assert len(kwargs['out']) == 4