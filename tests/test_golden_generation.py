#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.framework_api.golden_generation:
_exec_and_convert, _call_plugin_with_plan, _run_api_on_cpu, generate_golden.
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch, MagicMock

from ttk.core_modules.framework_api.golden_generation import (
    generate_golden, _exec_and_convert, _call_plugin_with_plan, _run_api_on_cpu,
)
from ttk.core_modules.testcase_manager.param_plan import ParamPlan
from ttk.utilities.simple_param_extractor import ParamInfo
from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper


@pytest.fixture(autouse=True)
def _clean_keeper():
    FrameworkApiInfoKeeper().clear_cache()
    yield
    FrameworkApiInfoKeeper().clear_cache()


def _register(api_name, params, source="test"):
    FrameworkApiInfoKeeper().register(api_name, params, source=source)


def _make_testcase(api_name, shapes, dtypes, attrs=None,
                   output_tensor_indexes=(), golden_api=None):
    from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
    case = TestcaseE2e()
    case.testcase_name = f"test_{api_name}"
    case.api_name = api_name
    case.is_valid = True
    case.fail_reason = None
    case.tensor_view_shapes = shapes
    case.tensor_dtypes = dtypes
    case.attributes = attrs or {}
    case.output_tensor_indexes = output_tensor_indexes
    case.golden_api = golden_api
    return case


class TestExecAndConvert:

    def test_function_single_output(self):
        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            t = torch.tensor([1.0, 2.0])
            mock_fn = MagicMock(return_value=t)
            mock_resolve.return_value = (mock_fn, False)
            result = _exec_and_convert("torch.fake_add", [torch.zeros(2)], {})
            mock_fn.assert_called_once()
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_function_tuple_output(self):
        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            out = (torch.tensor([1.0]), torch.tensor([2.0]))
            mock_fn = MagicMock(return_value=out)
            mock_resolve.return_value = (mock_fn, False)
            result = _exec_and_convert("torch.fake_sort", [torch.zeros(3)], {})
            assert len(result) == 2
            np.testing.assert_array_equal(result[0], [1.0])
            np.testing.assert_array_equal(result[1], [2.0])

    def test_tensor_method_call(self):
        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            mock_tensor = MagicMock()
            mock_tensor.relu_ = MagicMock(return_value=torch.tensor([3.0]))
            mock_resolve.return_value = ("relu_", True)
            result = _exec_and_convert("torch.Tensor.relu_", [mock_tensor], {})
            mock_tensor.relu_.assert_called_once()
            assert len(result) == 1

    def test_function_with_kwargs(self):
        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            mock_fn = MagicMock(return_value=torch.tensor([1.0]))
            mock_resolve.return_value = (mock_fn, False)
            _exec_and_convert(
                "torch.fake_add",
                [torch.zeros(2), torch.ones(2)],
                {"alpha": 2.0})
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs == {"alpha": 2.0}


class TestCallPluginWithPlan:

    def test_plugin_with_plan_returns_numpy(self):
        _register("torch.plugin_test", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int", default="0"),
        ])
        case = _make_testcase("torch.plugin_test", ((3, 4),), ('float32',))
        arr = np.random.randn(3, 4).astype(np.float32)
        case.tensors = [arr]

        def fake_golden(input, dim=0):
            return input * 2

        result = _call_plugin_with_plan(case, fake_golden)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], arr * 2)

    def test_plugin_returns_list(self):
        _register("torch.plugin_list", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.plugin_list", ((3, 4),), ('float32',))
        arr = np.random.randn(3, 4).astype(np.float32)
        case.tensors = [arr]

        def fake_golden(input):
            return [input, input * 2]

        result = _call_plugin_with_plan(case, fake_golden)
        assert len(result) == 2

    def test_plugin_returns_torch_tensor(self):
        _register("torch.plugin_torch_ret", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.plugin_torch_ret", ((2,),), ('float32',))
        arr = np.array([1.0, 2.0], dtype=np.float32)
        case.tensors = [arr]

        def fake_golden(input):
            if isinstance(input, torch.Tensor):
                return input * 3
            return torch.from_numpy(input) * 3

        result = _call_plugin_with_plan(case, fake_golden)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [3.0, 6.0])


class TestRunCpuGolden:

    def test_simple_cpu_golden(self):
        _register("torch.relu", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.relu", ((3,),), ('float32',))
        raw = [np.array([-1.0, 2.0, -3.0], dtype=np.float32)]
        result = _run_api_on_cpu("torch.relu", raw, case, None)
        assert len(result) == 1
        np.testing.assert_array_almost_equal(result[0], [0.0, 2.0, 0.0])

    def test_cpu_golden_with_none(self):
        _register("torch.cpu_none", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor", is_optional=True),
        ])
        case = _make_testcase("torch.cpu_none", ((3,), None), ('float32', None))
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32), None]

        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            mock_fn = MagicMock(return_value=torch.tensor([4.0, 5.0, 6.0]))
            mock_resolve.return_value = (mock_fn, False)
            result = _run_api_on_cpu("torch.cpu_none", raw, case, None)
            mock_fn.assert_called_once()
            call_args = mock_fn.call_args[0]
            assert call_args[0] is not None
            assert call_args[1] is None

    def test_cpu_golden_add(self):
        _register("torch.add", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = _make_testcase("torch.add", ((3,), (3,)), ('float32', 'float32'))
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32),
               np.array([4.0, 5.0, 6.0], dtype=np.float32)]
        result = _run_api_on_cpu("torch.add", raw, case, None)


class TestGenerateGolden:

    def test_priority1_golden_api(self):
        _register("torch.golden_target", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase(
            "torch.base_api", ((3,),), ('float32',),
            golden_api="torch.golden_target")
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            mock_fn = MagicMock(return_value=torch.tensor([10.0, 20.0, 30.0]))
            mock_resolve.return_value = (mock_fn, False)
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], [10.0, 20.0, 30.0])

    def test_priority2_custom_plugin(self):
        _register("torch.relu", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.relu", ((3,),), ('float32',))
        raw = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
        case.tensors = raw

        fake_func = MagicMock(return_value=np.array([4.0, 5.0, 6.0]))
        with patch('ttk.core_modules.framework_api.golden_generation.get_plugin_function') as mock_get:
            mock_get.return_value = fake_func
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_equal(result[0], [4.0, 5.0, 6.0])

    def test_priority3_cpu_fallback(self):
        _register("torch.relu", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.relu", ((4,),), ('float32',))
        raw = [np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)]

        with patch('ttk.core_modules.framework_api.golden_generation.get_plugin_function') as mock_get:
            mock_get.return_value = None
            result = generate_golden(case, raw)
            assert len(result) == 1
            np.testing.assert_array_almost_equal(
                result[0], [0.0, 0.0, 1.0, 2.0])

    def test_priority3_cpu_fails_raises(self):
        _register("torch.noexist_api", [
            ParamInfo(name="input", type="Tensor"),
        ])
        case = _make_testcase("torch.noexist_api", ((3,),), ('float32',))
        raw = [np.array([1.0], dtype=np.float32)]

        with patch('ttk.core_modules.framework_api.golden_generation.get_plugin_function') as mock_get, \
             patch('ttk.core_modules.framework_api.golden_generation._run_api_on_cpu') as mock_cpu:
            mock_get.return_value = None
            mock_cpu.side_effect = RuntimeError("no CPU impl")
            with pytest.raises(RuntimeError, match="cannot run on CPU"):
                generate_golden(case, raw)

    def test_with_nested_distribution(self):
        _register("torch.stack_test", [
            ParamInfo(name="tensors", type="tuple of Tensors"),
        ])
        case = _make_testcase(
            "torch.stack_test",
            (((3,), (3,)),),
            (('float32', 'float32'),))
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        raw = [a, b]

        with patch('ttk.core_modules.framework_api.golden_generation.resolve_api') as mock_resolve:
            mock_fn = MagicMock(return_value=torch.zeros((2, 3)))
            mock_resolve.return_value = (mock_fn, False)
            result = generate_golden(case, raw)
            call_args = mock_fn.call_args[0]
            assert isinstance(call_args[0], (list, tuple))
