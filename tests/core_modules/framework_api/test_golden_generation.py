#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
测试 ttk.core_modules.framework_api.golden_generation 模块。

覆盖内容：
- generate_golden: 优先级链路（golden_api → 自定义插件 → CPU 回退）。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper
from ttk.core_modules.framework_api.golden_generation import generate_golden
from ttk.utilities.simple_param_extractor import ParamInfo


@pytest.fixture(autouse=True)
def _clean_keeper():
    FrameworkApiInfoKeeper().clear_cache()
    yield
    FrameworkApiInfoKeeper().clear_cache()


def _register(api_name, params, source="test"):
    """向 FrameworkApiInfoKeeper 注册 API 参数信息。"""
    FrameworkApiInfoKeeper().register(api_name, params, source=source)


def _make_testcase(api_name, shapes, dtypes, attrs=None,
                   output_tensor_indexes=(), golden_api=None):
    """构造一个 TestcaseE2e 测试用例。"""
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


class TestGenerateGolden:
    """generate_golden: 优先级链路 golden_api → 自定义插件 → CPU 回退。"""

    def test_priority1_golden_api(self):
        """优先级 1：指定 golden_api 时走 golden_api 路径。"""
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
        """优先级 2：有自定义插件时走插件路径。"""
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
        """优先级 3：无插件时回退到 CPU 执行。"""
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
        """CPU 回退失败时抛出 RuntimeError。"""
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
        """嵌套分布（tuple of Tensors）正确传递。"""
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
            generate_golden(case, raw)
            call_args = mock_fn.call_args[0]
            assert isinstance(call_args[0], (list, tuple))
