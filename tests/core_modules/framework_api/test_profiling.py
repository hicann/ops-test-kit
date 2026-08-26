#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
测试 ttk.core_modules.framework_api.profiling 模块。

覆盖内容：
- to_non_contiguous_view / default_generate_inputs: 非连续视图与默认输入生成；
- _apply_pre_compare: pre_compare 钩子的跳过条件、原地/返回模式、端到端集成；
- _try_custom_compare: 自定义 compare 的跳过条件、返回值适配、错误处理、端到端集成。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttk.core_modules.framework_api.input_generation import (
    default_generate_inputs,
    to_non_contiguous_view,
)


def _make_switches(**overrides):
    """构造测试用 SWITCHES 对象，默认 input_distribution='uniform'。"""
    from ttk.utilities.classes import SWITCHES

    sw = SWITCHES()
    sw.input_distribution = "uniform"
    for k, v in overrides.items():
        setattr(sw, k, v)
    return sw


def _make_testcase(
    shapes,
    dtypes,
    attrs=None,
    output_tensor_indexes=(),
    pure_output_indexes=None,
    storage_shapes=(),
    view_strides=(),
    view_offsets=(),
    input_data_ranges=None,
):
    """构造一个 MagicMock 测试用例，模拟 flat_tensor_* 系列接口。"""
    case = MagicMock()
    case.flat_tensor_view_shapes = shapes
    case.flat_tensor_dtypes = dtypes
    case.flat_input_data_ranges = input_data_ranges or ()
    case.pure_output_indexes = pure_output_indexes or []
    case.flat_tensor_storage_shapes = storage_shapes
    case.flat_tensor_view_strides = view_strides
    case.flat_tensor_view_offsets = view_offsets

    def flat_storage_side_effect(idx):
        if storage_shapes and idx < len(storage_shapes):
            val = storage_shapes[idx]
            if val is not None:
                return val
        return shapes[idx] if idx < len(shapes) else None

    def flat_stride_side_effect(idx):
        if view_strides and idx < len(view_strides):
            s = view_strides[idx]
            if s is not None and s != ():
                return s
        from ttk.utilities.container_utils import shape_stride

        v = shapes[idx] if idx < len(shapes) else None
        return shape_stride(v) if v is not None else None

    def flat_offset_side_effect(idx):
        if view_offsets and idx < len(view_offsets):
            val = view_offsets[idx]
            if val is not None:
                return val
        return 0

    case.flat_storage_shape.side_effect = flat_storage_side_effect
    case.flat_view_stride.side_effect = flat_stride_side_effect
    case.flat_view_offset.side_effect = flat_offset_side_effect
    return case


# --- 模块级 compare 辅助函数（供参数化测试引用） ---


def _compare_single_dict(*outputs):
    """返回单个 dict 结果。"""
    return {"pass": True, "precision": 99.5}


class TestToNonContiguousView:
    """to_non_contiguous_view: 通过 as_strided 构造非连续视图。"""

    def test_stride_with_offset(self):
        """带偏移的非连续视图，验证数据正确读取。"""
        storage = np.arange(12, dtype=np.float32)
        view_shape = (3, 2)
        view_stride = (4, 2)
        view_offset = 1
        result = to_non_contiguous_view(storage, view_shape, view_stride, view_offset)
        assert result.shape == (3, 2)
        expected = np.array([[1, 3], [5, 7], [9, 11]], dtype=np.float32)
        np.testing.assert_array_equal(np.array(result), expected)


class TestDefaultGenerateInputs:
    """default_generate_inputs: 默认输入张量生成逻辑。"""

    def test_basic_input_generation(self):
        """基本场景：两个 float32 张量正确生成。"""
        case = _make_testcase(shapes=((4,), (4,)), dtypes=("float32", "float32"))
        switches = _make_switches()
        inputs = default_generate_inputs(case, switches)
        assert len(inputs) == 2
        assert inputs[0].shape == (4,)
        assert inputs[0].dtype == np.float32
        assert inputs[1].shape == (4,)
        assert inputs[1].dtype == np.float32


class TestApplyPreCompareSkip:
    """_apply_pre_compare 的跳过条件：插件路径缺失/空 golden。"""

    def _make_switches_with_path(self, plugin_path=None):
        sw = _make_switches()
        sw.plugin_path = plugin_path
        return sw

    @pytest.mark.parametrize(
        "api_name, plugin_path, spec_return, result_init, golden_init, expected_result, expected_golden",
        [
            pytest.param(
                "softmax_v2",
                None,
                None,
                [np.array([1, 2])],
                [np.array([3, 4])],
                [1, 2],
                [3, 4],
                id="no_plugin_path",
            ),
            pytest.param(
                "add",
                "/fake/path",
                "mock",
                [np.array([1, 2])],
                [],
                [1, 2],
                None,
                id="empty_golden",
            ),
        ],
    )
    def test_skip_conditions(
        self, api_name, plugin_path, spec_return, result_init, golden_init, expected_result, expected_golden
    ):
        """验证各跳过条件下 result/golden 保持不变。"""
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare

        case = MagicMock()
        case.api_name = api_name
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path=plugin_path)
        result = list(result_init)
        golden = list(golden_init)
        spec_val = MagicMock() if spec_return == "mock" else spec_return
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=spec_val):
            _apply_pre_compare(case, result, golden, sw)
        if expected_result is not None:
            np.testing.assert_array_equal(result[0], expected_result)
        if expected_golden is not None:
            np.testing.assert_array_equal(golden[0], expected_golden)


class TestApplyPreCompareInplace:
    """_apply_pre_compare 原地修改模式：pre_compare 直接修改传入数组。"""

    def test_inplace_no_tensor_list(self):
        """无 TensorList 时 pre_compare 原地修改所有输出。"""
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare

        def pre_compare(*outputs):
            for arr in outputs:
                arr[:] = arr * 2

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])


class TestApplyPreCompareReturnMode:
    """_apply_pre_compare 返回模式：pre_compare 返回新列表替换 result/golden。"""

    def test_return_mode_no_tensor_list(self):
        """无 TensorList 时返回值直接替换。"""
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare

        def pre_compare(*outputs):
            return [arr * 2 for arr in outputs]

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0, 2.0])]
        golden = [np.array([3.0, 4.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            _apply_pre_compare(case, result, golden, sw)
        np.testing.assert_array_equal(result[0], [2.0, 4.0])
        np.testing.assert_array_equal(golden[0], [6.0, 8.0])

    def test_exception_propagates(self):
        """pre_compare 抛出的异常原样向上传播。"""
        from ttk.core_modules.framework_api.profiling import _apply_pre_compare

        def pre_compare(*outputs):
            raise RuntimeError("boom")

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        result = [np.array([1.0])]
        golden = [np.array([2.0])]
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=pre_compare):
            with pytest.raises(RuntimeError, match="boom"):
                _apply_pre_compare(case, result, golden, sw)


class TestPreCompareEndToEnd:
    """_do_profile 端到端：验证 pre_compare 在比较前被调用。"""

    def test_do_profile_calls_pre_compare_before_compare(self):
        """Verify _do_profile calls _apply_pre_compare and passes transformed data to _evaluate_eager_precision."""
        from ttk.core_modules.framework_api import profiling as prof_module

        def pre_compare(*outputs):
            return [arr * 2 for arr in outputs]

        captured = {}

        def fake_evaluate_eager_precision(
            testcase, raw_inputs, result_nps, golden_nps, switches, perf, return_struct, third_parties=None
        ):
            captured["result"] = list(result_nps)
            captured["golden"] = list(golden_nps)

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.get_param_plan.return_value = MagicMock()
        case.is_torch_dtype_support.return_value = True

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        mock_backend = MagicMock()
        mock_backend.use_device.return_value = False
        # Task 7 contract: graph mode guards on is_npu() (CPU-style -> False,
        # graph skipped). MagicMock auto-returns truthy, so pin explicitly.
        mock_backend.is_npu.return_value = False

        mock_return_struct = MagicMock()

        fake_result = [np.array([1.0, 2.0])]
        fake_golden = [np.array([10.0, 20.0])]

        with patch(
            "ttk.core_modules.framework_api.profiling.get_spec_attr",
            side_effect=lambda api, attr, path: pre_compare if attr == "pre_compare" else None,
        ), patch.object(prof_module, "get_process_context"), patch.object(
            prof_module, "resolve_api", return_value=(MagicMock(__name__="add"), False)
        ), patch.object(prof_module, "generate_inputs", return_value=[np.array([1.0])]), patch.object(
            prof_module, "_dump_inputs"
        ), patch.object(prof_module, "DeviceLock"), patch.object(prof_module, "_profiling_print"), patch.object(
            prof_module, "_execute_eager", return_value=(fake_result, MagicMock(), None)
        ), patch.object(prof_module, "_dump_outputs"), patch.object(
            prof_module, "_generate_golden_data", return_value=fake_golden
        ), patch.object(prof_module, "_evaluate_eager_precision", side_effect=fake_evaluate_eager_precision):
            prof_module._do_profile(case, mock_backend, {}, {}, 0, sw, mock_return_struct)

        np.testing.assert_array_equal(captured["result"][0], [2.0, 4.0])
        np.testing.assert_array_equal(captured["golden"][0], [20.0, 40.0])


class TestTryCustomCompareSkip:
    """_try_custom_compare 的跳过条件：返回 None 的各种场景。"""

    def _make_switches_with_path(self, plugin_path=None):
        sw = _make_switches()
        sw.plugin_path = plugin_path
        return sw

    @pytest.mark.parametrize(
        "api_name, plugin_path, spec_return, result_arr, golden_arr",
        [
            pytest.param(
                "add",
                None,
                None,
                [np.array([1.0, 2.0])],
                [np.array([3.0, 4.0])],
                id="no_plugin_path",
            ),
            pytest.param(
                "add",
                "/fake/path",
                "mock",
                [],
                [np.array([2.0])],
                id="empty_result_nps",
            ),
        ],
    )
    def test_skip_returns_none(self, api_name, plugin_path, spec_return, result_arr, golden_arr):
        """验证各跳过条件下 _try_custom_compare 返回 None。"""
        from ttk.core_modules.framework_api.profiling import _try_custom_compare

        case = MagicMock()
        case.api_name = api_name
        case.testcase_name = "test_0"
        sw = self._make_switches_with_path(plugin_path=plugin_path)
        spec_val = MagicMock() if spec_return == "mock" else spec_return
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=spec_val):
            assert _try_custom_compare(case, result_arr, golden_arr, sw) is None


class TestTryCustomCompareAdapt:
    """_try_custom_compare 返回值适配：dict/列表/嵌套、精度格式化、非法返回报错。"""

    def _make_case(self, api_name="add", output_dist=()):
        case = MagicMock()
        case.api_name = api_name
        case.testcase_name = "test_0"
        case.output_dist = output_dist
        return case

    def _make_switches(self):
        sw = _make_switches()
        sw.plugin_path = "/fake/path"
        return sw

    @pytest.mark.parametrize(
        "output_dist, compare_fn, result_arr, golden_arr, expected_p, expected_is_pass",
        [
            pytest.param(
                (),
                _compare_single_dict,
                [np.array([1.0, 2.0])],
                [np.array([1.0, 2.0])],
                "99.5%",
                True,
                id="single_dict",
            ),
        ],
    )
    def test_dict_list_return(self, output_dist, compare_fn, result_arr, golden_arr, expected_p, expected_is_pass):
        """compare 返回 dict 时的精度拼接与 pass 判定。"""
        from ttk.core_modules.framework_api.profiling import _try_custom_compare

        case = self._make_case(output_dist=output_dist)
        sw = self._make_switches()
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare_fn):
            p, log, is_pass = _try_custom_compare(case, result_arr, golden_arr, sw)
        assert p == expected_p
        assert is_pass == expected_is_pass

    @pytest.mark.parametrize(
        "compare_return, match, result_arr, golden_arr",
        [
            pytest.param(
                {"precision": 99.0},
                r"missing required key\(s\): 'pass'",
                [np.array([1.0])],
                [np.array([1.0])],
                id="missing_pass",
            ),
        ],
    )
    def test_invalid_return_raises(self, compare_return, match, result_arr, golden_arr):
        """compare 返回非法结构时抛出 ValueError。"""
        from ttk.core_modules.framework_api.profiling import _try_custom_compare

        def compare(*outputs):
            return compare_return

        case = self._make_case()
        sw = self._make_switches()
        with patch("ttk.core_modules.framework_api.profiling.get_spec_attr", return_value=compare):
            with pytest.raises(ValueError, match=match):
                _try_custom_compare(case, result_arr, golden_arr, sw)


class TestCustomCompareEndToEnd:
    """_evaluate_eager_precision 端到端：自定义 compare / 回退 / 失败路径。"""

    def test_evaluate_precision_uses_custom_compare(self):
        """Verify _evaluate_eager_precision uses custom compare when available."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            return {"pass": True, "precision": 100.0}

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch(
            "ttk.core_modules.framework_api.profiling.get_spec_attr",
            side_effect=lambda *a, **k: custom_compare if a[-2] == "compare" else None,
        ):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "100.0%"
        assert return_struct.precision_status == "PASS"

    def test_evaluate_precision_falls_back_to_builtin(self):
        """Verify _evaluate_eager_precision uses built-in compare when no custom compare."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = None  # no plugin path → no custom compare

        result = [np.array([1.0, 2.0])]
        golden = [np.array([1.0, 2.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        captured = {}

        def fake_compare(*args, **kwargs):
            captured["called"] = True
            return "100.0%", "log", True, {}

        with patch.object(prof_module, "compare", side_effect=fake_compare):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert captured.get("called") is True
        assert return_struct.eager_precision == "100.0%"

    def test_evaluate_precision_custom_compare_failure(self):
        """Verify custom compare exception → COMPARE_FAILURE."""
        from ttk.core_modules.framework_api import profiling as prof_module
        from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

        def custom_compare(*outputs):
            raise RuntimeError("custom boom")

        case = MagicMock()
        case.api_name = "add"
        case.testcase_name = "test_0"
        case.output_dist = ()
        case.precision_tolerances = None
        case.flat_precision_tolerances = None
        case.flat_absolute_precision = None

        sw = _make_switches()
        sw.plugin_path = "/fake/path"

        result = [np.array([1.0])]
        golden = [np.array([1.0])]
        return_struct = FrameworkApiReturnStructure()
        mock_perf = MagicMock()
        mock_perf.elapsed_us = 10.0
        mock_perf.kernel_details = None

        with patch(
            "ttk.core_modules.framework_api.profiling.get_spec_attr",
            side_effect=lambda *a, **k: custom_compare if a[-2] == "compare" else None,
        ):
            prof_module._evaluate_eager_precision(case, [], result, golden, sw, mock_perf, return_struct)

        assert return_struct.eager_precision == "COMPARE_FAILURE"
        assert return_struct.precision_status == "FAIL"
