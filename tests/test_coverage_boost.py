#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Coverage tests for:
- testcase_op.py: clear_atomic, tensor_dict, _do_shape_inference, _expand_indices,
  _auto_set_inplace_indexes, supported_rerun_title, hash_cases_to_groups
- op_interface.py: prepare_operator_parameters, construct_optiling_attrs
- static_compilation.py: static_compilation, stc_compile
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock

from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp


def _make_testcase(op_name="Add", input_shapes=((8,), (8,)),
                   input_dtypes=("float16", "float16"),
                   output_shapes=((8,),),
                   output_dtypes=("float16",),
                   **kwargs):
    case = TestcaseOp()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    case.input_ori_shapes = kwargs.pop("input_ori_shapes", input_shapes)
    case.output_ori_shapes = kwargs.pop("output_ori_shapes", output_shapes)
    case.attributes = kwargs.pop("attributes", {})
    n_in = len(input_shapes)
    n_out = len(output_shapes or ())
    case.input_formats = kwargs.pop("input_formats", ("ND",) * n_in)
    case.input_ori_formats = kwargs.pop("input_ori_formats", ("ND",) * n_in)
    case.output_formats = kwargs.pop("output_formats", ("ND",) * n_out)
    case.output_ori_formats = kwargs.pop("output_ori_formats", ("ND",) * n_out)
    case.input_data_ranges = kwargs.pop("input_data_ranges", (None,) * n_in)
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


@pytest.fixture(autouse=True)
def _mock_op_info(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


def _validate(case, op_info=None):
    if op_info is None:
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
    mock_gs = MagicMock()
    mock_gs.core_type = None
    mock_gs.DAVINCI_HBM_SIZE_LIMIT = 30
    mock_gs.op_impl_mode = None
    mock_gs.kernel_meta = "/tmp"
    mock_gs.short_soc_version = "Ascend910B2"
    with patch('ttk.core_modules.testcase_manager.testcase_op.get_global_storage', return_value=mock_gs), \
         patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock:
        mock.return_value.info_of.return_value = op_info
        case.validate()


def _mock_compile_result(has_kernel_json=True, clear_atomic=False):
    r = MagicMock()
    r.compile_result = "SUCC"
    r.kernel_name = "test_kernel"
    r.kernel_dir = "/tmp"
    r.tiling_key = 0
    r.block_dim = 1
    r.func_params = ("x",)
    r.tiling_result = MagicMock()
    r.tiling_result.tiling_time = (1.0,)
    r.tiling_result.all_set = MagicMock()
    if has_kernel_json:
        r.kernel_json_info = MagicMock()
        r.kernel_json_info.clear_atomic = clear_atomic
        r.kernel_json_info.optional_input_gen_placeholder.return_value = True
        r.kernel_json_info.optional_output_gen_placeholder.return_value = True
        r.kernel_json_info.dynamic_param_is_folded.return_value = False
        r.kernel_json_info.parameters = []
        r.kernel_json_info.workspaces = ()
        r.kernel_json_info.global_workspace_size = 0
        r.kernel_json_info.oom_enabled.return_value = False
    else:
        r.kernel_json_info = None
    r.workspaces = ()
    r.compile_time = 0.1
    return r


class TestClearAtomicProperties:

    def test_dyn_clear_atomic_from_compile_result(self):
        case = _make_testcase()
        case.dyn_compile_result = _mock_compile_result(clear_atomic=True)
        assert case.dyn_clear_atomic is True

    def test_dyn_clear_atomic_false(self):
        case = _make_testcase()
        case.dyn_compile_result = _mock_compile_result(clear_atomic=False)
        assert case.dyn_clear_atomic is False

    def test_dyn_clear_atomic_force_ops(self):
        case = _make_testcase(op_name="ones_like")
        case.dyn_compile_result = _mock_compile_result(clear_atomic=False)
        assert case.dyn_clear_atomic is True

    def test_cst_clear_atomic_from_override(self):
        case = _make_testcase()
        case._cst_clear_atomic = True
        assert case.cst_clear_atomic is True

    def test_cst_clear_atomic_default(self):
        case = _make_testcase()
        case.cst_clear_atomic = False
        assert case.cst_clear_atomic is False

    def test_bin_clear_atomic_from_compile_result(self):
        case = _make_testcase()
        case.bin_compile_result = _mock_compile_result(clear_atomic=True)
        assert case.bin_clear_atomic is True

    def test_clear_atomic_no_compile_result(self):
        case = _make_testcase()
        assert case.dyn_clear_atomic is False
        assert case.cst_clear_atomic is False
        assert case.bin_clear_atomic is False


class TestTensorDictProperties:

    def test_dyn_tensor_dict_construction(self):
        case = _make_testcase()
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert inputs[0]["shape"] == (-1,)
        assert inputs[0]["dtype"] == "float16"
        assert inputs[0]["format"] == "ND"
        assert inputs[0]["ori_shape"] == (-1,)
        assert outputs[0]["shape"] == (-1,)

    def test_dyn_tensor_dict_construction_2(self):
        case = _make_testcase()
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert inputs[0]["shape"] == (-1,)
        assert inputs[0]["ori_shape"] == (-1,)

    def test_tensor_dict_with_none_shape(self):
        case = _make_testcase(input_shapes=(None, (8,)), input_dtypes=("float16", "float16"))
        _validate(case)
        inputs, _ = case.dyn_tensor_dict
        assert inputs[0] is None
        assert inputs[1]["shape"] == (-1,)

    def test_tensor_dict_cached(self):
        case = _make_testcase()
        _validate(case)
        td1 = case.dyn_tensor_dict
        td2 = case.dyn_tensor_dict
        assert td1 is td2

    def test_bin_tensor_dict_no_const(self):
        case = _make_testcase()
        _validate(case)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            inputs, outputs = case.bin_tensor_dict
        assert inputs[0]["shape"] == (-2,)
        assert inputs[1]["shape"] == (-2,)


class TestShapeInference:

    def test_elewise_inference(self):
        result = TestcaseOp._do_shape_inference(
            ((8,), (8,)), "ELEWISE", {})
        assert result == ((8,),)

    def test_reduce_inference_with_axis(self):
        result = TestcaseOp._do_shape_inference(
            ((8, 16),), "REDUCE", {"axis": (1,)})
        assert result is not None

    def test_reduce_inference_with_axes(self):
        result = TestcaseOp._do_shape_inference(
            ((8, 16, 32),), "REDUCE", {"axes": (1, 2)})
        assert result is not None

    def test_reduce_inference_no_axis(self):
        result = TestcaseOp._do_shape_inference(
            ((8, 16),), "REDUCE", {})
        assert result is not None

    def test_elewise_with_args(self):
        result = TestcaseOp._do_shape_inference(
            ((8,), (8,)), "ELEWISE(1, None)", {})
        assert result is not None

    def test_none_in_inputs_raises(self):
        with pytest.raises(ValueError, match="None input"):
            TestcaseOp._do_shape_inference(
                (None,), "ELEWISE", {})

    def test_invalid_inference_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            TestcaseOp._do_shape_inference(
                ((8,),), "INVALID_TYPE", {})


class TestExpandIndices:

    def test_simple_expand(self):
        result = TestcaseOp._expand_indices(3, (), [0, 1, 2])
        assert result == [0, 1, 2]

    def test_with_none(self):
        result = TestcaseOp._expand_indices(3, (), [0, None, 2])
        assert result == [0, None, 2]

    def test_with_list_distribution(self):
        result = TestcaseOp._expand_indices(5, (2, 0, 0), [0, 1])
        assert len(result) == 3


class TestAutoSetInplaceIndexes:

    def test_no_inplace(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.output_inplace_indexes is None or case.output_inplace_indexes == ()

    def test_with_inplace(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "paramType": "default"},
                       {"name": "y", "paramType": "default"}],
            "outputs": [{"name": "x", "paramType": "default"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.output_inplace_indexes == (0,)

    def test_param_type_mismatch(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "paramType": "dynamic"},
                       {"name": "y", "paramType": "default"}],
            "outputs": [{"name": "x", "paramType": "default"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.fail_reason == "OUTPUT_INPLACE_INDEXES_ERROR"


class TestConstInputIndexesOpInfo:

    def test_value_depend_required(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"},
                       {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == (0,)

    def test_value_depend_optional(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "optional"},
                       {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == (0,)

    def test_no_value_depend(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "ignore"},
                       {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == ()

    def test_no_op_info(self):
        case = _make_testcase()
        _validate(case)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            assert case.const_input_indexes == ()

    def test_const_inputs_never_removed(self):
        case = _make_testcase(input_shapes=((1,), (1,)))
        _validate(case)
        assert case.input_shapes == ((1,), (1,))


class TestSupportedRerunTitle:

    def test_returns_expected_titles(self):
        titles = TestcaseOp.supported_rerun_title()
        assert "dyn_perf_us" in titles
        assert "precision_status" in titles


class TestHashCasesToGroups:

    def test_empty_input(self):
        result = TestcaseOp.hash_cases_to_groups(set())
        assert result == {}

    def test_groups_by_hash(self):
        case1 = _make_testcase()
        case2 = _make_testcase()
        _validate(case1)
        _validate(case2)
        result = TestcaseOp.hash_cases_to_groups({case1, case2})
        assert len(result) == 1
        assert len(list(result.values())[0]) == 2


class TestSetCaseCoreType:

    def test_core_type_from_global_storage(self):
        case = _make_testcase()
        _validate(case)
        assert case.core_type == "AiCore"

    def test_core_type_from_op_info(self):
        op_info = {
            "coreType.value": "VectorCore",
            "inputs": [{"name": "x"}],
            "outputs": [{"name": "y"}],
        }
        case = _make_testcase()
        _validate(case, op_info=op_info)
        assert case.core_type == "VectorCore"

    def test_core_type_default_when_no_op_info(self):
        case = _make_testcase()
        _validate(case, op_info=None)
        assert case.core_type == "AiCore"


class TestStcShapeSizeCheck:

    def test_shape_out_of_bound(self):
        case = _make_testcase(input_shapes=((999999999,),), input_dtypes=("float32",))
        mock_gs = MagicMock()
        mock_gs.core_type = None
        mock_gs.DAVINCI_HBM_SIZE_LIMIT = 0
        mock_gs.op_impl_mode = None
        mock_gs.kernel_meta = "/tmp"
        mock_gs.short_soc_version = "Ascend910B2"
        with patch('ttk.core_modules.testcase_manager.testcase_op.get_global_storage', return_value=mock_gs), \
             patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = {
                "coreType.value": "AiCore",
                "inputs": [{"name": "x"}],
                "outputs": [{"name": "z"}],
            }
            case.validate()
        assert not case.is_valid
        assert case.fail_reason == "STC_SHAPE_OUT_OF_BOUND"


class TestGetDynFuncParamName:

    def test_basic(self):
        case = _make_testcase()
        case.dyn_func_params = ("x", "y", "axis")
        result = case.get_dyn_func_param_name(2)
        assert result == "axis"

    def test_with_list_distribution(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        case.dyn_func_params = ("inputs", "axis")
        result = case.get_dyn_func_param_name(1)
        assert result == "inputs"


# ============= op_interface.py tests =============

from ttk.core_modules.operator.op_interface import (
    OperatorInterface, OperatorNotFoundError
)


@pytest.fixture
def _clear_caches():
    pass

class TestPrepareOperatorParameters:

    def test_dyn_mode(self):
        case = _make_testcase()
        _validate(case)
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.op_output_defined.return_value = True
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "dyn")
        assert len(ipt) == 2
        assert ipt[0]["shape"] == (-1,)

    def test_bin_mode(self):
        case = _make_testcase()
        _validate(case)
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi, \
             patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock_oi2:
            mock_oi.return_value.op_output_defined.return_value = True
            mock_oi2.return_value.info_of.return_value = None
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "bin")
        assert len(ipt) == 2

    def test_unsupported_mode_raises(self):
        case = _make_testcase()
        with pytest.raises(RuntimeError, match="not supported"):
            OperatorInterface.prepare_operator_parameters(case, "invalid")

    def test_op_output_not_defined(self):
        case = _make_testcase()
        _validate(case)
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.op_output_defined.return_value = False
            ipt, opt = OperatorInterface.prepare_operator_parameters(case, "dyn")
        assert opt == ()


class TestConstructOptilingAttrs:

    def test_from_attr_dictionary_no_op_info(self):
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"axis": 1, "keep_dims": True})
        assert len(result) == 2
        assert result[0]["name"] == "axis"
        assert result[0]["dtype"] == "int"

    def test_from_op_info(self):
        op_info = {
            "attr": [
                {"name": "axis", "type": "Int", "defaultValue": None},
                {"name": "keep_dims", "type": "Bool", "defaultValue": False},
            ]
        }
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            result = OperatorInterface.construct_optiling_attrs("Add", {"axis": 1})
        assert len(result) == 2
        assert result[0]["value"] == 1

    def test_required_attr_missing_raises(self):
        op_info = {
            "attr": [
                {"name": "axis", "type": "Int", "defaultValue": None},
            ]
        }
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            with pytest.raises(RuntimeError, match="Required attribute"):
                OperatorInterface.construct_optiling_attrs("Add", {})

    def test_private_attrs(self):
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"@private_key": 42})
        assert any(r["name"] == "private_key" for r in result)

    def test_skip_special_prefixes(self):
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = None
            result = OperatorInterface.construct_optiling_attrs("Add", {"!skip": 1, "#skip": 2, "keep": 3})
        assert len(result) == 1
        assert result[0]["name"] == "keep"


class TestGetOpFuncParams:

    def test_from_callable(self):
        def my_op(x, y, axis):
            pass
        result = OperatorInterface.get_op_func_params(operator_func=my_op)
        assert result == ("x", "y", "axis")

    def test_from_op_name(self):
        op_info = {
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
            "attr": [{"name": "axis"}],
        }
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as mock_oi:
            mock_oi.return_value.info_of.return_value = op_info
            result = OperatorInterface.get_op_func_params(op_name="Add")
        assert result == ("x", "y", "z", "axis")

    def test_no_args_returns_empty(self):
        result = OperatorInterface.get_op_func_params()
        assert result == ()


class TestOpTypeFromSourceCode:

    def test_not_callable(self):
        assert OperatorInterface.get_op_type_from_source_code("not_a_func") is None


class TestEnableShapeInt64:

    def test_small_shape(self):
        tensors = [{"shape": (8,), "dtype": "float16"}]
        assert OperatorInterface._enable_shape_int64(tensors) is False

    def test_none_tensor(self):
        tensors = [None]
        assert OperatorInterface._enable_shape_int64(tensors) is False

    def test_list_of_tensors(self):
        tensors = [[{"shape": (8,), "dtype": "float16"}]]
        assert OperatorInterface._enable_shape_int64(tensors) is False


class TestPrintFuncParams:

    def test_basic(self):
        result = OperatorInterface.print_func_params(("x", "y"), {"axis": 1}, [{"shape": (8,)}])
        assert len(result) > 2
        assert "***Params***" in result


class TestAddCompileInfoToOpContext:

    def test_dict_context(self):
        cxt = {}
        case = _make_testcase()
        case.core_type = "AiCore"
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        assert cxt["_sgt_cube_vector_core_type"] == "AiCore"

    def test_dict_context_with_hash_param(self):
        cxt = {}
        case = _make_testcase(attributes={"#custom_key": "value"})
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        assert cxt["custom_key"] == "value"

    def test_object_context(self):
        cxt = MagicMock()
        case = _make_testcase(attributes={"#custom_key": "value"})
        OperatorInterface.add_compile_info_to_op_context(cxt, case)
        cxt.add_compile_info.assert_called()


# ============= static_compilation.py tests =============

class TestStaticCompilation:

    @patch('ttk.core_modules.npu.op.compilation.static_compilation.get_process_context')
    @patch('ttk.core_modules.npu.op.compilation.static_compilation.get_global_storage')
    @patch('ttk.core_modules.npu.op.compilation.static_compilation.OperatorInterface')
    def test_switch_disabled(self, mock_oi_cls, mock_gs, mock_pc):
        mock_gs.return_value.kernel_meta = "/tmp/test_kernel_meta"
        mock_sw = MagicMock()
        mock_sw.enabled = False
        mock_gs.return_value.cst_switches = mock_sw
        mock_pc.return_value.notify_status = MagicMock()

        from ttk.core_modules.npu.op.compilation.static_compilation import static_compilation
        case = _make_testcase()
        result = static_compilation(case, "Cst")
        assert result.compile_result == "CST_OFF"

    @patch('ttk.core_modules.npu.op.compilation.static_compilation.get_process_context')
    @patch('ttk.core_modules.npu.op.compilation.static_compilation.get_global_storage')
    @patch('ttk.core_modules.npu.op.compilation.static_compilation.OperatorInterface')
    def test_cst_dyn_invalid(self, mock_oi_cls, mock_gs, mock_pc):
        mock_gs.return_value.kernel_meta = "/tmp/test_kernel_meta"
        mock_sw = MagicMock()
        mock_sw.enabled = True
        mock_sw.realtime = True
        mock_gs.return_value.cst_switches = mock_sw
        mock_pc.return_value.notify_status = MagicMock()

        from ttk.core_modules.npu.op.compilation.static_compilation import static_compilation
        case = _make_testcase()
        case.is_valid = False
        case.fail_reason = "DYN_INPUT_MISSING"
        result = static_compilation(case, "Cst")
        assert result.compile_result == "DYN_INPUT_MISSING"

class TestNormalizeMode:

    def test_lowercase(self):
        from ttk.core_modules.npu.op.compilation.common import normalize_mode
        assert normalize_mode("cst") == "Cst"

    def test_uppercase(self):
        from ttk.core_modules.npu.op.compilation.common import normalize_mode
        assert normalize_mode("CST") == "Cst"

    def test_invalid_raises(self):
        from ttk.core_modules.npu.op.compilation.common import normalize_mode
        with pytest.raises(NotImplementedError):
            normalize_mode("invalid_long")
