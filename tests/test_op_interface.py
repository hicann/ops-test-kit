#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Coverage boost for ttk.core_modules.operator.op_interface
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_op import TestcaseOp
import ttk.core_modules.operator.tbe_interface as _tbe_mod
_tbe_mod.Opc = MagicMock

import ttk.core_modules.operator.op_interface as oi_mod
from ttk.core_modules.operator.op_interface import (
    OperatorInterface, OperatorNotFoundError
)


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


def _mock_opc():
    opc = MagicMock()
    opc.op_info.OpInfo.return_value = MagicMock()
    opc.build_config.return_value.__enter__ = MagicMock(return_value=None)
    opc.build_config.return_value.__exit__ = MagicMock(return_value=False)
    opc.api_config.bit_width_32.return_value.__enter__ = MagicMock(return_value=None)
    opc.api_config.bit_width_32.return_value.__exit__ = MagicMock(return_value=False)
    opc.api_config.bit_width_64.return_value.__enter__ = MagicMock(return_value=None)
    opc.api_config.bit_width_64.return_value.__exit__ = MagicMock(return_value=False)
    opc.op_context.OpContext.return_value.__enter__ = MagicMock(return_value=None)
    opc.op_context.OpContext.return_value.__exit__ = MagicMock(return_value=False)
    opc.get_computes.return_value = []
    opc.get_compile_info.return_value = {}
    opc.get_tiling_op_type.return_value = "Add"
    return opc


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


@pytest.fixture(autouse=True)
def _mock_singleton():
    p = patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper')
    mock_cls = p.start()
    mock_cls.return_value.info_of.return_value = None
    with patch('ttk.core_modules.operator.op_interface.Opc', return_value=_mock_opc()):
        oi = OperatorInterface()
        oi._opc = _mock_opc()
    yield oi
    p.stop()


class TestWithCoreType:

    def test_sets_core_type(self, _mock_singleton):
        oi = _mock_singleton
        result = oi.with_core_type("VectorCore")
        assert result is oi


class TestGetOpGeneralizeFunc:

    def test_empty_op_type(self, _mock_singleton):
        assert _mock_singleton.get_op_generalize_func("") is None

    def test_none_op_type(self, _mock_singleton):
        assert _mock_singleton.get_op_generalize_func(None) is None

    def test_valid_op_type(self, _mock_singleton):
        _mock_singleton._opc.get_param_generalization.return_value = "func"
        assert _mock_singleton.get_op_generalize_func("Add") == "func"


class TestSwitchOpc:

    def test_switch_tbe(self, _mock_singleton):
        _mock_singleton._switch_opc("tbe")
        _mock_singleton._opc.switch_opc.assert_called_with("tbe")

    def test_switch_invalid(self, _mock_singleton):
        _mock_singleton._switch_opc("invalid")
        _mock_singleton._opc.switch_opc.assert_not_called()


class TestEnableShapeInt64:

    def test_shape_exceeds_int32(self):
        big = np.iinfo(np.int32).max + 1
        assert OperatorInterface._enable_shape_int64([{"shape": (big,), "dtype": "float16"}]) is True

    def test_dtype_bytes_exceeds_int32(self):
        assert OperatorInterface._enable_shape_int64([{"shape": (65536, 32768), "dtype": "float32"}]) is True

    def test_shape_prod_zero(self):
        assert OperatorInterface._enable_shape_int64([{"shape": (0,), "dtype": "float16"}]) is False

    def test_nested_list(self):
        assert OperatorInterface._enable_shape_int64([[{"shape": (8,), "dtype": "float16"}]]) is False


class TestDtypeStrToType:

    def test_list_type(self):
        assert OperatorInterface.dtype_str_to_type("listInt") is list

    def test_int_type(self):
        assert OperatorInterface.dtype_str_to_type("int") == int


class TestGetOpTypeFromSourceCode:

    def test_with_register_operator(self):
        func = MagicMock()
        with patch('inspect.getsource', return_value='@register_operator("AddOp")\ndef add(x, y): pass'):
            assert OperatorInterface.get_op_type_from_source_code(func) == "AddOp"

    def test_os_error(self):
        func = MagicMock()
        with patch('inspect.getsource', side_effect=OSError):
            assert OperatorInterface.get_op_type_from_source_code(func) is None


class TestGetOpFuncParams:

    def test_runtime_error(self):
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            with pytest.raises(RuntimeError, match="not configured"):
                OperatorInterface.get_op_func_params(op_name="MissingOp")


class TestGetOpFuncParameterDict:

    def test_from_op_info(self):
        op_info = {
            "inputs": [{"name": "x", "paramType": "required"},
                       {"name": "y", "paramType": "optional"}],
            "outputs": [{"name": "z"}],
            "attr": [{"name": "axis", "type": "int", "defaultValue": 1},
                     {"name": "keep_dims", "type": "bool", "defaultValue": None}],
        }
        mock_oik = MagicMock()
        mock_oik.info_of.return_value = op_info
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper', return_value=mock_oik):
            result = OperatorInterface.get_op_func_parameter_dict(op_name="FakeOp_param")
        assert len(result) == 5
        assert result["y"].default is None
        assert result["axis"].default == 1

    def test_returns_none(self):
        assert OperatorInterface.get_op_func_parameter_dict() is None


class TestAddAdditionToOpContext:

    def test_basic(self):
        cxt = MagicMock()
        op_info = MagicMock(op_name="Add")
        OperatorInterface.add_addition_to_op_context(cxt, {"!key": "val"}, op_info)
        cxt.add_addition.assert_any_call("op_name", "Add")
        cxt.add_addition.assert_any_call("key", "val")


class TestAddPrivateAttrToOpInfo:

    def test_no_private_attrs(self):
        op_info = MagicMock(spec=[])
        OperatorInterface.add_private_attr_to_op_info((), {}, op_info)

    def test_with_private_attrs(self):
        op_info = MagicMock()
        OperatorInterface.add_private_attr_to_op_info(
            ({"name": "axis"}, {"name": "scale"}), {"@scale": 0.5}, op_info)
        op_info.private_attrs = {"name": "scale"}


class TestConstructCompileContextOpInfo:

    def test_with_op_type(self, _mock_singleton):
        oi = _mock_singleton
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {})
        oi._opc.op_info.OpInfo.assert_called_with("AddOp", "AddOp")

    def test_unknown_op_type(self, _mock_singleton):
        oi = _mock_singleton
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m, \
             patch.object(OperatorInterface, 'get_op_type_from_source_code', return_value=None):
            m.return_value.op_type_of.return_value = None
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {})
        oi._opc.op_info.OpInfo.assert_called_with("UNKNOWN", "UNKNOWN")

    def test_with_impl_mode(self, _mock_singleton):
        oi = _mock_singleton
        mock_oi = MagicMock()
        oi._opc.op_info.OpInfo.return_value = mock_oi
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi._construct_compile_context_op_info(lambda: None, "Add", "kernel", {"impl_mode": "hp"})
        assert mock_oi.precision_mode == "hp"


class TestCompileOp:

    def test_success(self, _mock_singleton):
        oi = _mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[])
        func = MagicMock()
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs):
            result = oi._compile_op("Dyn", "Add", func, ("x",), [], {})
        assert isinstance(result, float)

    def test_failure_raises(self, _mock_singleton):
        oi = _mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[])
        func = MagicMock(side_effect=RuntimeError("err"))
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs):
            with pytest.raises(RuntimeError):
                oi._compile_op("Dyn", "Add", func, ("x",), [], {})

    def test_non_callable(self, _mock_singleton):
        oi = _mock_singleton
        mock_gs = MagicMock(kernel_compile_options=[])
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs):
            with pytest.raises(RuntimeError, match="not callable"):
                oi._compile_op("Cst", "Add", "str", ("x",), [], {})


class TestSetCommonCompileContext:

    def test_basic(self, _mock_singleton):
        oi = _mock_singleton
        cxt = MagicMock()
        cxt.get_op_mode.return_value = "dynamic"
        case = _make_testcase()
        case.kb_pid = 42
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.op_type_of.return_value = "AddOp"
            oi.set_common_compile_context(cxt, case, MagicMock(), "kernel")
        cxt.add_addition.assert_any_call("master_pid", 42)


class TestSetDynamicCompileContext:

    def test_static_mode(self, _mock_singleton):
        oi = _mock_singleton
        cxt = MagicMock()
        cxt.get_op_mode.return_value = "static"
        mock_op_info = MagicMock()
        cxt.get_op_info.return_value = [MagicMock(), mock_op_info]
        case = _make_testcase(attributes={"axis": 1})
        dyn_params = ({"shape": (-1,)},) * 3
        with patch.object(oi, 'set_common_compile_context'), \
             patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            oi.set_dynamic_compile_context(cxt, case, MagicMock(), "kernel", dyn_params)
        assert mock_op_info.inputs is not None


class TestGetDynOperator:

    def test_not_found(self, _mock_singleton):
        oi = _mock_singleton
        with patch('ttk.core_modules.operator.op_interface.OpInfoKeeper') as m:
            m.return_value.get_operator_function.return_value = None
            assert oi.get_dyn_operator(_make_testcase()) is None


class TestCompileDynamicShape:

    def test_operator_not_found(self, _mock_singleton):
        oi = _mock_singleton
        with patch.object(oi, 'get_dyn_operator', return_value=None):
            assert oi.compile_dynamic_shape((), _make_testcase(), "k") is None

    def test_compile_success(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase()
        mock_gs = MagicMock(auto_switch=False, kernel_compile_options=[])
        dyn_params = ({"shape": (-1,), "dtype": "float16"}, {"shape": (-1,), "dtype": "float16"},
                      {"shape": (-1,), "dtype": "float16"})
        with patch.object(oi, 'get_dyn_operator', return_value=MagicMock()), \
             patch.object(OperatorInterface, 'get_op_func_params', return_value=("x",)), \
             patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs), \
             patch('ttk.core_modules.operator.op_interface.OpInfoKeeper'), \
             patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock_oi2, \
             patch.object(oi, 'set_dynamic_compile_context'), \
             patch.object(oi, '_compile_op', return_value=0.1), \
             patch.object(OperatorInterface, 'construct_optiling_attrs', return_value=()):
            mock_oi2.return_value.info_of.return_value = None
            result = oi.compile_dynamic_shape(dyn_params, case, "k")
        assert result is not None



class TestPrepareOperatorParametersConst:

    def _run(self, oi, case, op_info=None, op_output_defined=False,
             func_params=("x", "y"), param_transform=None):
        """Helper to run prepare_operator_parameters_const with mocks."""
        mocks = [
            patch.object(oi, 'get_dyn_operator', return_value=MagicMock()),
            patch.object(OperatorInterface, 'get_op_func_params', return_value=func_params),
            patch('ttk.core_modules.operator.op_interface.OpInfoKeeper'),
            patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper'),
        ]
        if param_transform is not None:
            mocks.append(
                patch('ttk.core_modules.operator.op_interface.param_transformation',
                      return_value=param_transform))
        with mocks[0] as m_op, mocks[1] as m_fp, mocks[2] as m_oi, mocks[3] as m_oi2:
            m_oi.return_value.op_output_defined.return_value = op_output_defined
            m_oi2.return_value.info_of.return_value = op_info
            if param_transform is not None:
                with mocks[4]:
                    return oi.prepare_operator_parameters_const(case)
            return oi.prepare_operator_parameters_const(case)

    def test_operator_not_found(self, _mock_singleton):
        oi = _mock_singleton
        with patch.object(oi, 'get_dyn_operator', return_value=None):
            with pytest.raises(OperatorNotFoundError):
                oi.prepare_operator_parameters_const(_make_testcase())

    def test_normal_inputs_no_output(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=False)
        assert len(ipt) == 2
        assert ipt[0]["shape"] == (3, 4)
        assert ipt[1]["shape"] == (5, 6)
        assert opt == ()

    def test_normal_inputs_with_output(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True)
        assert len(ipt) == 2
        assert len(opt) == 1
        assert opt[0]["shape"] == (3, 6)
        assert opt[0]["dtype"] == "float16"

    def test_none_input(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(input_shapes=(None, (3, 4)))
        ipt, opt = self._run(oi, case, op_output_defined=False)
        assert ipt[0] is None
        assert ipt[1]["shape"] == (3, 4)

    def test_const_input(self, _mock_singleton):
        oi = _mock_singleton
        op_info = {
            "inputs": [{"name": "x"}, {"name": "y", "valueDepend": "required"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=((3, 4), (2,)),
            attributes={"y": [1, 2]})
        ipt, opt = self._run(oi, case, op_info=op_info, op_output_defined=False,
                             param_transform={"y": [1, 2]})
        assert "const_value" in ipt[1]
        assert ipt[1]["name"] == "y"
        assert ipt[1]["dtype"] == "float16"
        assert ipt[0]["shape"] == (3, 4)

    def test_tl_input(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=((7, 6),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True,
                             func_params=("x", "y"))
        assert isinstance(ipt[0], tuple)
        assert len(ipt[0]) == 2
        assert ipt[0][0]["shape"] == (3, 4)
        assert ipt[0][0]["dtype"] == "float16"
        assert ipt[0][1]["shape"] == (5, 6)
        assert ipt[0][1]["dtype"] == "float16"
        assert ipt[1]["shape"] == (7, 8)

    def test_tl_output(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(
            input_shapes=((3, 4),),
            input_dtypes=("float16",),
            output_shapes=(((2, 3), (4, 5)),),
            output_dtypes=(("float16", "float16"),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True,
                             func_params=("x",))
        assert isinstance(opt[0], tuple)
        assert len(opt[0]) == 2
        assert opt[0][0]["shape"] == (2, 3)
        assert opt[0][1]["shape"] == (4, 5)

    def test_const_plus_tl_mixed(self, _mock_singleton):
        oi = _mock_singleton
        op_info = {
            "inputs": [{"name": "x", "valueDepend": "required"},
                       {"name": "y"},
                       {"name": "z"}],
            "outputs": [{"name": "out"}],
        }
        case = _make_testcase(
            input_shapes=((2, 3), ((3, 4), (5, 6)), (7, 8)),
            input_dtypes=("float16", ("float16", "float16"), "float16"),
            output_shapes=((7, 6),),
            attributes={"x": [[1, 2, 3], [4, 5, 6]]},
        )
        ipt, opt = self._run(oi, case, op_info=op_info, op_output_defined=False,
                             func_params=("x", "y", "z"),
                             param_transform={"x": [[1, 2, 3], [4, 5, 6]]})
        # pos0: const
        assert "const_value" in ipt[0]
        assert ipt[0]["name"] == "x"
        # pos1: TL (non-const)
        assert isinstance(ipt[1], tuple)
        assert len(ipt[1]) == 2
        assert ipt[1][0]["shape"] == (3, 4)
        assert ipt[1][1]["shape"] == (5, 6)
        # pos2: non-const normal
        assert ipt[2]["shape"] == (7, 8)

    def test_tl_input_and_tl_output(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((2, 3), (4, 5)),),
            output_dtypes=(("float16", "float16"),),
        )
        ipt, opt = self._run(oi, case, op_output_defined=True,
                             func_params=("x", "y"))
        assert isinstance(ipt[0], tuple)
        assert len(ipt[0]) == 2
        assert isinstance(opt[0], tuple)
        assert len(opt[0]) == 2
        assert opt[0][0]["shape"] == (2, 3)
        assert opt[0][1]["shape"] == (4, 5)


class TestPrepareTilingParams:

    def test_basic(self, _mock_singleton):
        oi = _mock_singleton
        case = _make_testcase()
        with patch.object(OperatorInterface, 'construct_optiling_attrs', return_value=()), \
             patch.object(oi, 'prepare_operator_parameters_const',
                          return_value=(({"shape": (8,), "range": (None, None)},), ({"shape": (8,)},))):
            inputs, outputs, attrs = oi.prepare_tiling_params(case)
        assert len(attrs) == 0


class TestCallConstOpTiling:

    def test_success(self, _mock_singleton):
        oi = _mock_singleton
        mock_cr = MagicMock(tiling_op_type="Add", compile_info={})
        oi._opc.do_op_tiling.return_value = {"tiling_key": 0}
        mock_gs = MagicMock(tiling_run_time=1)
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs), \
             patch.object(oi, 'prepare_tiling_params',
                          return_value=(({"shape": (8,)},), ({"shape": (8,)},), ())), \
             patch('ttk.core_modules.operator.op_interface.adapter_before_tiling'):
            result = oi.call_const_op_tiling(mock_cr, _make_testcase())
        assert "tiling_time" in result

    def test_tiling_failure(self, _mock_singleton):
        oi = _mock_singleton
        mock_cr = MagicMock(tiling_op_type="Add", compile_info={})
        oi._opc.do_op_tiling.side_effect = RuntimeError("some error")
        mock_gs = MagicMock(tiling_run_time=1)
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs), \
             patch.object(oi, 'prepare_tiling_params',
                          return_value=(({"shape": (8,)},), ({"shape": (8,)},), ())), \
             patch('ttk.core_modules.operator.op_interface.adapter_before_tiling'), \
             patch('os.path.expanduser', return_value="/tmp/plog"), \
             patch('glob.glob', return_value=[]):
            with pytest.raises(RuntimeError, match="OPTILING_FAILURE"):
                oi.call_const_op_tiling(mock_cr, _make_testcase())

    def test_undefined_symbol(self, _mock_singleton):
        oi = _mock_singleton
        mock_cr = MagicMock(tiling_op_type="Add", compile_info={})
        oi._opc.do_op_tiling.side_effect = RuntimeError("undefined symbol in foo")
        mock_gs = MagicMock(tiling_run_time=1)
        with patch('ttk.core_modules.operator.op_interface.get_global_storage', return_value=mock_gs), \
             patch.object(oi, 'prepare_tiling_params',
                          return_value=(({"shape": (8,)},), ({"shape": (8,)},), ())), \
             patch('ttk.core_modules.operator.op_interface.adapter_before_tiling'):
            with pytest.raises(RuntimeError, match="undefined symbol"):
                oi.call_const_op_tiling(mock_cr, _make_testcase())


class TestAdapterBeforeTiling:

    def test_non_conv2d(self):
        from ttk.core_modules.operator.op_interface import adapter_before_tiling
        case = _make_testcase()
        cr = MagicMock(compile_info={"tiling_type": "binary", "op_type_list": ["Matmul"]})
        adapter_before_tiling(case, cr, (), ())

    def test_conv2d_nc1hwc0(self):
        from ttk.core_modules.operator.op_interface import adapter_before_tiling
        case = _make_testcase(attributes={"groups": 1})
        cr = MagicMock(compile_info={"tiling_type": "binary", "op_type_list": ["Conv2D"]})
        fi = [None, {"format": "NC1HWC0", "shape": (1, 1, 8, 8, 16)}]
        adapter_before_tiling(case, cr, fi, [None])
        assert fi[1]["format"] == "FRACTAL_Z"

    def test_conv2d_nchw(self):
        from ttk.core_modules.operator.op_interface import adapter_before_tiling
        case = _make_testcase(attributes={"groups": 1})
        cr = MagicMock(compile_info={"tiling_type": "binary", "op_type_list": ["Conv2D"]})
        fi = [None, {"format": "NCHW", "shape": (16, 3, 3, 3)}]
        adapter_before_tiling(case, cr, fi, [None])
        assert fi[1]["format"] == "FRACTAL_Z"
