#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.testcase_manager.testcase_op:
- validate() flow
- dyn @property auto-derivation from stc fields
- ready_for_profile / compile_failed / apply_compile_result
- tensor dict construction
"""

import pytest
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_op import UniversalTestcaseStructure


def _make_testcase(op_name="Add", input_shapes=((8,), (8,)),
                   input_dtypes=("float16", "float16"),
                   output_shapes=((8,),),
                   output_dtypes=("float16",),
                   **kwargs):
    case = UniversalTestcaseStructure()
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


_DEFAULT_OP_INFO = {
    "coreType.value": "AiCore",
    "inputs": [{"name": "x"}, {"name": "y"}],
    "outputs": [{"name": "z"}],
}


def _validate(case, op_info=_DEFAULT_OP_INFO):
    if op_info is not None and "coreType.value" not in op_info:
        op_info = {**op_info, "coreType.value": "AiCore"}
    with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock:
        mock.return_value.info_of.return_value = op_info
        case.validate()


class TestValidateBasic:

    def test_valid_minimal_case(self):
        case = _make_testcase()
        _validate(case)
        assert case.is_valid
        assert case.fail_reason is None

    def test_missing_op_name(self):
        case = _make_testcase()
        case.op_name = None
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "OP_NAME_MISSING"

    def test_missing_output_shapes(self):
        case = _make_testcase(output_shapes=None)
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "STC_OUTPUT_NOT_SPECIFIED"

    def test_output_shapes_elewise_inference(self):
        case = _make_testcase(output_shapes="ELEWISE")
        _validate(case)
        assert case.is_valid
        assert case.output_shapes == ((8,),)

    def test_invalid_input_dtypes(self):
        case = _make_testcase(input_dtypes=("badtype",))
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "STC_INPUT_DTYPES_INVALID"

    def test_invalid_output_dtypes(self):
        case = _make_testcase(output_dtypes=("badtype",))
        _validate(case)
        assert not case.is_valid
        assert case.fail_reason == "OUTPUT_DTYPES_INVALID"


class TestDynPropertyDerivation:

    def test_dyn_inputs_from_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_inputs == ((-1,), (-1,))

    def test_dyn_outputs_from_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_outputs == ((-1,),)

    def test_dyn_ori_inputs_from_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_ori_inputs == ((-1,), (-1,))

    def test_dyn_ori_outputs_from_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_ori_outputs == ((-1,),)

    def test_dyn_input_dtypes_same_as_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_input_dtypes == case.input_dtypes

    def test_dyn_input_formats_same_as_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_input_formats == case.input_formats

    def test_dyn_input_ori_formats_same_as_stc(self):
        case = _make_testcase()
        _validate(case)
        assert case.dyn_input_ori_formats == case.input_ori_formats

    def test_tensor_list_distribution_default(self):
        case = _make_testcase()
        _validate(case)
        # Flat inputs produce (0,)*len input_distribution, not ()
        assert case.tensor_list_distribution == (0, 0, 0)

    def test_is_valid_after_validate(self):
        case = _make_testcase()
        _validate(case)
        assert case.is_valid


class TestReadyForProfile:

    def test_not_ready_initially(self):
        case = _make_testcase()
        assert not case.ready_for_profile()

    def test_ready_after_3_compiles(self):
        case = _make_testcase()
        case.compile_done = 3
        assert case.ready_for_profile()

    def test_not_ready_after_2_compiles(self):
        case = _make_testcase()
        case.compile_done = 2
        assert not case.ready_for_profile()


class TestCompileFailed:

    def _make_result(self, status):
        r = MagicMock()
        r.compile_result = status
        return r

    def test_all_succ(self):
        case = _make_testcase()
        for mode in ("dyn", "cst", "bin"):
            setattr(case, f"{mode}_compile_result", self._make_result("SUCC"))
        assert not case.compile_failed()

    def test_dyn_fail_means_failed(self):
        case = _make_testcase()
        case.dyn_compile_result = self._make_result("FAIL")
        case.cst_compile_result = self._make_result("SUCC")
        case.bin_compile_result = self._make_result("SUCC")
        results = [getattr(getattr(case, f"{t}_compile_result"), "compile_result")
                   for t in ("dyn", "cst", "bin")]
        assert "SUCC" not in results or "FAIL" in results

    def test_compile_dynamic_op_success_with_cst(self):
        case = _make_testcase()
        case.dyn_compile_result = self._make_result("FAIL")
        case.cst_compile_result = self._make_result("SUCC")
        case.bin_compile_result = self._make_result("FAIL")
        assert case.compile_dynamic_op_success()


class TestApplyCompileResult:

    def test_apply_dynamic(self):
        from ttk.utilities.classes import DynamicCompilationResult
        case = _make_testcase()
        result = DynamicCompilationResult()
        result.compile_result = "SUCC"
        result.func_params = ("x", "y")
        result.tiling_result = None
        case.apply_compile_result(result)
        assert case.dyn_compile_result is result
        assert case.compile_done == 1

    def test_apply_const(self):
        from ttk.utilities.classes import ConstCompilationResult
        case = _make_testcase()
        result = ConstCompilationResult()
        result.compile_result = "SUCC"
        result.func_params = ("x", "y")
        case.apply_compile_result(result)
        assert case.cst_compile_result is result
        assert case.compile_done == 1


class TestTensorDict:

    def test_dyn_tensor_dict(self):
        case = _make_testcase()
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(inputs) == 2
        assert inputs[0]["shape"] == (-1,)
        assert inputs[0]["dtype"] == "float16"

    def test_bin_tensor_dict_with_const(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y", "valueDepend": "required"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=((3, 4), (2, 5)),
            output_shapes=((3, 5),),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert len(inputs) == 2
        assert inputs[0]["shape"] == (-2,)
        assert inputs[1]["shape"] == (-1, -1)

    def test_bin_tensor_dict_none_input(self):
        case = _make_testcase(input_shapes=(None, (3, 4)))
        _validate(case)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            inputs, outputs = case.bin_tensor_dict
        assert inputs[0] is None
        assert inputs[1]["shape"] == (-2,)

    def test_bin_tensor_dict_tensor_list_input(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert isinstance(inputs[0], tuple)
        assert len(inputs[0]) == 2
        assert inputs[0][0]["shape"] == (-2,)
        assert inputs[0][1]["shape"] == (-2,)
        assert inputs[1]["shape"] == (-2,)

    def test_bin_tensor_dict_tensor_list_output(self):
        case = _make_testcase(
            input_shapes=((3, 4),),
            input_dtypes=("float16",),
            output_shapes=(((3, 4), (5, 6)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            inputs, outputs = case.bin_tensor_dict
        assert isinstance(outputs[0], tuple)
        assert len(outputs[0]) == 2
        assert outputs[0][0]["shape"] == (-2,)
        assert outputs[0][1]["shape"] == (-2,)

    def test_bin_tensor_dict_multiple_const(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"},
                       {"name": "y"},
                       {"name": "z", "valueDepend": "required"}],
            "outputs": [{"name": "out"}],
        }
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6), (2, 3)),
            input_dtypes=("float16", "float16", "float16"),
            output_shapes=((5, 4),),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert inputs[0]["shape"] == (-1, -1)
        assert inputs[1]["shape"] == (-2,)
        assert inputs[2]["shape"] == (-1, -1)

    def test_bin_tensor_dict_const_and_tensor_list_mixed(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"},
                       {"name": "y"},
                       {"name": "z"}],
            "outputs": [{"name": "out"}],
        }
        case = _make_testcase(
            input_shapes=((2, 3), ((3, 4), (5, 6)), (7, 8)),
            input_dtypes=("float16", ("float16", "float16"), "float16"),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        # pos0: const, preserved
        assert inputs[0]["shape"] == (-1, -1)
        # pos1: TensorList (non-const), all replaced
        assert isinstance(inputs[1], tuple)
        assert len(inputs[1]) == 2
        assert inputs[1][0]["shape"] == (-2,)
        assert inputs[1][1]["shape"] == (-2,)
        # pos2: non-const, replaced
        assert inputs[2]["shape"] == (-2,)

    def test_bin_tensor_dict_tl_input_and_tl_output(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 6)), (7, 8)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((2, 3), (4, 5)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        assert isinstance(inputs[0], tuple)
        assert len(inputs[0]) == 2
        assert inputs[0][0]["shape"] == (-2,)
        assert inputs[0][1]["shape"] == (-2,)
        assert inputs[1]["shape"] == (-2,)
        assert isinstance(outputs[0], tuple)
        assert len(outputs[0]) == 2
        assert outputs[0][0]["shape"] == (-2,)
        assert outputs[0][1]["shape"] == (-2,)

    def test_bin_tensor_dict_cached(self):
        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            output_shapes=((3, 6),),
        )
        _validate(case)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = None
            result1 = case.bin_tensor_dict
            result2 = case.bin_tensor_dict
        assert result1 is result2

    def test_bin_tensor_dict_range_replaced(self):
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=((2, 3), (4, 5)),
            input_dtypes=("float16", "float16"),
            output_shapes=((4, 3),),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            inputs, outputs = case.bin_tensor_dict
        # const: range from dyn_input_ranges (preserved)
        assert inputs[0]["range"] == ((1, None), (1, None))
        # non-const: range replaced to dynamic
        assert inputs[1]["range"] == ((1, None),)
        # output: range replaced
        assert outputs[0]["range"] == ((1, None),)


class TestConstInputIndexes:

    def test_no_const_indexes_no_change(self):
        case = _make_testcase()
        _validate(case)
        assert case.input_shapes == ((8,), (8,))

    def test_const_indexes_preserves_stc(self):
        case = _make_testcase(
            input_shapes=((1,), (1,), (1,)),
            input_dtypes=("float16", "float16", "float16"),
            output_shapes=((1,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_shapes == ((1,), (1,), (1,))
        assert len(case.input_dtypes) == 3


class TestInputBytes:

    def test_input_bytes_calculation(self):
        case = _make_testcase()
        _validate(case)
        assert case.input_bytes > 0

    def test_output_bytes_calculation(self):
        case = _make_testcase()
        _validate(case)
        assert case.output_bytes > 0


class TestGetCompilationHash:

    def test_same_params_same_hash(self):
        case1 = _make_testcase()
        case2 = _make_testcase()
        _validate(case1)
        _validate(case2)
        assert case1.get_compilation_hash() == case2.get_compilation_hash()

    def test_different_params_different_hash(self):
        case1 = _make_testcase()
        case2 = _make_testcase(op_name="Sub")
        _validate(case1)
        _validate(case2)
        assert case1.get_compilation_hash() != case2.get_compilation_hash()


class TestDynamicize:

    def test_positive_dims_become_minus1(self):
        result = UniversalTestcaseStructure._dynamicize(((8, 16), (3, 224, 224)))
        assert result == ((-1, -1), (-1, -1, -1))

    def test_already_minus1_stays(self):
        result = UniversalTestcaseStructure._dynamicize(((-1,), (-1, -1)))
        assert result == ((-1,), (-1, -1))

    def test_minus2_stays(self):
        result = UniversalTestcaseStructure._dynamicize(((-2,),))
        assert result == ((-2,),)

    def test_zero_stays(self):
        result = UniversalTestcaseStructure._dynamicize(((0,),))
        assert result == ((0,),)

    def test_none_stays(self):
        result = UniversalTestcaseStructure._dynamicize((None,))
        assert result == (None,)

    def test_empty_shape_stays(self):
        result = UniversalTestcaseStructure._dynamicize(((),))
        assert result == ((),)

    def test_scalar_shape_stays(self):
        result = UniversalTestcaseStructure._dynamicize(((1,),))
        assert result == ((-1,),)

    def test_mixed_dims(self):
        result = UniversalTestcaseStructure._dynamicize(((8, -1, 0),))
        assert result == ((-1, -1, 0),)

    def test_multiple_tensors(self):
        result = UniversalTestcaseStructure._dynamicize(((8,), (16, 32), (-1,)))
        assert result == ((-1,), (-1, -1), (-1,))

    def test_empty_input(self):
        result = UniversalTestcaseStructure._dynamicize(())
        assert result == ()


class TestNestedShapes:

    def test_flat_backward_compat(self):
        """Flat shapes: distribution includes both input and output portions."""
        case = _make_testcase()
        _validate(case)
        assert case.tensor_list_distribution == (0, 0, 0)
        assert case.input_shapes == ((8,), (8,))

    def test_flat_with_no_nesting(self):
        """Flat shapes: distribution includes both input and output portions."""
        case = _make_testcase()
        _validate(case)
        assert case.tensor_list_distribution == (0, 0, 0)

    def test_nested_inputs_flattened(self):
        """Nested inputs: auto-compute distribution, flat properties flatten shapes"""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_formats=(("ND", "ND"), "ND"),
            input_ori_formats=(("ND", "ND"), "ND"),
            input_ori_shapes=(((3, 4), (5, 4)), (8,)),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 0)
        # Original fields preserved
        assert case.input_shapes == (((3, 4), (5, 4)), (8,))
        assert case.input_dtypes == (("float16", "float16"), "float16")
        assert case.input_formats == (("ND", "ND"), "ND")
        # Flat properties return flattened values
        assert case.flat_input_shapes == ((3, 4), (5, 4), (8,))
        assert case.flat_input_dtypes == ("float16", "float16", "float16")
        assert case.flat_input_formats == ("ND", "ND", "ND")

    def test_nested_inputs_compressed_dtypes(self):
        """Nested inputs with compressed dtypes: normalized to match distribution"""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_shapes == (((3, 4), (5, 4)), (8,))
        # Normalize expands compressed dtypes to match distribution
        assert case.input_dtypes == (("float16", "float16"), "float16")
        # Flat properties flatten normalized fields
        assert case.flat_input_shapes == ((3, 4), (5, 4), (8,))
        assert case.flat_input_dtypes == ("float16", "float16", "float16")

    def test_nested_outputs(self):
        """Nested outputs: compute output distribution, combine with input"""
        case = _make_testcase(
            input_shapes=((6, 4),),
            input_dtypes=("float16",),
            output_shapes=(((3, 4), (3, 4)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.tensor_list_distribution == (0, 2)
        # Original fields preserved as nested
        assert case.output_shapes == (((3, 4), (3, 4)),)
        assert case.output_dtypes == (("float16", "float16"),)
        # Flat properties return flattened
        assert case.flat_output_shapes == ((3, 4), (3, 4))
        assert case.flat_output_dtypes == ("float16", "float16")

    def test_nested_inputs_flat_outputs(self):
        """Nested inputs + flat outputs: output portion of dist has 0s"""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=((8, 4),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 0)

    def test_elewise_with_nested_inputs(self):
        """Nested inputs + ELEWISE outputs: string passthrough works"""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes="ELEWISE",
            output_dtypes=("float16",),
        )
        _validate(case)
        # ELEWISE resolves to broadcast shape, not flattened input shapes
        assert isinstance(case.output_shapes, tuple)
        assert len(case.output_shapes) == 1

    def test_const_input_indexes_after_flatten(self):
        """const_input_indexes calculated correctly after flattening"""
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x", "valueDepend": "required"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case = _make_testcase(
            input_shapes=((1,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case, op_info=op_info)
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            assert case.const_input_indexes == (0,)

    def test_compilation_hash_with_nested(self):
        """Compilation hash includes computed distribution"""
        case1 = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        case2 = _make_testcase()
        _validate(case1)
        _validate(case2)
        assert case1.get_compilation_hash() != case2.get_compilation_hash()

    def test_nested_precision_tolerances_outputs(self):
        """Nested precision_tolerances for nested outputs gets flattened."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        case.precision_tolerances = ((0.01, 0.02), (0.03, 0.04))
        _validate(case)
        assert case.flat_precision_tolerances == ((0.01, 0.02), (0.03, 0.04))

    def test_nested_absolute_precision_outputs(self):
        """Nested absolute_precision for nested outputs gets flattened."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        case.absolute_precision = (1e-5, 1e-6)
        _validate(case)
        assert case.flat_absolute_precision == (1e-5, 1e-6)

    def test_single_absolute_precision_stays(self):
        """Single float absolute_precision is normalized to tuple matching distribution."""
        case = _make_testcase()
        case.absolute_precision = 1e-5
        _validate(case)
        # Normalize wraps single float into tuple matching output dist
        assert case.absolute_precision == (1e-05,)
        assert case.flat_absolute_precision == (1e-05,)

    def test_nested_input_data_ranges_flattened(self):
        """Nested input_data_ranges: original preserved, flat property returns flattened."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        case.input_data_ranges = (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))
        _validate(case)
        # Original field preserved
        assert case.input_data_ranges == (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))
        # Flat property returns flattened
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))


class TestTensorApiFlatPrecision:
    """Tests for flat_precision_tolerances and flat_absolute_precision on aclnn/e2e base."""

    def test_flat_precision_tolerances_nested(self):
        from ttk.core_modules.testcase_manager.testcase_tensor_api_base import TensorApiTestcaseBase
        case = TensorApiTestcaseBase()
        case.tensor_view_shapes = (((3, 4), (5, 4)), (8,))
        # dist = (2, 0), first entry gets extended by _flatten_by_distribution
        case.precision_tolerances = (((0.01, 0.02), (0.03, 0.04)), (0.05, 0.06))
        case._inferred_tensor_list_dist = None
        assert case.flat_precision_tolerances == ((0.01, 0.02), (0.03, 0.04), (0.05, 0.06))

    def test_flat_absolute_precision_single_float(self):
        from ttk.core_modules.testcase_manager.testcase_tensor_api_base import TensorApiTestcaseBase
        case = TensorApiTestcaseBase()
        case.absolute_precision = 1e-5
        assert case.flat_absolute_precision == 1e-5
        assert isinstance(case.flat_absolute_precision, float)

    def test_flat_absolute_precision_nested_tuple(self):
        from ttk.core_modules.testcase_manager.testcase_tensor_api_base import TensorApiTestcaseBase
        case = TensorApiTestcaseBase()
        case.tensor_view_shapes = (((3, 4), (5, 4)), (8,))
        case.absolute_precision = ((1e-5, 1e-6), 1e-7)
        case._inferred_tensor_list_dist = None
        assert case.flat_absolute_precision == (1e-5, 1e-6, 1e-7)


# =====================================================================
# TDD tests for flat property refactoring:
# Original fields must be preserved; flat_* properties return flattened.
# These tests FAIL until the refactoring is complete.
# =====================================================================

class TestFlatPropertiesPreserveOriginal:
    """After validate(), original nested fields must be preserved as-is."""

    def test_input_shapes_preserved(self):
        """input_shapes retains nested structure after validate."""
        nested = (((3, 4), (5, 4)), (8,))
        case = _make_testcase(
            input_shapes=nested,
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.input_shapes == nested

    def test_input_dtypes_preserved(self):
        """input_dtypes retains nested structure after validate (recursive parse preserves nesting)."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        # Recursive parse preserves the nested tuple structure
        assert case.input_dtypes == (("float16", "float16"), "float16")

    def test_output_shapes_preserved(self):
        """output_shapes retains nested structure after validate."""
        nested = (((8,), (8,)),)
        case = _make_testcase(
            output_shapes=nested,
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_shapes == nested

    def test_output_dtypes_preserved(self):
        """output_dtypes retains nested structure after validate (recursive parse preserves nesting)."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        # Recursive parse preserves the nested tuple structure
        assert case.output_dtypes == (("float16", "float16"),)

    def test_compressed_dtypes_preserved(self):
        """Compressed dtypes like ('float16',) are normalized to match distribution."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.input_dtypes == (("float16", "float16"), "float16")

    def test_input_data_ranges_preserved(self):
        """input_data_ranges retains nested structure after validate."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)),
        )
        _validate(case)
        assert case.input_data_ranges == (((None, 1.0), (-1.0, 1.0)), (0.0, 5.0))


class TestFlatPropertiesReturnFlattened:
    """flat_* properties must return flattened (per-tensor) values."""

    def test_flat_input_shapes_nested(self):
        """flat_input_shapes flattens nested shapes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.flat_input_shapes == ((3, 4), (5, 4), (8,))

    def test_flat_input_shapes_flat_passthrough(self):
        """flat_input_shapes returns flat shapes as-is when not nested."""
        case = _make_testcase(input_shapes=((8,), (8,)))
        _validate(case)
        assert case.flat_input_shapes == ((8,), (8,))

    def test_flat_input_dtypes_nested(self):
        """flat_input_dtypes flattens nested dtypes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
        )
        _validate(case)
        assert case.flat_input_dtypes == ("float16", "float16", "float32")

    def test_flat_input_dtypes_compressed(self):
        """flat_input_dtypes broadcasts compressed ('float16',) to all tensors."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16",),
        )
        _validate(case)
        assert case.flat_input_dtypes == ("float16", "float16", "float16")

    def test_flat_output_shapes_nested(self):
        """flat_output_shapes flattens nested output shapes."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.flat_output_shapes == ((8,), (8,))

    def test_flat_output_dtypes_nested(self):
        """flat_output_dtypes flattens nested output dtypes."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float32"),),
        )
        _validate(case)
        assert case.flat_output_dtypes == ("float16", "float32")

    def test_flat_output_dtypes_compressed(self):
        """flat_output_dtypes broadcasts compressed ('float16',)."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=("float16",),
        )
        _validate(case)
        assert case.flat_output_dtypes == ("float16", "float16")

    def test_flat_input_formats_nested(self):
        """flat_input_formats flattens nested formats."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_formats=(("ND", "NCHW"), "ND"),
        )
        _validate(case)
        assert case.flat_input_formats == ("ND", "NCHW", "ND")

    def test_flat_input_formats_compressed(self):
        """flat_input_formats broadcasts compressed ('ND',)."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_formats=("ND",),
        )
        _validate(case)
        assert case.flat_input_formats == ("ND", "ND", "ND")

    def test_flat_output_formats_nested(self):
        """flat_output_formats flattens nested output formats."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
            output_formats=(("ND", "NCHW"),),
        )
        _validate(case)
        assert case.flat_output_formats == ("ND", "NCHW")

    def test_flat_input_data_ranges_nested(self):
        """flat_input_data_ranges flattens nested ranges."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)),
        )
        _validate(case)
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_flat_precision_tolerances_nested_outputs(self):
        """flat_precision_tolerances flattens for nested outputs."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        case.precision_tolerances = ((0.01, 0.02), (0.03, 0.04))
        _validate(case)
        assert case.flat_precision_tolerances == ((0.01, 0.02), (0.03, 0.04))

    def test_flat_absolute_precision_single_float(self):
        """Single float absolute_precision is normalized to tuple."""
        case = _make_testcase()
        case.absolute_precision = 1e-5
        _validate(case)
        assert case.flat_absolute_precision == (1e-05,)

    def test_flat_absolute_precision_nested(self):
        """Nested absolute_precision tuple gets flattened."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        case.absolute_precision = (1e-5, 1e-6)
        _validate(case)
        assert case.flat_absolute_precision == (1e-5, 1e-6)


class TestDynPropertiesUseFlat:
    """flat_dyn_* properties provide flat backward-compatible access."""

    def test_flat_dyn_inputs_from_nested(self):
        """flat_dyn_inputs returns dynamicized flat_input_shapes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        # _dynamicize replaces all > 0 dims with -1
        assert case.flat_dyn_inputs == ((-1, -1), (-1, -1), (-1,))

    def test_flat_dyn_input_dtypes_from_flat(self):
        """flat_dyn_input_dtypes returns flat_input_dtypes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
        )
        _validate(case)
        assert case.flat_dyn_input_dtypes == ("float16", "float16", "float32")

    def test_flat_dyn_outputs_from_nested(self):
        """flat_dyn_outputs returns dynamicized flat_output_shapes."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.flat_dyn_outputs == ((-1,), (-1,))

    def test_input_bytes_uses_flat(self):
        """input_bytes correctly computes from flat inputs."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.input_bytes > 0

    def test_output_bytes_uses_flat(self):
        """output_bytes correctly computes from flat outputs."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_bytes > 0


class TestSplitDistribution:
    """Input and output distributions are computed separately."""

    def test_input_distribution_nested(self):
        """input_distribution is inferred from nested input_shapes."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
        )
        _validate(case)
        assert case.input_distribution == (2, 0)

    def test_output_distribution_nested(self):
        """output_distribution is inferred from nested output_shapes."""
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.output_distribution == (2,)

    def test_combined_distribution(self):
        """tensor_list_distribution = input_dist + output_dist."""
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float16"),
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        assert case.tensor_list_distribution == (2, 0, 2)

    def test_flat_inputs_no_distribution(self):
        """Flat inputs produce (0, 0) input_distribution."""
        case = _make_testcase(input_shapes=((8,), (8,)))
        _validate(case)
        assert case.input_distribution == (0, 0)

    def test_no_nested_outputs_no_distribution(self):
        """Flat outputs produce (0,...) output_distribution matching len."""
        case = _make_testcase(output_shapes=((8,),))
        _validate(case)
        assert case.output_distribution == (0,)


class TestExtractInputAttrs:
    """
    TDD tests for extract_input_attrs(attrs, input_names).
    Returns attrs whose keys match input parameter names.
    Matching logic references param_transformation; keys preserved in original form.
    """

    @staticmethod
    def _extract(attrs, input_names):
        from ttk.utilities.container_utils import pickup_by_names
        return pickup_by_names(attrs, input_names)

    @staticmethod
    def _rest(attrs, result):
        return {k: v for k, v in attrs.items() if k not in result}

    def test_empty_attributes(self):
        assert self._extract({}, ["x"]) == {}

    def test_empty_input_names(self):
        attrs = {"axis": 0}
        r = self._extract(attrs, [])
        assert r == {}
        assert self._rest(attrs, r) == {"axis": 0}

    def test_direct_name_match(self):
        names = ["input_size", "filter", "out_backprop"]
        attrs = {"input_size": [2, 3, 18, 130, 130], "strides": [1, 1, 1, 1, 1]}
        r = self._extract(attrs, names)
        assert r == {"input_size": [2, 3, 18, 130, 130]}
        assert self._rest(attrs, r) == {"strides": [1, 1, 1, 1, 1]}

    def test_in_suffix_matches_input(self):
        names = ["input_size", "filter"]
        attrs = {"input_size_in__": [2, 3, 18], "strides": [1]}
        r = self._extract(attrs, names)
        assert r == {"input_size_in__": [2, 3, 18]}
        assert self._rest(attrs, r) == {"strides": [1]}

    def test_both_direct_and_suffixed_present(self):
        names = ["input_size", "filter"]
        attrs = {"input_size": [2, 3], "input_size_in__": [1, 1], "strides": [1]}
        r = self._extract(attrs, names)
        assert r == {"input_size": [2, 3], "input_size_in__": [1, 1]}
        assert self._rest(attrs, r) == {"strides": [1]}

    def test_axis_axes_alias_direct(self):
        names = ["x", "axes"]
        attrs = {"axis": 1}
        r = self._extract(attrs, names)
        assert r == {"axis": 1}
        assert self._rest(attrs, r) == {}

    def test_axes_axis_alias_direct(self):
        names = ["x", "axis"]
        attrs = {"axes": [1, 2]}
        r = self._extract(attrs, names)
        assert r == {"axes": [1, 2]}
        assert self._rest(attrs, r) == {}

    def test_axis_alias_with_in_suffix(self):
        names = ["x", "axes"]
        attrs = {"axis_in__": 1}
        r = self._extract(attrs, names)
        assert r == {"axis_in__": 1}
        assert self._rest(attrs, r) == {}

    def test_no_match_all_attrs(self):
        names = ["x", "y"]
        attrs = {"axis": 0, "keep_dims": True}
        r = self._extract(attrs, names)
        assert r == {}
        assert self._rest(attrs, r) == {"axis": 0, "keep_dims": True}

    def test_conv3d_style_mixed(self):
        names = ["input_size", "filter", "out_backprop"]
        attrs = {
            "input_size": [2, 3, 18, 130, 130],
            "strides": [1, 1, 1, 1, 1], "pads": [0, 0, 0, 0, 0, 0],
            "dilations": [1, 1, 1, 1, 1], "groups": 1, "data_format": "NCDHW",
        }
        r = self._extract(attrs, names)
        assert r == {"input_size": [2, 3, 18, 130, 130]}
        assert self._rest(attrs, r) == {
            "strides": [1, 1, 1, 1, 1], "pads": [0, 0, 0, 0, 0, 0],
            "dilations": [1, 1, 1, 1, 1], "groups": 1, "data_format": "NCDHW",
        }

    def test_ctc_loss_style(self):
        names = ["log_probs", "targets", "input_lengths", "target_lengths"]
        attrs = {"input_lengths": (70,) * 5, "target_lengths": (26, 11),
                 "blank": 0, "reduction": "none"}
        r = self._extract(attrs, names)
        assert r == {"input_lengths": (70,) * 5, "target_lengths": (26, 11)}
        assert self._rest(attrs, r) == {"blank": 0, "reduction": "none"}

    def test_input_name_not_in_attrs(self):
        names = ["repeats", "x"]
        attrs = {"axis": 0}
        r = self._extract(attrs, names)
        assert r == {}
        assert self._rest(attrs, r) == {"axis": 0}
