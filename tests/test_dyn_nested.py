"""
Tests for dyn_* nested refactor:
- dyn_inputs/dyn_outputs return nested format matching stc_* structure
- dyn_input_dtypes etc. return nested format matching stc_* structure
"""

import pytest
from unittest.mock import patch

from ttk.core_modules.testcase_manager.testcase_op import UniversalTestcaseStructure


def _make_testcase(op_name="Add", input_shapes=((8,), (8,)),
                   input_dtypes=("float16", "float16"),
                   output_shapes=((8,),),
                   output_dtypes=("float16",),
                   const_input_indexes=None,
                   **kwargs):
    case = UniversalTestcaseStructure()
    case.testcase_name = f"test_{op_name or 'None'}"
    case.op_name = op_name
    case.input_shapes = input_shapes
    case.input_dtypes = input_dtypes
    case.output_shapes = output_shapes
    case.output_dtypes = output_dtypes
    if const_input_indexes is not None:
        case.const_input_indexes = const_input_indexes
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


def _validate(case):
    with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock:
        mock.return_value.info_of.return_value = None
        case.validate()


class TestDynInputsNested:
    def test_dyn_inputs_flat_no_tensorlist(self):
        case = _make_testcase(input_shapes=((8,), (8,)))
        _validate(case)
        assert case.dyn_inputs == ((-1,), (-1,))

    def test_dyn_inputs_nested_with_tensorlist(self):
        case = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)))
        _validate(case)
        assert case.dyn_inputs == (((-1, -1), (-1, -1)), (-1,))

    def test_dyn_inputs_empty(self):
        case = _make_testcase(input_shapes=(), input_dtypes=())
        _validate(case)
        assert case.dyn_inputs == ()

    def test_dyn_inputs_none_element(self):
        case = _make_testcase(input_shapes=((8,), None))
        _validate(case)
        assert case.dyn_inputs[0] == (-1,)
        assert case.dyn_inputs[1] is None

    def test_dynamicize_nested_negative_dims(self):
        result = UniversalTestcaseStructure._dynamicize_nested(((-2,), (8,)), (0, 0))
        assert result == ((-2,), (-1,))


class TestDynInputDtypesNested:
    def test_dyn_input_dtypes_flat(self):
        case = _make_testcase(input_dtypes=("float16", "float32"))
        _validate(case)
        assert case.dyn_input_dtypes == ("float16", "float32")

    def test_dyn_input_dtypes_nested(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
        )
        _validate(case)
        assert case.dyn_input_dtypes == (("float16", "float16"), "float32")

    def test_dyn_input_formats_nested(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_formats=(("ND", "ND"), "NCHW"),
        )
        _validate(case)
        assert case.dyn_input_formats == (("ND", "ND"), "NCHW")

    def test_dyn_outputs_nested(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),))
        _validate(case)
        assert case.dyn_outputs == (((-1,), (-1,)),)

    def test_dyn_ori_inputs_nested(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_ori_shapes=(((3, 4), (5, 4)), (8,)),
        )
        _validate(case)
        assert case.dyn_ori_inputs == (((-1, -1), (-1, -1)), (-1,))


class TestFlatDynProperties:
    def test_flat_dyn_inputs_no_tensorlist(self):
        case = _make_testcase(input_shapes=((8,), (8,)))
        _validate(case)
        assert case.flat_dyn_inputs == ((-1,), (-1,))

    def test_flat_dyn_inputs_with_tensorlist(self):
        case = _make_testcase(input_shapes=(((3, 4), (5, 4)), (8,)))
        _validate(case)
        assert case.flat_dyn_inputs == ((-1, -1), (-1, -1), (-1,))

    def test_flat_dyn_input_dtypes_with_tensorlist(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
        )
        _validate(case)
        assert case.flat_dyn_input_dtypes == ("float16", "float16", "float32")

    def test_flat_dyn_outputs_with_tensorlist(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),))
        _validate(case)
        assert case.flat_dyn_outputs == ((-1,), (-1,))

    def test_flat_dyn_inputs_empty(self):
        case = _make_testcase(input_shapes=(), input_dtypes=())
        _validate(case)
        assert case.flat_dyn_inputs == ()


class TestDynTensorDictNested:
    def test_dyn_tensor_dict_no_tensorlist(self):
        case = _make_testcase(input_shapes=((8,), (8,)), output_shapes=((8,),))
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(inputs) == 2
        assert len(outputs) == 1
        assert inputs[0]["shape"] == (-1,)
        assert inputs[0]["dtype"] == "float16"
        assert inputs[0]["format"] == "ND"
        assert inputs[0]["ori_shape"] == (-1,)
        assert outputs[0]["shape"] == (-1,)

    def test_dyn_tensor_dict_with_tensorlist(self):
        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=(("float16", "float16"), "float32"),
            output_shapes=((8,),),
        )
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(inputs) == 2
        # Position 0 is TensorList
        assert isinstance(inputs[0], tuple)
        assert len(inputs[0]) == 2
        assert inputs[0][0]["shape"] == (-1, -1)
        assert inputs[0][0]["dtype"] == "float16"
        assert inputs[0][1]["shape"] == (-1, -1)
        assert inputs[0][1]["dtype"] == "float16"
        # Position 1 is normal
        assert not isinstance(inputs[1], tuple)
        assert inputs[1]["dtype"] == "float32"

    def test_dyn_tensor_dict_output_tensorlist(self):
        case = _make_testcase(
            output_shapes=(((8,), (8,)),),
            output_dtypes=(("float16", "float16"),),
        )
        _validate(case)
        inputs, outputs = case.dyn_tensor_dict
        assert len(outputs) == 1
        assert isinstance(outputs[0], tuple)
        assert len(outputs[0]) == 2
        assert outputs[0][0]["shape"] == (-1,)
        assert outputs[0][0]["dtype"] == "float16"
        assert outputs[0][1]["shape"] == (-1,)

    def test_dyn_tensor_dict_none_element(self):
        case = _make_testcase(input_shapes=(None, (8,)), input_dtypes=("float16", "float16"))
        _validate(case)
        inputs, _ = case.dyn_tensor_dict
        assert inputs[0] is None
        assert inputs[1]["shape"] == (-1,)

    def test_dyn_tensor_dict_cached(self):
        case = _make_testcase()
        _validate(case)
        td1 = case.dyn_tensor_dict
        td2 = case.dyn_tensor_dict
        assert td1 is td2

    def test_dyn_tensor_dict_scalar_shape_eliminated(self):
        """Scalar shapes () should be converted to (1,) via eliminate_scalar_shapes."""
        case = _make_testcase(input_shapes=((), (8,)), input_dtypes=("float16", "float16"))
        _validate(case)
        inputs, _ = case.dyn_tensor_dict
        # () is dynamicized to () then eliminate_scalar_shapes → (1,)
        assert inputs[0]["shape"] == (1,)

    def test_eliminate_scalar_shapes_nested_static(self):
        """Test _eliminate_scalar_shapes_nested handles TensorList nesting."""
        shapes = (((), (8,)), ((),))
        result = UniversalTestcaseStructure._eliminate_scalar_shapes_nested(shapes)
        assert result == (((1,), (8,)), ((1,),))

    def test_eliminate_scalar_shapes_nested_none(self):
        shapes = (None, (8,))
        result = UniversalTestcaseStructure._eliminate_scalar_shapes_nested(shapes)
        assert result[0] is None
        assert result[1] == (8,)

    def test_eliminate_scalar_shapes_nested_empty(self):
        result = UniversalTestcaseStructure._eliminate_scalar_shapes_nested(())
        assert result == ()
