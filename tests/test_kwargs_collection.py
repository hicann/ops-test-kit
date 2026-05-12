"""
Tests for kwargs collection in golden/input generation:
- output_generation.__collect_dynamic_golden_kwargs
- input_generation.__collect_dynamic_kwargs

Verifies that kwargs use nested fields directly after normalize.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from ttk.core_modules.testcase_manager.testcase_op import UniversalTestcaseStructure

# Access module-level private functions via getattr to avoid name mangling in classes
from ttk.core_modules.npu.op import output_generation as _out_gen_mod
from ttk.core_modules.npu.op import input_generation as _in_gen_mod

_collect_golden_kwargs = getattr(_out_gen_mod, '__collect_dynamic_golden_kwargs')
_collect_input_kwargs = getattr(_in_gen_mod, '__collect_dynamic_kwargs')


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


def _validate(case):
    with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock:
        mock.return_value.info_of.return_value = None
        case.validate()


def _make_arrays(shapes, dtypes):
    arrays = []
    for shape, dtype in zip(shapes, dtypes):
        if shape is None:
            arrays.append(None)
        else:
            arrays.append(np.ones(shape, dtype=dtype))
    return arrays


def _mock_switches():
    sw = MagicMock()
    sw.dev_plat = "Ascend910B2"
    sw.short_soc_version = "Ascend910B"
    sw.golden_mode = "Enable"
    sw.plugin_path = None
    return sw


def _mock_op_info():
    mock = MagicMock()
    mock.return_value.info_of.return_value = {"inputs": []}
    return mock


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


# =====================================================================
# Tests for output_generation.__collect_dynamic_golden_kwargs
# =====================================================================

class TestGoldenKwargsNonTensorList:

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_input_dtypes_matches_stc(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,), (8,)),
            input_dtypes=("float16", "float32", "int32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes
        assert kwargs["input_dtypes"] == ("float16", "float32", "int32")

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_output_dtypes_matches_nested(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,),),
            input_dtypes=("float16",),
            output_shapes=((8,), (8,)),
            output_dtypes=("float16", "float32"),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["output_dtypes"] == case.output_dtypes

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_all_format_fields_match(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_formats=("ND", "NCHW"),
            input_ori_formats=("ND", "ND"),
            output_formats=("ND",),
            output_ori_formats=("ND",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_formats"] == case.input_formats
        assert kwargs["input_ori_formats"] == case.input_ori_formats
        assert kwargs["output_formats"] == case.output_formats
        assert kwargs["output_ori_formats"] == case.output_ori_formats

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_input_ori_shapes_matches_input_ori_shapes(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((3, 4), (5, 6)),
            input_dtypes=("float16", "float32"),
            output_shapes=((3, 4),),
            output_dtypes=("float16",),
            input_ori_shapes=((3, 4), (5, 6)),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_ori_shapes"] == case.input_ori_shapes


class TestGoldenKwargsTensorList:

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_input_dtypes_nested(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float16", "float32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_output_dtypes_with_tensor_list_output(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,),),
            input_dtypes=("float16",),
            output_shapes=(((3, 4), (5, 4)),),
            output_dtypes=("float16", "float32"),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["output_dtypes"] == case.output_dtypes

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_input_ori_shapes_nested(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float16", "float32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_ori_shapes=(((3, 4), (5, 4)), (8,)),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_ori_shapes"] == case.input_ori_shapes


class TestGoldenKwargsCompressedNormalized:

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_compressed_dtype_expanded(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,), (8,)),
            input_dtypes=("float16",),  # compressed → expand to 3
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        case.input_arrays = tuple(_make_arrays(
            case.flat_input_shapes, case.flat_input_dtypes))

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes
        assert kwargs["input_dtypes"] == ("float16", "float16", "float16")


class TestGoldenKwargsWithConstInput:

    @patch('ttk.core_modules.npu.op.output_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.output_generation.get_global_storage')
    def test_const_input_ori_shapes(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}
        op_info = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y", "valueDepend": "required"}, {"name": "z"}],
            "outputs": [{"name": "out"}],
        }

        case = _make_testcase(
            input_shapes=((8,), (3,), (8,)),
            input_dtypes=("float16", "int32", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_ori_shapes=((8,), (3,), (8,)),
        )
        with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as m:
            m.return_value.info_of.return_value = op_info
            case.validate()
        input_arrs = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)
        input_arrs[1] = np.array([1, 2, 3], dtype="int32")
        case.input_arrays = tuple(input_arrs)

        kwargs = _collect_golden_kwargs(case)
        assert kwargs["input_ori_shapes"] == case.input_ori_shapes
        assert kwargs["input_dtypes"] == case.input_dtypes


# =====================================================================
# Tests for input_generation.__collect_dynamic_kwargs
# =====================================================================

class TestInputKwargsNonTensorList:

    @patch('ttk.core_modules.npu.op.input_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.input_generation.get_global_storage')
    def test_input_dtypes_matches_stc(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,), (8,)),
            input_dtypes=("float16", "float32", "int32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        input_arrays = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes

    @patch('ttk.core_modules.npu.op.input_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.input_generation.get_global_storage')
    def test_output_fields_match(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,),),
            input_dtypes=("float16",),
            output_shapes=((8,), (8,)),
            output_dtypes=("float16", "float32"),
            output_formats=("ND", "NCHW"),
            output_ori_formats=("ND", "ND"),
        )
        _validate(case)
        input_arrays = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["output_dtypes"] == case.output_dtypes
        assert kwargs["output_formats"] == case.output_formats
        assert kwargs["output_ori_formats"] == case.output_ori_formats
        assert kwargs["output_ori_shapes"] == case.output_ori_shapes

    @patch('ttk.core_modules.npu.op.input_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.input_generation.get_global_storage')
    def test_input_ranges_matches_stc(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=((8,), (8,)),
            input_dtypes=("float16", "float16"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_data_ranges=((-1.0, 1.0), (-2.0, 2.0)),
        )
        _validate(case)
        input_arrays = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["input_ranges"] == case.input_data_ranges


class TestInputKwargsTensorList:

    @patch('ttk.core_modules.npu.op.input_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.input_generation.get_global_storage')
    def test_nested_input_dtypes(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float16", "float32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
        )
        _validate(case)
        input_arrays = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["input_dtypes"] == case.input_dtypes

    @patch('ttk.core_modules.npu.op.input_generation.OpInfoKeeper')
    @patch('ttk.core_modules.npu.op.input_generation.get_global_storage')
    def test_input_ori_shapes_nested(self, mock_sw, mock_op_info):
        mock_sw.return_value = _mock_switches()
        mock_op_info.return_value.info_of.return_value = {"inputs": []}

        case = _make_testcase(
            input_shapes=(((3, 4), (5, 4)), (8,)),
            input_dtypes=("float16", "float16", "float32"),
            output_shapes=((8,),),
            output_dtypes=("float16",),
            input_ori_shapes=(((3, 4), (5, 4)), (8,)),
        )
        _validate(case)
        input_arrays = _make_arrays(case.flat_input_shapes, case.flat_input_dtypes)

        kwargs = _collect_input_kwargs(case)
        assert kwargs["input_ori_shapes"] == case.input_ori_shapes
