"""
Tests for manual_golden_binaries / manual_output_data_dtypes
normalize + validate + reshape in testcase_op.py.

Mirrors TestNormalizeManualBinaries but for output side.
"""

import pytest
from unittest.mock import patch

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
def _mock_env(monkeypatch):
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)


def _validate(case):
    with patch('ttk.core_modules.operator.op_info_keeper.OpInfoKeeper') as mock:
        mock.return_value.info_of.return_value = {
            "coreType.value": "AiCore",
            "inputs": [{"name": "x"}, {"name": "y"}],
            "outputs": [{"name": "z"}],
        }
        case.validate()


# =====================================================================
# Normalize: basic type conversions
# =====================================================================

class TestNormalizeManualOutputBinaries:

    def test_empty_not_modified(self):
        case = _make_testcase()
        case.manual_golden_binaries = ()
        _validate(case)
        assert case.manual_golden_binaries == ()

    def test_none_not_modified(self):
        case = _make_testcase()
        case.manual_golden_binaries = None
        _validate(case)
        assert case.manual_golden_binaries is None

    def test_single_string_wrapped(self):
        case = _make_testcase(output_shapes=((8,),), output_dtypes=("float16",))
        case.manual_golden_binaries = 'out.bin'
        _validate(case)
        assert case.manual_golden_binaries == ('out.bin',)

    def test_flat_tuple_preserved(self):
        case = _make_testcase(output_shapes=((8,), (8,), (8,)),
                              output_dtypes=("float16", "float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin', 'o3.bin')
        _validate(case)
        assert case.manual_golden_binaries == ('o1.bin', 'o2.bin', 'o3.bin')

    def test_none_quoted_converted(self):
        case = _make_testcase(output_shapes=((8,), None, (8,)),
                              output_dtypes=("float16", "float16", "float32"))
        case.manual_golden_binaries = ('o1.bin', 'None', 'o3.bin')
        _validate(case)
        assert case.manual_golden_binaries == ('o1.bin', None, 'o3.bin')

    def test_list_converted_to_tuple(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ['o1.bin', 'o2.bin']
        _validate(case)
        assert isinstance(case.manual_golden_binaries, tuple)

    def test_invalid_type_rejected(self):
        case = _make_testcase(output_shapes=((8,),), output_dtypes=("float16",))
        case.manual_golden_binaries = 123
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "MANUAL_OUTPUT_BINARIES_INVALID"



# =====================================================================
# Validation: flat outputs
# =====================================================================

class TestValidateOutputBinariesFlat:

    def test_flat_count_matches(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin')
        _validate(case)
        assert case.is_valid

    def test_flat_with_none_output(self):
        case = _make_testcase(output_shapes=((8,), None, (8,)),
                              output_dtypes=("float16", "float16", "float32"))
        case.manual_golden_binaries = ('o1.bin', None, 'o3.bin')
        _validate(case)
        assert case.is_valid

    def test_flat_exceeds_outputs_rejected(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin', 'o3.bin')
        _validate(case)
        assert case.is_valid is False
        assert case.fail_reason == "MANUAL_OUTPUT_BINARIES_INVALID"

    def test_file_for_none_output_rejected(self):
        case = _make_testcase(output_shapes=((8,), None, (8,)),
                              output_dtypes=("float16", "float16", "float32"))
        case.manual_golden_binaries = ('o1.bin', 'unexpected.bin', 'o3.bin')
        _validate(case)
        assert case.is_valid is False

    def test_missing_file_for_non_none_rejected(self):
        case = _make_testcase(output_shapes=((8,), (8,), (8,)),
                              output_dtypes=("float16", "float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', None, 'o3.bin')
        _validate(case)
        assert case.is_valid is False

    def test_flat_trailing_none_padded(self):
        case = _make_testcase(output_shapes=((8,), None, None),
                              output_dtypes=("float16", "float16", "float16"))
        case.manual_golden_binaries = ('o1.bin',)
        _validate(case)
        assert case.manual_golden_binaries == ('o1.bin', None, None)


# =====================================================================
# Validation: nested (TensorList) outputs
# =====================================================================

class TestValidateOutputBinariesNested:

    def test_nested_preserved(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'),)
        _validate(case)
        assert case.manual_golden_binaries == (('o1.bin', 'o2.bin'),)

    def test_nested_with_none_in_tensorlist(self):
        case = _make_testcase(output_shapes=(((8,), None),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = (('o1.bin', None),)
        _validate(case)
        assert case.manual_golden_binaries == (('o1.bin', None),)

    def test_nested_rejected_without_tensorlist(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'),)
        _validate(case)
        assert case.is_valid is False

    def test_nested_top_level_count_mismatch(self):
        case = _make_testcase(output_shapes=(((8,), (8,)), (4,)),
                              output_dtypes=("float16", "float32"))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'),)
        _validate(case)
        assert case.is_valid is False

    def test_nested_tensorlist_position_is_str_rejected(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = ('o1.bin',)
        _validate(case)
        assert case.is_valid is False

    def test_nested_non_tensorlist_position_is_tuple_rejected(self):
        case = _make_testcase(output_shapes=(((8,), (8,)), (4,)),
                              output_dtypes=("float16", "float32"))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'), ('o3.bin',))
        _validate(case)
        assert case.is_valid is False

    def test_file_for_none_in_tensorlist_rejected(self):
        case = _make_testcase(output_shapes=(((8,), None),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = (('o1.bin', 'unexpected.bin'),)
        _validate(case)
        assert case.is_valid is False

    def test_missing_file_for_non_none_tensorlist_rejected(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = (('o1.bin', None),)
        _validate(case)
        assert case.is_valid is False


# =====================================================================
# Reshape: flat → nested
# =====================================================================

class TestReshapeOutputBinaries:

    def test_flat_to_nested(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin')
        _validate(case)
        assert case.manual_golden_binaries == (('o1.bin', 'o2.bin'),)

    def test_flat_with_none_to_nested(self):
        case = _make_testcase(output_shapes=(((8,), None),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = ('o1.bin', None)
        _validate(case)
        assert case.manual_golden_binaries == (('o1.bin', None),)

    def test_mixed_tensorlist_and_flat(self):
        case = _make_testcase(output_shapes=(((8,), (8,)), (4,)),
                              output_dtypes=("float16", "float32"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin', 'o3.bin')
        _validate(case)
        assert case.manual_golden_binaries == (('o1.bin', 'o2.bin'), 'o3.bin')

    def test_no_tensorlist_stays_flat(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin')
        _validate(case)
        assert case.manual_golden_binaries == ('o1.bin', 'o2.bin')


# =====================================================================
# Flat properties
# =====================================================================

class TestFlatOutputBinariesProperties:

    def test_flat_manual_golden_binaries_flat(self):
        case = _make_testcase(output_shapes=((8,), (8,)),
                              output_dtypes=("float16", "float16"))
        case.manual_golden_binaries = ('o1.bin', 'o2.bin')
        _validate(case)
        assert case.flat_manual_golden_binaries == ('o1.bin', 'o2.bin')

    def test_flat_manual_golden_binaries_nested(self):
        case = _make_testcase(output_shapes=(((8,), (8,)),),
                              output_dtypes=(("float16", "float16"),))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'),)
        _validate(case)
        assert case.flat_manual_golden_binaries == ('o1.bin', 'o2.bin')

    def test_flat_manual_golden_binaries_mixed(self):
        case = _make_testcase(output_shapes=(((8,), (8,)), (4,)),
                              output_dtypes=("float16", "float32"))
        case.manual_golden_binaries = (('o1.bin', 'o2.bin'), 'o3.bin')
        _validate(case)
        assert case.flat_manual_golden_binaries == ('o1.bin', 'o2.bin', 'o3.bin')

    def test_flat_manual_output_none_when_not_set(self):
        case = _make_testcase()
        _validate(case)
        assert case.flat_manual_golden_binaries is None
