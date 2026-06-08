#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Tests for ttk.core_modules.testcase_manager.testcase_e2e:
nested tensor structure, flat properties, compression, normalize, and validate.
"""

import pytest
from ttk.core_modules.testcase_manager.testcase_e2e import TestcaseE2e
from ttk.utilities.simple_param_extractor import APIParamInfo, ParamInfo, _MANUAL_OVERRIDES, OverloadInfo
from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper


NESTED_SHAPES = (((3, 3), (3, 2)), (3, 5))
FLAT_SHAPES = ((3, 3), (3, 5))
TENSORLIST_SHAPES = (((2, 3), (2, 5), (2, 1)),)


def _make(api_name="torch.dummy", **kwargs):
    case = TestcaseE2e()
    case.api_name = api_name
    case.is_valid = True
    case.fail_reason = None
    case.attributes = kwargs.pop("attributes", {})
    for k, v in kwargs.items():
        setattr(case, k, v)
    return case


@pytest.fixture
def make_testcase():
    return _make


class TestGetTensorListDistribution:

    def test_flat_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES)
        assert case.tensor_list_dist == (0, 0)

    def test_nested_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES)
        assert case.tensor_list_dist == (2, 0)

    def test_tensorlist_only(self, make_testcase):
        case = make_testcase(tensor_view_shapes=TENSORLIST_SHAPES)
        assert case.tensor_list_dist == (3,)

    def test_none(self, make_testcase):
        case = make_testcase(tensor_view_shapes=None)
        assert case.tensor_list_dist == ()


class TestFlatTensorViewShapes:

    def test_flat(self, make_testcase):
        case = make_testcase(tensor_view_shapes=FLAT_SHAPES)
        assert case.flat_tensor_view_shapes == FLAT_SHAPES

    def test_nested(self, make_testcase):
        case = make_testcase(tensor_view_shapes=NESTED_SHAPES)
        assert case.flat_tensor_view_shapes == ((3, 3), (3, 2), (3, 5))

    def test_tensorlist(self, make_testcase):
        case = make_testcase(tensor_view_shapes=TENSORLIST_SHAPES)
        assert case.flat_tensor_view_shapes == ((2, 3), (2, 5), (2, 1))


class TestFlatTensorDtypesCompression:

    def test_compressed_broadcast_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            tensor_dtypes=('float32',))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'float32')
        assert all(isinstance(x, str) for x in case.flat_tensor_dtypes)

    def test_per_param_after_normalize(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            tensor_dtypes=('float32', 'int8'))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'int8')

    def test_fully_nested(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            tensor_dtypes=(('float32', 'float16', 'int8'),))
        assert case.flat_tensor_dtypes == ('float32', 'float16', 'int8')

    def test_flat_no_nesting(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=FLAT_SHAPES,
            tensor_dtypes=('float32', 'float32'))
        assert case.flat_tensor_dtypes == ('float32', 'float32')


class TestNormalizeSkipsAlreadyNested:

    def test_fully_nested_dtypes_unchanged(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            tensor_dtypes=(('float32', 'float32', 'float32'),))
        case._normalize_compressed_fields()
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'float32')
        assert all(isinstance(x, str) for x in case.flat_tensor_dtypes)

    def test_fully_nested_ranges_unchanged(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            input_data_ranges=(((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),))
        case._normalize_compressed_fields()
        flat = case.flat_input_data_ranges
        assert flat == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))
        assert all(isinstance(x, tuple) and len(x) == 2 for x in flat)


class TestFlatInputDataRangesFramework:

    def test_compressed_broadcast(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            input_data_ranges=((-1.0, 1.0),))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    def test_per_param(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=((None, 1.0), (-1.0, 1.0)))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (None, 1.0), (-1.0, 1.0))

    def test_already_nested(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            input_data_ranges=(((None, 1.0), (-1.0, 1.0)), (0.0, 5.0)))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))

    def test_fully_nested_tensorlist(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            input_data_ranges=(((None, 1.0), (-1.0, 1.0), (0.0, 5.0)),))
        case._normalize_compressed_fields()
        assert case.flat_input_data_ranges == ((None, 1.0), (-1.0, 1.0), (0.0, 5.0))


class TestPureOutputIndexes:

    def test_flat_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            output_tensor_indexes=(2,))
        assert case.pure_output_indexes == [2]

    def test_tensorlist_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=NESTED_SHAPES,
            output_tensor_indexes=(0,))
        assert case.pure_output_indexes == [0, 1]

    def test_multi_output(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3), (2, 3), (2, 3)),
            output_tensor_indexes=(1, 3))
        assert case.pure_output_indexes == [1, 3]


class TestValidateFrameworkApi:

    @pytest.fixture(autouse=True)
    def mock_api_info(self):
        from unittest.mock import patch
        info = APIParamInfo(
            api_name="torch.dummy",
            overloads=[
                [ParamInfo(name="input", type="Tensor"),
                 ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="tensors", type="List[Tensor]")],
            ])
        with patch.object(TestcaseE2e, 'get_api_info', return_value=info):
            yield

    def test_valid_flat(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_valid_nested_compressed(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=TENSORLIST_SHAPES,
            tensor_dtypes=('float32',))
        case.validate()
        assert case.is_valid
        assert case.flat_tensor_dtypes == ('float32', 'float32', 'float32')

    def test_missing_api_name(self, make_testcase):
        case = make_testcase(api_name="", tensor_view_shapes=((2, 3),), tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid

    def test_missing_shapes(self, make_testcase):
        case = make_testcase(tensor_view_shapes=(), tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid

    def test_output_index_out_of_range(self, make_testcase):
        case = make_testcase(
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(5,))
        case.validate()
        assert not case.is_valid


class TestFlattenByDistributionFramework:

    def test_already_nested(self):
        result = TestcaseE2e._flatten_by_distribution(
            (((None, 1.0), (-1.0, 1.0)), 'x'), (2, 0))
        assert result == ((None, 1.0), (-1.0, 1.0), 'x')

    def test_scalar_broadcast(self):
        result = TestcaseE2e._flatten_by_distribution(
            ('x',), (3,))
        assert result == ('x', 'x', 'x')


class TestIsFieldAlreadyNestedFramework:
    """Tests for already-nested detection via _normalize_field_by_dist with _is_scalar_group."""

    def _assert_already_nested(self, field, dist):
        """Verify scalar field is detected as already-nested (no modification)."""
        case = TestcaseE2e()
        case.is_valid = True
        case.tensor_dtypes = field
        case._normalize_field_by_dist("tensor_dtypes", dist, TestcaseE2e._is_scalar_group)
        assert case.is_valid is True
        assert case.tensor_dtypes == field

    def _assert_not_already_nested(self, field, dist):
        """Verify scalar field is NOT already-nested (gets normalized or rejected)."""
        case = TestcaseE2e()
        case.is_valid = True
        case.tensor_dtypes = field
        case._normalize_field_by_dist("tensor_dtypes", dist, TestcaseE2e._is_scalar_group)
        assert case.tensor_dtypes != field or case.is_valid is False

    def test_matches(self):
        self._assert_already_nested((('float32', 'float32', 'float32'),), (3,))

    def test_not_matches(self):
        self._assert_not_already_nested(('float32',), (3,))


class TestCheckTensorConfiguration:
    """Tests for _check_tensor_configuration — hard validation against API signature."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def _register_api(self, api_name, params):
        FrameworkApiInfoKeeper().register(api_name, params, source="test")

    def test_tensor_count_mismatch_too_many_invalidates(self, make_testcase):
        self._register_api("torch.test_count", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = make_testcase(
            api_name="torch.test_count",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32'))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_tensor_count_too_few_required_invalidates(self, make_testcase):
        self._register_api("torch.test_req", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
        ])
        case = make_testcase(
            api_name="torch.test_req",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "TENSOR_COUNT_MISMATCH"

    def test_optional_out_omitted_valid(self, make_testcase):
        self._register_api("torch.test_opt", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
        ])
        case = make_testcase(
            api_name="torch.test_opt",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_optional_out_provided_valid(self, make_testcase):
        self._register_api("torch.test_opt2", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
        ])
        case = make_testcase(
            api_name="torch.test_opt2",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32'),
            output_tensor_indexes=(2,))
        case.validate()
        assert case.is_valid

    def test_tensor_count_match_valid(self, make_testcase):
        self._register_api("torch.test_ok", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = make_testcase(
            api_name="torch.test_ok",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_tensorlist_type_mismatch_flat_for_list_param(self, make_testcase):
        self._register_api("torch.test_tl_mismatch", [
            ParamInfo(name="tensors", type="tuple of Tensors"),
            ParamInfo(name="dim", type="int"),
        ])
        case = make_testcase(
            api_name="torch.test_tl_mismatch",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "PARAM_TYPE_MISMATCH"

    def test_tensorlist_type_match_valid(self, make_testcase):
        self._register_api("torch.test_tl_ok", [
            ParamInfo(name="tensors", type="tuple of Tensors"),
            ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
        ])
        case = make_testcase(
            api_name="torch.test_tl_ok",
            tensor_view_shapes=(((2, 3), (2, 5)), (3, 3)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert case.is_valid

    def test_single_tensor_nested_for_tensor_param_invalid(self, make_testcase):
        self._register_api("torch.test_nested_mismatch", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ])
        case = make_testcase(
            api_name="torch.test_nested_mismatch",
            tensor_view_shapes=(((2, 3), (2, 5)), (3, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "PARAM_TYPE_MISMATCH"

    def test_api_not_found_skips_validation(self, make_testcase):
        case = make_testcase(
            api_name="nonexistent.api",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "API_PARSE_FAIL"

    def test_none_element_for_tensorlist_skipped(self, make_testcase):
        self._register_api("torch.test_none_tl", [
            ParamInfo(name="tensors", type="tuple of Tensors"),
            ParamInfo(name="out", type="Tensor", default="None", is_optional=True),
        ])
        case = make_testcase(
            api_name="torch.test_none_tl",
            tensor_view_shapes=(None, (3, 3)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert case.is_valid

    def test_live_torch_add_count_valid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.add",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_live_torch_add_count_invalid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.add",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "TENSOR_COUNT_MISMATCH"

    def test_live_torch_cat_tensorlist_valid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.cat",
            tensor_view_shapes=(((2, 3), (2, 5)),),
            tensor_dtypes=('float32',))
        case.validate()
        assert case.is_valid

    def test_live_torch_cat_flat_tensor_invalid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.cat",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "PARAM_TYPE_MISMATCH"

    def test_live_torch_div_tensor_tensor_valid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.div",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_live_torch_div_tensor_scalar_valid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.div",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',),
            attributes={'other': 2.0})
        case.validate()
        assert case.is_valid


class TestCheckMultiOverload:
    """Tests for multi-overload matching in _check_tensor_configuration."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_multi_overload_matches_second(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.test_multi", [
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            [ParamInfo(name="input", type="Tensor")],
        ], source="test")
        case = make_testcase(
            api_name="torch.test_multi",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert case.is_valid

    def test_multi_overload_matches_first(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.test_multi2", [
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            [ParamInfo(name="input", type="Tensor")],
        ], source="test")
        case = make_testcase(
            api_name="torch.test_multi2",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_multi_overload_no_match(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.test_multi3", [
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            [ParamInfo(name="input", type="Tensor")],
        ], source="test")
        case = make_testcase(
            api_name="torch.test_multi3",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32'))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "INPUT_COUNT_EXCEEDED"

    def test_multi_overload_tensorlist_vs_tensor(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.test_multi_tl", [
            [ParamInfo(name="tensors", type="tuple of Tensors")],
            [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
        ], source="test")
        case_nested = make_testcase(
            api_name="torch.test_multi_tl",
            tensor_view_shapes=(((2, 3), (2, 5)),),
            tensor_dtypes=('float32',))
        case_nested.validate()
        assert case_nested.is_valid

        case_flat = make_testcase(
            api_name="torch.test_multi_tl",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case_flat.validate()
        assert case_flat.is_valid


class TestOutputTensorExcludesFromInputCount:
    """Tests for _check_tensor_configuration excluding output_tensor_indexes."""

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_all_tensors_marked_output_invalidates(self, make_testcase):
        """1 tensor with output_tensor_indexes=(0,) → 0 input tensors → invalid."""
        FrameworkApiInfoKeeper().register("torch.abs", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch.abs",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',),
            output_tensor_indexes=(0,))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "ALL_TENSORS_MARKED_OUTPUT"

    def test_output_tensor_excluded_from_count(self, make_testcase):
        """2 tensors, 1 output → 1 input tensor, API needs 1 input → valid."""
        FrameworkApiInfoKeeper().register("torch.abs_valid", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch.abs_valid",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert case.is_valid

    def test_two_outputs_one_input_valid(self, make_testcase):
        """3 tensors, 2 outputs → 1 input. API with 1 required tensor + 2 optional → valid."""
        FrameworkApiInfoKeeper().register("torch.sort_like", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", default="None",
                      is_optional=True, is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch.sort_like",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32'),
            output_tensor_indexes=(1, 2))
        case.validate()
        assert case.is_valid

    def test_live_torch_abs_all_output_invalid(self, make_testcase):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        case = make_testcase(
            api_name="torch.abs",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',),
            output_tensor_indexes=(0,))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "ALL_TENSORS_MARKED_OUTPUT"


class TestInplaceTensorMethod:

    def setup_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        FrameworkApiInfoKeeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_is_inplace_tensor_method_static(self):
        assert TestcaseE2e._is_inplace_tensor_method("torch.Tensor.add_")
        assert TestcaseE2e._is_inplace_tensor_method("torch.Tensor.relu_")
        assert not TestcaseE2e._is_inplace_tensor_method("torch.Tensor.add")
        assert not TestcaseE2e._is_inplace_tensor_method("torch.add")
        assert not TestcaseE2e._is_inplace_tensor_method("")

    def test_auto_fill_output_for_inplace(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.Tensor.fake_add_", [
            ParamInfo(name="self", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
            ParamInfo(name="alpha", type="Number", default="1"),
        ], source="test")
        case = make_testcase(
            api_name="torch.Tensor.fake_add_",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'),
            attributes={'alpha': '2'})
        case.validate()
        assert case.is_valid
        assert case.output_tensor_indexes == (0,)

    def test_inplace_self_not_excluded_from_input_count(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.Tensor.fake_mul_", [
            ParamInfo(name="self", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ], source="test")
        case = make_testcase(
            api_name="torch.Tensor.fake_mul_",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.is_valid

    def test_inplace_pure_output_excludes_self(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.Tensor.fake_sub_", [
            ParamInfo(name="self", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ], source="test")
        case = make_testcase(
            api_name="torch.Tensor.fake_sub_",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.pure_output_indexes == []

    def test_non_inplace_not_auto_filled(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.Tensor.fake_add", [
            ParamInfo(name="self", type="Tensor"),
            ParamInfo(name="other", type="Tensor"),
        ], source="test")
        case = make_testcase(
            api_name="torch.Tensor.fake_add",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'))
        case.validate()
        assert case.output_tensor_indexes is None or case.output_tensor_indexes == ()


class TestOutputConfigurationValidation:

    def test_required_out_missing_fails(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch_npu.test_req_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch_npu.test_req_out",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "MISSING_REQUIRED_OUTPUT"

    def test_required_tensor_list_out_wrong_count_fails(self, make_testcase):
        info = APIParamInfo(
            api_name="torch_npu.test_tl_out",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
            ]],
            _return_counts=[4],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch_npu.test_tl_out", info)
        case = make_testcase(
            api_name="torch_npu.test_tl_out",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32'),
            output_tensor_indexes=(1, 2))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "OUTPUT_COUNT_MISMATCH"

    def test_required_tensor_list_out_correct_count_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch_npu.test_tl_out_ok",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
            ]],
            _return_counts=[4],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch_npu.test_tl_out_ok", info)
        case = make_testcase(
            api_name="torch_npu.test_tl_out_ok",
            tensor_view_shapes=((2, 3), (2, 3), (2, 3), (2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32', 'float32', 'float32', 'float32'),
            output_tensor_indexes=(1, 2, 3, 4))
        case.validate()
        assert case.is_valid

    def test_optional_out_no_output_passes(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.opt_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_optional=True, is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch.opt_out",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',))
        case.validate()
        assert case.is_valid

    def test_optional_out_with_output_passes(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.opt_out2", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_optional=True, is_keyword_only=True),
        ], source="test")
        case = make_testcase(
            api_name="torch.opt_out2",
            tensor_view_shapes=((2, 3), (2, 3)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert case.is_valid

    def test_no_out_param_normal_case(self, make_testcase):
        FrameworkApiInfoKeeper().register("torch.no_out", [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int"),
        ], source="test")
        case = make_testcase(
            api_name="torch.no_out",
            tensor_view_shapes=((2, 3),),
            tensor_dtypes=('float32',),
            attributes={"dim": "0"})
        case.validate()
        assert case.is_valid


class TestOutputConfigUnknownTensorListCount:
    """Test _check_output_configuration when out_expected_count=0 with is_tensor_list=True.

    This happens for APIs parsed from TypeError multi-overload (e.g. torch.sort)
    where we know out is TensorList but don't know the exact count.
    """

    def test_optional_tensor_list_unknown_count_no_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="tuple of Tensors", is_optional=True, is_keyword_only=True),
            ]],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like", info)
        case = make_testcase(
            api_name="torch.sort_like",
            tensor_view_shapes=((4, 4),),
            tensor_dtypes=('float32',))
        case.validate()
        assert case.is_valid

    def test_optional_tensor_list_unknown_count_with_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like2",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="tuple of Tensors", is_optional=True, is_keyword_only=True),
            ]],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like2", info)
        case = make_testcase(
            api_name="torch.sort_like2",
            tensor_view_shapes=((4, 4), (4, 4), (4, 4)),
            tensor_dtypes=('float32', 'float32', 'float32'),
            output_tensor_indexes=(1, 2))
        case.validate()
        assert case.is_valid

    def test_required_tensor_list_unknown_count_with_out_passes(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like3",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
            ]],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like3", info)
        case = make_testcase(
            api_name="torch.sort_like3",
            tensor_view_shapes=((4, 4), (4, 4)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert case.is_valid

    def test_required_tensor_list_unknown_count_no_out_fails(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.sort_like4",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
            ]],
            _return_counts=[0],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.sort_like4", info)
        case = make_testcase(
            api_name="torch.sort_like4",
            tensor_view_shapes=((4, 4),),
            tensor_dtypes=('float32',))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "MISSING_REQUIRED_OUTPUT"

    def test_known_count_exact_validation_still_works(self, make_testcase):
        info = APIParamInfo(
            api_name="torch.exact_tl",
            overloads=[[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="out", type="tuple of Tensors", is_keyword_only=True),
            ]],
            _return_counts=[2],
            source="test",
        )
        FrameworkApiInfoKeeper().register("torch.exact_tl", info)
        case = make_testcase(
            api_name="torch.exact_tl",
            tensor_view_shapes=((4, 4), (4, 4)),
            tensor_dtypes=('float32', 'float32'),
            output_tensor_indexes=(1,))
        case.validate()
        assert not case.is_valid
        assert case.fail_reason == "OUTPUT_COUNT_MISMATCH"
