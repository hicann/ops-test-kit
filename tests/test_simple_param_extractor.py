#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""
Tests for ttk.utilities.simple_param_extractor:
ParamInfo, APIParamInfo, signature parsing, type inference, manual overrides.
"""

import pytest
from ttk.utilities.simple_param_extractor import (
    ParamInfo, APIParamInfo,
    _split_by_comma, _parse_params_from_signature,
    _infer_type_from_name, _infer_type_from_value,
    _normalize_npu_type, _parse_npu_declaration,
    register_api_params, get_api_params, extract_api_params,
    _MANUAL_OVERRIDES,
)


class TestParamInfoProperties:

    def test_is_tensor(self):
        assert ParamInfo(name="x", type="Tensor").is_tensor is True

    def test_is_tensor_lowercase(self):
        assert ParamInfo(name="x", type="tensor").is_tensor is True

    def test_not_tensor(self):
        assert ParamInfo(name="x", type="int").is_tensor is False

    def test_is_tensor_list_tuple_of(self):
        assert ParamInfo(name="x", type="tuple of Tensors").is_tensor_list is True

    def test_is_tensor_list_list_of(self):
        assert ParamInfo(name="x", type="list of Tensors").is_tensor_list is True

    def test_is_tensor_list_bracket(self):
        assert ParamInfo(name="x", type="Tensor[]").is_tensor_list is True

    def test_is_tensor_list_generic(self):
        assert ParamInfo(name="x", type="List[Tensor]").is_tensor_list is True

    def test_is_scalar_types(self):
        for t in ("Number", "Scalar", "int", "float", "bool", "str"):
            assert ParamInfo(name="x", type=t).is_scalar is True

    def test_is_scalar_list_types(self):
        for t in ("Scalar[]", "List[Scalar]", "list of Numbers"):
            assert ParamInfo(name="x", type=t).is_scalar_list is True

    def test_is_tensor_like(self):
        assert ParamInfo(name="x", type="Tensor").is_tensor_like is True
        assert ParamInfo(name="x", type="tuple of Tensors").is_tensor_like is True
        assert ParamInfo(name="x", type="int").is_tensor_like is False

    def test_defaults(self):
        p = ParamInfo(name="x")
        assert p.type == ""
        assert p.default is None
        assert p.is_optional is False
        assert p.is_var_positional is False

    def test_var_positional_flag(self):
        p = ParamInfo(name="tensors", type="Tensor", is_var_positional=True)
        assert p.is_var_positional is True
        assert p.is_tensor is True
        assert p.is_tensor_like is True


class TestAPIParamInfoProperties:

    def test_tensors_filters_correctly(self):
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        assert len(info.tensors) == 2
        assert info.tensor_count == 2

    def test_scalars_filters_correctly(self):
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="alpha", type="float"),
                ParamInfo(name="dim", type="int"),
            ],
        )
        assert len(info.scalars) == 2
        assert info.scalar_count == 2

    def test_tensor_distribution_flat(self):
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
        )
        assert info.tensor_distribution == (0, 0)

    def test_tensor_distribution_with_tensorlist(self):
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="dim", type="int"),
            ],
        )
        assert info.tensor_distribution == (-1,)

    def test_tensor_distribution_mixed(self):
        info = APIParamInfo(
            api_name="test",
            params=[
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="out", type="Tensor"),
            ],
        )
        assert info.tensor_distribution == (-1, 0)

    def test_empty_params(self):
        info = APIParamInfo(api_name="test")
        assert info.tensor_count == 0
        assert info.scalar_count == 0
        assert info.tensor_distribution == ()


class TestSplitByComma:

    def test_simple(self):
        assert _split_by_comma("a, b, c") == ["a", "b", "c"]

    def test_nested_brackets(self):
        assert _split_by_comma("a, (b, c), d") == ["a", "(b, c)", "d"]

    def test_deep_nesting(self):
        assert _split_by_comma("a, [b, (c, d)], e") == ["a", "[b, (c, d)]", "e"]

    def test_empty_string(self):
        assert _split_by_comma("") == []

    def test_no_commas(self):
        assert _split_by_comma("single") == ["single"]

    def test_trailing_comma(self):
        result = _split_by_comma("a, b,")
        assert result == ["a", "b"]


class TestParseParamsFromSignature:

    def test_typed_params(self):
        sig = "Tensor input, Tensor other, Scalar alpha"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 3
        assert params[0].name == "input"
        assert params[0].type == "Tensor"
        assert params[1].name == "other"
        assert params[1].type == "Tensor"
        assert params[2].name == "alpha"
        assert params[2].type == "Scalar"

    def test_optional_params(self):
        sig = "Tensor input, int dim=-1, bool keepdim=False"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 3
        assert params[1].is_optional is True
        assert params[1].default == -1
        assert params[2].is_optional is True

    def test_keyword_only_marker(self):
        sig = "Tensor input, *, int dim"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 2
        assert params[1].is_optional is True

    def test_tuple_of_tensors(self):
        sig = "tuple of Tensors tensors, int dim"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert params[0].type == "tuple of Tensors"
        assert params[0].name == "tensors"

    def test_empty_signature(self):
        assert _parse_params_from_signature("") is None
        assert _parse_params_from_signature("  ") is None

    def test_names_only_fallback(self):
        sig = "input, dim, keepdim"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 3
        assert params[0].type == "Tensor"
        assert params[1].type == "int"
        assert params[2].type == "bool"


class TestInferTypeFromName:

    def test_tensor_names(self):
        for name in ("input", "output", "other", "self", "tensor", "weight"):
            assert _infer_type_from_name(name) == "Tensor"

    def test_tensor_list_names(self):
        for name in ("tensors", "inputs", "targets"):
            assert _infer_type_from_name(name) == "tuple of Tensors"

    def test_scalar_float_names(self):
        for name in ("alpha", "beta", "gamma", "epsilon", "dropout", "p"):
            assert _infer_type_from_name(name) == "float"

    def test_scalar_int_names(self):
        for name in ("dim", "axis", "size", "stride", "padding", "dilation"):
            assert _infer_type_from_name(name) == "int"

    def test_special_bool_names(self):
        assert _infer_type_from_name("inplace") == "bool"
        assert _infer_type_from_name("training") == "bool"
        assert _infer_type_from_name("requires_grad") == "bool"
        assert _infer_type_from_name("pin_memory") == "bool"

    def test_special_dtype(self):
        assert _infer_type_from_name("dtype") == "Dtype"

    def test_unknown_defaults_tensor(self):
        assert _infer_type_from_name("unknown_var") == "Tensor"

    def test_case_insensitive(self):
        assert _infer_type_from_name("Input") == "Tensor"
        assert _infer_type_from_name("DIM") == "int"


class TestInferTypeFromValue:

    def test_none(self):
        assert _infer_type_from_value("None") == "Optional[Tensor]"

    def test_bool(self):
        assert _infer_type_from_value("True") == "bool"
        assert _infer_type_from_value("False") == "bool"

    def test_string(self):
        assert _infer_type_from_value("'abc'") == "str"
        assert _infer_type_from_value('"abc"') == "str"

    def test_float(self):
        assert _infer_type_from_value("1.5") == "float"
        assert _infer_type_from_value("1e-3") == "float"

    def test_int(self):
        assert _infer_type_from_value("42") == "int"
        assert _infer_type_from_value("-1") == "int"


class TestNormalizeNpuType:

    def test_known_types(self):
        assert _normalize_npu_type("Tensor") == "Tensor"
        assert _normalize_npu_type("TensorList") == "tuple of Tensors"
        assert _normalize_npu_type("Scalar") == "Number"
        assert _normalize_npu_type("ScalarList") == "list of Numbers"

    def test_nullable_types(self):
        assert _normalize_npu_type("int?") == "int"
        assert _normalize_npu_type("float?") == "float"
        assert _normalize_npu_type("bool?") == "bool"

    def test_passthrough(self):
        assert _normalize_npu_type("CustomType") == "CustomType"


class TestParseNpuDeclaration:

    def test_basic(self):
        sig = "Tensor input, Tensor other, Scalar alpha"
        params = _parse_npu_declaration(sig)
        assert params is not None
        assert len(params) == 3
        assert params[0].name == "input"
        assert params[0].type == "Tensor"

    def test_optional_param(self):
        sig = "Tensor input, int? dim=None"
        params = _parse_npu_declaration(sig)
        assert params is not None
        assert params[1].is_optional is True

    def test_tensor_list(self):
        sig = "TensorList tensors, int dim"
        params = _parse_npu_declaration(sig)
        assert params is not None
        assert params[0].type == "tuple of Tensors"

    def test_empty(self):
        assert _parse_npu_declaration("") is None
        assert _parse_npu_declaration(None) is None

    def test_return_type_skipped(self):
        sig = "Tensor input -> Tensor"
        params = _parse_npu_declaration(sig)
        assert params is not None
        assert len(params) == 1

    def test_star_makes_keyword_only(self):
        sig = "Tensor input, *, Tensor? smooth_scales=None, Tensor? group_index=None, ScalarType? dst_type=None"
        params = _parse_npu_declaration(sig)
        assert params is not None
        assert len(params) == 4
        assert params[0].name == "input"
        assert params[0].is_keyword_only is False
        assert params[1].name == "smooth_scales"
        assert params[1].is_keyword_only is True
        assert params[2].name == "group_index"
        assert params[2].is_keyword_only is True
        assert params[3].name == "dst_type"
        assert params[3].is_keyword_only is True

    def test_no_star_all_positional(self):
        sig = "Tensor input, Tensor other, Scalar alpha=1"
        params = _parse_npu_declaration(sig)
        assert params is not None
        for p in params:
            assert p.is_keyword_only is False


class TestManualOverrides:

    def setup_method(self):
        _MANUAL_OVERRIDES.clear()

    def test_register_and_get(self):
        params = [ParamInfo(name="input", type="Tensor"), ParamInfo(name="dim", type="int")]
        register_api_params("torch.custom_op", params, source="test")
        info = get_api_params("torch.custom_op")
        assert info is not None
        assert info.tensor_count == 1
        assert info.source == "test"

    def test_override_takes_priority(self):
        params = [ParamInfo(name="x", type="Tensor")]
        register_api_params("torch.add", params, source="test_override")
        info = get_api_params("torch.add")
        assert info is not None
        assert info.source == "test_override"

    def test_no_match_returns_none_for_unknown(self):
        info = get_api_params("nonexistent.module.func")
        assert info is None

    def teardown_method(self):
        _MANUAL_OVERRIDES.clear()


class TestExtractApiParamsTorch:
    """Tests that require torch to be importable."""

    def test_torch_add(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.add")
        assert info is not None
        assert info.tensor_count >= 2

    def test_torch_block_diag_var_positional(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.block_diag")
        assert info is not None
        assert len(info.params) == 1
        assert info.params[0].is_var_positional is True
        assert info.params[0].is_tensor is True
        assert info.params[0].name == "tensors"

    def test_torch_cat(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.cat")
        assert info is not None
        has_tensorlist = any(p.is_tensor_list for p in info.tensors)
        assert has_tensorlist, f"torch.cat should have TensorList param, got {info.tensors}"

    def test_torch_stack(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.stack")
        if info is None:
            pytest.skip("torch.stack uses simple TypeError, no type info available")
        has_tensorlist = any(p.is_tensor_list for p in info.tensors)
        assert has_tensorlist, f"torch.stack should have TensorList param, got {info.tensors}"


class TestAPIParamInfoMultiOverload:

    def test_overloads_init_from_list(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Number")],
            ],
            source="test")
        assert len(info.overloads) == 2
        assert info.params == info.overloads[0].params
        assert info.tensor_count == 2

    def test_overloads_backward_compat(self):
        params = [ParamInfo(name="input", type="Tensor")]
        info = APIParamInfo(api_name="test", params=params, source="test")
        assert len(info.overloads) == 1
        assert info.overloads[0].params == params

    def test_match_overload_count_only(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor"),
                 ParamInfo(name="out", type="Tensor", default="None", is_optional=True)],
                [ParamInfo(name="input", type="Tensor")],
            ],
            source="test")
        matched, _, oidx = info.match_overload(2)
        assert matched
        assert oidx == 0
        matched, _, oidx = info.match_overload(1)
        assert matched
        assert oidx == 1
        matched, _, _ = info.match_overload(4)
        assert not matched

    def test_match_overload_type_check(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="tensors", type="tuple of Tensors")],
                [ParamInfo(name="input", type="Tensor")],
            ],
            source="test")
        matched, _, oidx = info.match_overload(1, [False])
        assert matched
        assert oidx == 1
        matched, _, oidx = info.match_overload(1, [True])
        assert matched
        assert oidx == 0

    def test_match_overload_no_match(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="input", type="Tensor"), ParamInfo(name="other", type="Tensor")],
            ],
            source="test")
        matched, _, _ = info.match_overload(3)
        assert not matched
        matched, _, _ = info.match_overload(0)
        assert not matched

    def test_match_overload_skip_flags(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="tensors", type="tuple of Tensors"),
                 ParamInfo(name="out", type="Tensor", default="None", is_optional=True)],
            ],
            source="test")
        matched, _, oidx = info.match_overload(1, [False], [True])
        assert matched
        assert oidx == 0

    def test_live_torch_div_multi_overload(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.div")
        assert info is not None
        assert len(info.overloads) >= 2
        matched, tensors, oidx = info.match_overload(2)
        assert matched
        matched, tensors, oidx = info.match_overload(1)
        assert matched
        assert oidx >= 0


class TestTensorMethodSelfInjection:

    def test_resolve_tensor_method(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from ttk.utilities.simple_param_extractor import _resolve_function
        obj = _resolve_function("torch.Tensor.add_")
        assert obj is not None

    def test_resolve_tensor_method_nonexistent(self):
        from ttk.utilities.simple_param_extractor import _resolve_function
        obj = _resolve_function("torch.Tensor.nonexistent_method_xyz")
        assert obj is None

    def test_get_api_params_injects_self(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = get_api_params("torch.Tensor.add_")
        assert info is not None
        assert info.params[0].name == "self"
        assert info.params[0].type == "Tensor"
        assert info.tensor_count >= 2

    def test_is_tensor_method_util(self):
        from ttk.utilities.simple_param_extractor import _is_tensor_method
        assert _is_tensor_method("torch.Tensor.add_")
        assert _is_tensor_method("torch.Tensor.relu_")
        assert not _is_tensor_method("torch.add")
        assert not _is_tensor_method("torch.ops.aten.add")
        assert not _is_tensor_method("")


class TestNormalizeArgsTypeUnion:
    """Tests for _normalize_args_type union type handling (comma-separated and 'or')."""

    def test_comma_separated_union(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        # "int, float" should produce union type, not just take first
        result = _normalize_args_type("int, float")
        assert result == "int|float"

    def test_comma_separated_with_enum_values(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        # torch.norm p parameter: int, float, inf, -inf, 'fro', 'nuc'
        result = _normalize_args_type("int, float, inf, -inf, 'fro', 'nuc'")
        assert result == "int|float|str"

    def test_comma_separated_with_tuple_variants(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        result = _normalize_args_type("int, tuple of ints, list of ints")
        assert result == "int|tuple of ints"

    def test_comma_separated_single_type(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        result = _normalize_args_type("bool")
        assert result == "bool"

    def test_or_union_int_tuple(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        result = _normalize_args_type("int or Tuple[int]")
        assert result == "int|tuple of ints"

    def test_or_union_float_tuple(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        result = _normalize_args_type("float or Tuple[float]")
        assert result == "float|tuple of floats"

    def test_or_union_tensor_filtered(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        # Tensor members should be excluded (handled by overload resolution)
        result = _normalize_args_type("Tensor or Number")
        assert result == "Number"

    def test_or_union_tensor_float_filtered(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        result = _normalize_args_type("Tensor or float")
        assert result == "float"

    def test_tuple_bracket_types(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        assert _normalize_args_type("Tuple[int]") == "tuple of ints"
        assert _normalize_args_type("Tuple[float]") == "tuple of floats"
        assert _normalize_args_type("Tuple[int, int]") == "tuple of ints"

    def test_comma_separated_dedup(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        # Duplicate types should be deduplicated
        result = _normalize_args_type("int, int, float")
        assert result == "int|float"

    def test_quoted_string_values_recognized_as_str(self):
        from ttk.utilities.simple_param_extractor import _normalize_args_type
        # Single-quoted values in comma-separated list → 'str'
        result = _normalize_args_type("'constant', 'reflect', 'replicate'")
        assert result == "str"


class TestCoerceValueUnion:
    """Tests for _coerce_value with union types (containing '|')."""

    def test_int_or_float_or_str_with_int(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        assert coerce_value("2", "int|float|str") == 2

    def test_int_or_float_or_str_with_float(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        assert coerce_value("2.0", "int|float|str") == 2.0

    def test_int_or_float_or_str_with_str(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        assert coerce_value("fro", "int|float|str") == "fro"

    def test_int_or_tuple_of_ints_with_int(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        assert coerce_value("3", "int|tuple of ints") == 3

    def test_int_or_tuple_of_ints_with_list(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        assert coerce_value("[1,2,3]", "int|tuple of ints") == (1, 2, 3)

    def test_union_all_fail_raises(self):
        from ttk.core_modules.testcase_manager.param_plan import coerce_value
        import pytest
        with pytest.raises(ValueError, match="union type"):
            coerce_value("not_a_number", "int|float")


class TestEnrichTypesFromAnnotations:
    """Tests for _enrich_types_from_annotations preserving union types."""

    def test_union_type_not_overwritten_by_annotations(self):
        from ttk.utilities.simple_param_extractor import (
            ParamInfo, _enrich_types_from_annotations
        )
        # Simulate: doc_result has p='int|float|str' from Args section
        # annotations has p='float' — union should NOT be overwritten
        params = [ParamInfo(name='p', type='int|float|str', default=None)]
        ann_params = [ParamInfo(name='p', type='float', default='fro')]
        _enrich_types_from_annotations(params, ann_params)
        assert params[0].type == 'int|float|str'
        # But default should be filled from annotations
        assert params[0].default == 'fro'

    def test_single_inferred_type_overwritten(self):
        from ttk.utilities.simple_param_extractor import (
            ParamInfo, _enrich_types_from_annotations
        )
        # Single inferred type 'int' SHOULD be overwritten by annotations
        params = [ParamInfo(name='x', type='int', default=None)]
        ann_params = [ParamInfo(name='x', type='float', default=1.0)]
        _enrich_types_from_annotations(params, ann_params)
        assert params[0].type == 'float'

    def test_non_inferred_type_not_overwritten(self):
        from ttk.utilities.simple_param_extractor import (
            ParamInfo, _enrich_types_from_annotations, _is_inferred_type
        )
        # 'Dtype' is in _INFERRED (from _infer_type_from_name), so it WILL
        # be overwritten by annotations — that's correct behavior.
        # Test a type that is NOT in _INFERRED and won't be overwritten.
        params = [ParamInfo(name='alpha', type='int|float', default=None)]
        ann_params = [ParamInfo(name='alpha', type='float', default=None)]
        _enrich_types_from_annotations(params, ann_params)
        assert params[0].type == 'int|float'


class TestNnFunctionalPyi:
    """Tests for torch.nn.functional.xxx pyi loading (reexport mapping + def parsing)."""

    def setup_method(self):
        import ttk.utilities.simple_param_extractor as spe
        spe._PYI_CACHE = None

    def teardown_method(self):
        import ttk.utilities.simple_param_extractor as spe
        spe._PYI_CACHE = None

    def test_conv_tbc_reexport_resolved(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.nn.functional.conv_tbc")
        assert info is not None
        assert info.source == "pyi-stub"
        assert info.tensor_count == 3
        param_names = [p.name for p in info.params]
        assert "input" in param_names
        assert "weight" in param_names
        assert "bias" in param_names
        assert "pad" in param_names

    def test_max_unpool2d_def_signature(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.nn.functional.max_unpool2d")
        assert info is not None
        assert info.tensor_count >= 2
        param_names = [p.name for p in info.params]
        assert "input" in param_names
        assert "indices" in param_names

    def test_pyi_cache_has_nn_functional_entries(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from ttk.utilities.simple_param_extractor import _load_pyi_signatures
        sigs = _load_pyi_signatures()
        nn_keys = [k for k in sigs if k.startswith("torch.nn.functional.")]
        assert len(nn_keys) > 0

    def test_reexport_same_params_as_torch_original(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from ttk.utilities.simple_param_extractor import _load_pyi_signatures
        sigs = _load_pyi_signatures()
        torch_sigs = sigs.get("torch.conv_tbc")
        nn_sigs = sigs.get("torch.nn.functional.conv_tbc")
        assert torch_sigs is not None
        assert nn_sigs is not None
        assert len(torch_sigs[0]) == len(nn_sigs[0])
        for tp, np_ in zip(torch_sigs[0], nn_sigs[0]):
            assert tp.name == np_.name
            assert tp.type == np_.type

    def test_nonexistent_nn_functional_returns_none(self):
        info = extract_api_params("torch.nn.functional.nonexistent_api_xyz")
        assert info is None

    def test_conv1d_nn_functional_has_overloads(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        from ttk.utilities.simple_param_extractor import _load_pyi_signatures
        sigs = _load_pyi_signatures()
        nn_sigs = sigs.get("torch.nn.functional.conv1d")
        assert nn_sigs is not None
        assert len(nn_sigs) >= 1


class TestParseSignatureVarPositional:
    """Tests for _parse_params_from_signature recognizing *args prefix."""

    def test_star_prefix_marks_var_positional(self):
        sig = "*operands: Tensor"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 1
        assert params[0].name == "operands"
        assert params[0].is_var_positional is True
        assert params[0].type == "Tensor"

    def test_star_prefix_without_type(self):
        sig = "equation, *operands"
        params = _parse_params_from_signature(sig)
        assert params is not None
        names = [p.name for p in params]
        assert "operands" in names
        op = next(p for p in params if p.name == "operands")
        assert op.is_var_positional is True

    def test_no_star_not_var_positional(self):
        sig = "input: Tensor, dim: int"
        params = _parse_params_from_signature(sig)
        assert params is not None
        for p in params:
            assert p.is_var_positional is False

    def test_mixed_normal_and_var_positional(self):
        sig = "equation: str, *operands: Tensor"
        params = _parse_params_from_signature(sig)
        assert params is not None
        assert len(params) == 2
        assert params[0].name == "equation"
        assert params[0].is_var_positional is False
        assert params[1].name == "operands"
        assert params[1].is_var_positional is True


class TestMatchOverloadVarPositional:
    """Tests for match_overload handling VAR_POSITIONAL (*args) tensor params."""

    def test_var_pos_accepts_any_count_above_required(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="tensors", type="Tensor", is_var_positional=True)],
            ],
            source="test")
        matched, tensors, oidx = info.match_overload(2)
        assert matched
        matched, tensors, oidx = info.match_overload(5)
        assert matched
        matched, tensors, oidx = info.match_overload(1)
        assert matched

    def test_var_pos_skips_nested_flag_check(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="equation", type="str"),
                 ParamInfo(name="operands", type="tuple of Tensors", is_var_positional=True)],
            ],
            source="test")
        matched, _, _ = info.match_overload(2, [False, False])
        assert matched

    def test_non_var_pos_still_enforces_upper_bound(self):
        info = APIParamInfo(
            api_name="test",
            overloads=[
                [ParamInfo(name="input", type="Tensor"),
                 ParamInfo(name="other", type="Tensor")],
            ],
            source="test")
        matched, _, _ = info.match_overload(3)
        assert not matched

    def test_live_torch_einsum_var_positional(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.einsum")
        assert info is not None
        var_pos_params = [p for p in info.params if p.is_var_positional]
        assert len(var_pos_params) >= 1
        assert var_pos_params[0].name == "operands"
        matched, _, _ = info.match_overload(2, [False, False])
        assert matched


class TestMeshgridExtraction:
    """Tests for torch.meshgrid full extraction chain (docstring + inspect upgrade)."""

    def test_meshgrid_has_var_positional_tensor(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.meshgrid")
        assert info is not None
        var_pos = [p for p in info.params if p.is_var_positional]
        assert len(var_pos) == 1
        assert var_pos[0].name == "tensors"
        assert var_pos[0].is_tensor_like

    def test_meshgrid_indexing_is_keyword_only(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.meshgrid")
        indexing = next((p for p in info.params if p.name == "indexing"), None)
        assert indexing is not None
        assert indexing.is_keyword_only
        assert indexing.type == "str"

    def test_meshgrid_match_overload_with_two_tensors(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.meshgrid")
        matched, _, _ = info.match_overload(2, [False, False])
        assert matched


class TestDocstringArgsIndentation:
    """Tests for _parse_docstring_args_section relative indentation handling."""

    def test_deeply_indented_args_section(self):
        from ttk.utilities.simple_param_extractor import _parse_docstring_args_section
        doc = """Some description.

        Args:
            tensors (list of Tensor): list of tensors.
                Continuation line that should be skipped.
            indexing: (str, optional): the indexing mode.

        Returns:
            Something.
"""
        result = _parse_docstring_args_section(doc)
        assert "tensors" in result
        assert result["tensors"] == ("list of Tensor", False)

    def test_standard_indented_args_section(self):
        from ttk.utilities.simple_param_extractor import _parse_docstring_args_section
        doc = """Some description.

Args:
    input (Tensor): the input.
    dim (int): dimension.

Returns:
    Something.
"""
        result = _parse_docstring_args_section(doc)
        assert "input" in result
        assert "dim" in result


class TestInspectKeywordOnly:
    """Tests for inspect.signature mapping KEYWORD_ONLY to ParamInfo."""

    def test_keyword_only_param_detected(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.meshgrid")
        indexing = next(p for p in info.params if p.name == "indexing")
        assert indexing.is_keyword_only

    def test_positional_param_not_keyword_only(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.add")
        input_p = next(p for p in info.params if p.name == "input")
        assert not input_p.is_keyword_only


class TestAtenSchemaExtraction:
    """Tests for torch.ops.aten.* parameter extraction via _schemas."""

    def test_aten_convolution_basic(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        assert info is not None
        assert "aten._schemas" in info.source
        assert info.tensor_count >= 2
        names = [p.name for p in info.params]
        assert "input" in names
        assert "weight" in names
        assert "stride" in names
        assert "padding" in names
        assert "groups" in names

    def test_aten_convolution_has_out_overload(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        assert info is not None
        assert len(info.overloads) == 2
        out_overload = info.overloads[1].params
        out_params = [p for p in out_overload if p.name == "out"]
        assert len(out_params) == 1
        assert out_params[0].is_keyword_only

    def test_aten_add_multi_overload(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.add")
        assert info is not None
        assert len(info.overloads) >= 10
        tensor_overloads = []
        for ov in info.overloads:
            tc = sum(1 for p in ov.params if p.is_tensor_like)
            if tc >= 1:
                tensor_overloads.append(ov.params)
        assert len(tensor_overloads) >= 4

    def test_aten_abs_simple(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.abs")
        assert info is not None
        assert info.tensor_count == 1
        assert info.params[0].name == "self"
        assert info.params[0].type == "Tensor"

    def test_aten_convolution_tensor_attr_separation(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        assert info is not None
        tensors = info.tensors
        scalars = info.scalars
        tensor_names = [p.name for p in tensors]
        scalar_names = [p.name for p in scalars]
        assert "input" in tensor_names
        assert "weight" in tensor_names
        assert "stride" not in tensor_names
        assert "padding" not in tensor_names
        assert "transposed" in scalar_names
        assert "groups" in scalar_names

    def test_aten_optional_tensor_bias(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        bias_param = next((p for p in info.params if p.name == "bias"), None)
        assert bias_param is not None
        assert bias_param.is_tensor
        assert bias_param.is_optional

    def test_aten_default_values(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.add")
        assert info is not None
        tensor_ov = None
        for ov in info.overloads:
            names = [p.name for p in ov.params]
            if "alpha" in names and "self" in names:
                if sum(1 for p in ov.params if p.is_tensor_like) >= 2:
                    tensor_ov = ov.params
                    break
        assert tensor_ov is not None
        alpha = next(p for p in tensor_ov if p.name == "alpha")
        assert alpha.is_optional
        assert alpha.default == 1

    def test_aten_matmul(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.matmul")
        assert info is not None
        assert info.tensor_count >= 2

    def test_aten_relu(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.relu")
        assert info is not None
        assert info.tensor_count >= 1

    def test_aten_nonexistent_returns_none(self):
        info = extract_api_params("torch.ops.aten.nonexistent_op_xyz")
        assert info is None

    def test_aten_type_mapping_optional(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("Optional[Tensor]") == "Tensor"
        assert _normalize_npu_type("Optional[int]") == "int"

    def test_aten_type_mapping_list(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("List[int]") == "tuple of ints"
        assert _normalize_npu_type("List[Tensor]") == "tuple of Tensors"

    def test_aten_type_mapping_symint(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("SymInt") == "int"
        assert _normalize_npu_type("SymInt[]") == "tuple of ints"

    def test_aten_convolution_match_overload_2tensors(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        matched, tensors, oidx = info.match_overload(2)
        assert matched
        assert len(tensors) >= 2

    def test_aten_convolution_match_overload_3tensors_with_out(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        matched, tensors, oidx = info.match_overload(3)
        assert matched
        assert len(info.overloads) == 2
        out_ov = info.overloads[1].params
        assert any(p.name == "out" and p.is_keyword_only for p in out_ov)

    def test_aten_npu_conv2d_not_affected(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch_npu.npu_conv2d")
        if info is not None:
            assert "torch_npu" in info.source

    def test_aten_source_format(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.convolution")
        assert "aten._schemas" in info.source
        assert "2 overloads" in info.source

    def test_aten_cat_tensor_list_and_default_dim(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.cat")
        assert info is not None
        tensors_param = next(p for p in info.params if p.name == "tensors")
        assert tensors_param.is_tensor_list
        dim_param = next(p for p in info.params if p.name == "dim")
        assert dim_param.is_optional
        assert dim_param.default == 0

    def test_aten_stack_tensor_list_and_default_dim(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.stack")
        assert info is not None
        tensors_param = next(p for p in info.params if p.name == "tensors")
        assert tensors_param.is_tensor_list
        dim_param = next(p for p in info.params if p.name == "dim")
        assert dim_param.is_optional
        assert dim_param.default == 0

    def test_aten_where_self_three_tensors(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.where")
        assert info is not None
        matched, tensors, oidx = info.match_overload(3)
        assert matched
        tensor_names = [p.name for p in tensors]
        assert "condition" in tensor_names
        assert "self" in tensor_names
        assert "other" in tensor_names

    def test_aten_eq_21_overloads(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.eq")
        assert info is not None
        assert len(info.overloads) == 21

    def test_aten_batch_norm_many_optional_tensors(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.batch_norm")
        assert info is not None
        for name in ("weight", "bias", "running_mean", "running_var"):
            param = next((p for p in info.params if p.name == name), None)
            assert param is not None, f"Missing param: {name}"
            assert param.is_tensor, f"{name} should be tensor"
            assert param.is_optional, f"{name} should be optional (Optional[Tensor])"
        training = next(p for p in info.params if p.name == "training")
        assert training.type == "bool"

    def test_aten_mean_dim_defaults_and_keyword_only(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.mean")
        assert info is not None
        dim_ov = None
        for ov in info.overloads:
            names = [p.name for p in ov.params]
            if "keepdim" in names and "self" in names:
                non_kw = [p for p in ov.params if not p.is_keyword_only]
                if any(p.name == "dim" for p in non_kw):
                    dim_ov = ov
                    break
        assert dim_ov is not None
        keepdim = next(p for p in dim_ov.params if p.name == "keepdim")
        assert keepdim.is_optional
        assert keepdim.default is False
        dtype_param = next((p for p in dim_ov.params if p.name == "dtype"), None)
        if dtype_param:
            assert dtype_param.is_keyword_only

    def test_aten_sort_defaults(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.sort")
        assert info is not None
        default_ov = None
        for ov in info.overloads:
            names = [p.name for p in ov.params]
            if "dim" in names and "descending" in names and "stable" not in names:
                tensor_params = [p for p in ov.params if p.is_tensor]
                if len(tensor_params) == 1:
                    default_ov = ov
                    break
        assert default_ov is not None
        dim_p = next(p for p in default_ov.params if p.name == "dim")
        assert dim_p.default == -1
        desc_p = next(p for p in default_ov.params if p.name == "descending")
        assert desc_p.default is False

    def test_aten_div_rounding_mode_keyword_only(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten.div")
        assert info is not None
        mode_ov = None
        for ov in info.overloads:
            names = [p.name for p in ov.params]
            if "rounding_mode" in names:
                mode_ov = ov
                break
        assert mode_ov is not None
        rm = next(p for p in mode_ov.params if p.name == "rounding_mode")
        assert rm.is_keyword_only
        assert rm.is_optional

    def test_aten_unsafe_index_list_optional_tensor(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        info = extract_api_params("torch.ops.aten._unsafe_index")
        assert info is not None
        indices_param = next((p for p in info.params if p.name == "indices"), None)
        assert indices_param is not None
        assert indices_param.is_tensor_list

    def test_normalize_npu_type_optional_list(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("Optional[List[int]]") == "tuple of ints"
        assert _normalize_npu_type("Optional[List[Tensor]]") == "tuple of Tensors"

    def test_normalize_npu_type_list_optional_tensor(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("List[Optional[Tensor]]") == "tuple of Tensors"

    def test_normalize_npu_type_number_and_complex(self):
        from ttk.utilities.simple_param_extractor import _normalize_npu_type
        assert _normalize_npu_type("number") == "Number"
        assert _normalize_npu_type("complex") == "Number"
        assert _normalize_npu_type("ScalarType") == "torch.dtype"


class TestOverloadTensorLayout:

    def test_layout_no_out(self):
        from ttk.utilities.simple_param_extractor import OverloadTensorLayout
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int"),
        ]
        layout = OverloadTensorLayout.build(params, return_count=1)
        assert layout.input_count == 1
        assert layout.required_input_count == 1
        assert layout.has_var_input is False
        assert layout.out_param is None
        assert layout.is_out_required is False
        assert layout.is_out_tensor_list is False
        assert layout.out_expected_count == 0
        assert len(layout.input_params) == 1

    def test_layout_optional_single_out(self):
        from ttk.utilities.simple_param_extractor import OverloadTensorLayout
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_optional=True, is_keyword_only=True),
        ]
        layout = OverloadTensorLayout.build(params, return_count=1)
        assert layout.input_count == 1
        assert layout.out_param is not None
        assert layout.is_out_required is False
        assert layout.is_out_tensor_list is False
        assert layout.out_expected_count == 1

    def test_layout_required_single_out(self):
        from ttk.utilities.simple_param_extractor import OverloadTensorLayout
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor", is_keyword_only=True),
        ]
        layout = OverloadTensorLayout.build(params, return_count=1)
        assert layout.out_param is not None
        assert layout.is_out_required is True
        assert layout.is_out_tensor_list is False
        assert layout.out_expected_count == 1

    def test_layout_required_tensor_list_out(self):
        from ttk.utilities.simple_param_extractor import OverloadTensorLayout
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ]
        layout = OverloadTensorLayout.build(params, return_count=4)
        assert layout.out_param is not None
        assert layout.is_out_required is True
        assert layout.is_out_tensor_list is True
        assert layout.out_expected_count == 4

    def test_layout_optional_tensor_list_out(self):
        from ttk.utilities.simple_param_extractor import OverloadTensorLayout
        params = [
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor[]", is_optional=True, is_keyword_only=True),
        ]
        layout = OverloadTensorLayout.build(params, return_count=3)
        assert layout.out_param is not None
        assert layout.is_out_required is False
        assert layout.is_out_tensor_list is True
        assert layout.out_expected_count == 3


class TestOverloadInfo:

    def test_auto_layout(self):
        from ttk.utilities.simple_param_extractor import OverloadInfo, OverloadTensorLayout
        ov = OverloadInfo(params=[
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="dim", type="int"),
        ], return_count=1)
        assert isinstance(ov.layout, OverloadTensorLayout)
        assert ov.layout.input_count == 1

    def test_pickle_roundtrip(self):
        import pickle
        from ttk.utilities.simple_param_extractor import OverloadInfo
        ov = OverloadInfo(params=[
            ParamInfo(name="input", type="Tensor"),
            ParamInfo(name="out", type="Tensor[]", is_keyword_only=True),
        ], return_count=4)
        data = pickle.dumps(ov)
        restored = pickle.loads(data)
        assert restored.return_count == 4
        assert restored.layout.is_out_tensor_list is True
        assert restored.layout.out_expected_count == 4
        assert len(restored.params) == 2

    def test_api_param_info_pickle(self):
        import pickle
        info = APIParamInfo(
            api_name="torch.test_op",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="dim", type="int"),
            ],
            source="test",
        )
        data = pickle.dumps(info)
        restored = pickle.loads(data)
        assert restored.api_name == "torch.test_op"
        assert len(restored.overloads) == 1
        assert len(restored.params) == 2


class TestParseReturnCountFromFirstLine:

    def test_single_return(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line("add(a, b) -> Tensor") == 1

    def test_tuple_return(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line(
            "sort(input, *, out=None) -> (Tensor, LongTensor)") == 2

    def test_triple_return(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line(
            "svd(input, *, out=None) -> (Tensor, Tensor, Tensor)") == 3

    def test_no_annotation(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line("split(input, size)") == 0

    def test_none_return(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line("foo(x) -> None") == 0

    def test_unresolvable_type(self):
        from ttk.utilities.simple_param_extractor import _parse_return_count_from_first_line
        assert _parse_return_count_from_first_line(
            "sort(input) -> torch.return_types.sort") == 0


class TestDocstringKeywordArgs:

    def test_keyword_args_lowercase_parsed(self):
        from ttk.utilities.simple_param_extractor import _parse_docstring_args_section
        doc = """
func(input, *, out=None) -> (Tensor, LongTensor)
Args:
    input (Tensor): the input tensor.
Keyword args:
    out (tuple, optional): output buffers.
Returns: something.
"""
        args = _parse_docstring_args_section(doc)
        assert 'input' in args
        assert 'out' in args
        assert args['out'] == ('tuple', True)

    def test_keyword_args_uppercase_parsed(self):
        from ttk.utilities.simple_param_extractor import _parse_docstring_args_section
        doc = """
func(input, *, out=None)
Args:
    input (Tensor): the input tensor.
Keyword Args:
    out (Tensor, optional): output buffer.
"""
        args = _parse_docstring_args_section(doc)
        assert 'out' in args

    def test_keyword_arguments_parsed(self):
        from ttk.utilities.simple_param_extractor import _parse_docstring_args_section
        doc = """
func(input, *, out=None)
Args:
    input (Tensor): the input tensor.
Keyword arguments:
    out (Tensor, optional): output buffer.
"""
        args = _parse_docstring_args_section(doc)
        assert 'out' in args


class TestDocstringReturnCountAndOutUpgrade:

    def test_topk_return_count(self):
        info = extract_api_params('torch.topk')
        assert info is not None
        assert info.overloads[0].return_count == 2

    def test_svd_return_count(self):
        info = extract_api_params('torch.svd')
        assert info is not None
        assert info.overloads[0].return_count == 3

    def test_topk_out_is_tensor_list(self):
        info = extract_api_params('torch.topk')
        assert info is not None
        ov = info.overloads[0]
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 2

    def test_svd_out_is_tensor_list(self):
        info = extract_api_params('torch.svd')
        assert info is not None
        ov = info.overloads[0]
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 3

    def test_add_return_count_single(self):
        info = extract_api_params('torch.add')
        assert info is not None
        assert info.overloads[0].return_count == 1

    def test_linalg_eig_return_count(self):
        info = extract_api_params('torch.linalg.eig')
        assert info is not None
        assert info.overloads[0].return_count == 2
        assert info.overloads[0].layout.is_out_tensor_list is True

    def test_frexp_return_count_and_out(self):
        info = extract_api_params('torch.frexp')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 2
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 2

    def test_geqrf_return_count_and_out(self):
        info = extract_api_params('torch.geqrf')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 2
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 2

    def test_lu_unpack_return_count_and_out(self):
        info = extract_api_params('torch.lu_unpack')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 3
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 3

    def test_linalg_svd_return_count_and_out(self):
        info = extract_api_params('torch.linalg.svd')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 3
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 3

    def test_linalg_qr_return_count_and_out(self):
        info = extract_api_params('torch.linalg.qr')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 2
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 2


class TestNpuHansDecodeEncode:
    """Focused coverage for torch_npu.npu_hans_decode/encode.

    These APIs exercised the Tensor[] out and required-out features.
    """

    def test_decode_signature(self):
        info = extract_api_params('torch_npu.npu_hans_decode')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 1
        assert ov.layout.input_count == 4
        assert ov.layout.is_out_required is True
        assert ov.layout.is_out_tensor_list is False
        assert ov.layout.out_expected_count == 1
        params_by_name = {p.name: p for p in ov.params}
        assert params_by_name['mantissa'].is_tensor
        assert params_by_name['fixed'].is_tensor
        assert params_by_name['var'].is_tensor
        assert params_by_name['pdf'].is_tensor
        assert params_by_name['reshuff'].is_optional
        assert params_by_name['out'].is_keyword_only
        assert not params_by_name['out'].is_optional

    def test_encode_signature(self):
        info = extract_api_params('torch_npu.npu_hans_encode')
        assert info is not None
        ov = info.overloads[0]
        assert ov.return_count == 4
        assert ov.layout.input_count == 1
        assert ov.layout.is_out_required is True
        assert ov.layout.is_out_tensor_list is True
        assert ov.layout.out_expected_count == 4
        params_by_name = {p.name: p for p in ov.params}
        assert params_by_name['input'].is_tensor
        assert params_by_name['statistic'].is_optional
        assert params_by_name['reshuff'].is_optional
        assert params_by_name['out'].is_keyword_only
        assert params_by_name['out'].is_tensor_list
        assert not params_by_name['out'].is_optional

    def test_encode_match_overload(self):
        info = extract_api_params('torch_npu.npu_hans_encode')
        matched, params, oidx = info.match_overload(1, ((),), False)
        assert matched
        assert oidx == 0

    def test_decode_match_overload(self):
        info = extract_api_params('torch_npu.npu_hans_decode')
        matched, params, oidx = info.match_overload(4, ((), (), (), ()), False)
        assert matched
        assert oidx == 0

    def test_pickle_roundtrip(self):
        import pickle
        for api_name in ('torch_npu.npu_hans_decode', 'torch_npu.npu_hans_encode'):
            info = extract_api_params(api_name)
            data = pickle.dumps(info)
            restored = pickle.loads(data)
            assert restored.overloads[0].return_count == info.overloads[0].return_count
            assert restored.overloads[0].layout.is_out_tensor_list == info.overloads[0].layout.is_out_tensor_list
            assert restored.overloads[0].layout.out_expected_count == info.overloads[0].layout.out_expected_count
            assert restored.overloads[0].layout.is_out_required == info.overloads[0].layout.is_out_required
