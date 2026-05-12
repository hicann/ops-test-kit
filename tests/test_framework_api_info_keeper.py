#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""
Tests for ttk.core_modules.framework_api.framework_api_info_keeper:
cache, validate_testcase_params, get_tensor_distribution, register.
"""

import pytest
from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper
from ttk.utilities.simple_param_extractor import (
    APIParamInfo, ParamInfo, _MANUAL_OVERRIDES,
)


def _keeper():
    return FrameworkApiInfoKeeper()


class TestFrameworkApiInfoKeeperCache:

    def setup_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_clear_cache(self):
        keeper = _keeper()
        keeper._cache["test_key"] = None
        keeper.clear_cache()
        assert len(keeper._cache) == 0

    def test_get_unknown_returns_none(self):
        result = _keeper().get("nonexistent.module.func")
        assert result is None

    def test_get_caches_result(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[ParamInfo(name="input", type="Tensor")],
            source="test",
        )
        keeper._cache["test.api"] = info
        result = keeper.get("test.api")
        assert result is info

    def test_register_with_api_param_info(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.registered",
            params=[ParamInfo(name="x", type="Tensor")],
            source="manual",
        )
        keeper.register("test.registered", info)
        assert keeper.get("test.registered") is info

    def test_register_with_param_list(self):
        keeper = _keeper()
        params = [ParamInfo(name="x", type="Tensor"), ParamInfo(name="dim", type="int")]
        keeper.register("test.registered2", params, source="config")
        result = keeper.get("test.registered2")
        assert result is not None
        assert result.tensor_count == 1
        assert result.source == "config"

    def test_get_torch_api_if_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        keeper = _keeper()
        info = keeper.get("torch.add")
        assert info is not None
        assert info.tensor_count >= 2
        cached = keeper.get("torch.add")
        assert cached is info


class TestValidateTestcaseParams:

    def setup_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_mismatch_returns_error_message(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[ParamInfo(name="input", type="Tensor")],
            source="test",
        )
        keeper._cache["test.api"] = info
        error = keeper.validate_testcase_params("test.api", tensor_count=3)
        assert error is not None
        assert "3" in error
        assert "1" in error

    def test_match_returns_none(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        keeper._cache["test.api"] = info
        error = keeper.validate_testcase_params("test.api", tensor_count=2)
        assert error is None

    def test_unknown_api_returns_none(self):
        error = _keeper().validate_testcase_params("unknown.api", tensor_count=1)
        assert error is None


class TestGetTensorDistribution:

    def setup_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def teardown_method(self):
        _keeper().clear_cache()
        _MANUAL_OVERRIDES.clear()

    def test_flat_tensors(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[
                ParamInfo(name="input", type="Tensor"),
                ParamInfo(name="other", type="Tensor"),
            ],
            source="test",
        )
        keeper._cache["test.api"] = info
        assert keeper.get_tensor_distribution("test.api") == (0, 0)

    def test_with_tensorlist(self):
        keeper = _keeper()
        info = APIParamInfo(
            api_name="test.api",
            params=[
                ParamInfo(name="tensors", type="tuple of Tensors"),
                ParamInfo(name="out", type="Tensor"),
            ],
            source="test",
        )
        keeper._cache["test.api"] = info
        assert keeper.get_tensor_distribution("test.api") == (-1, 0)

    def test_unknown_api_returns_empty(self):
        assert _keeper().get_tensor_distribution("unknown.api") == ()

    def test_torch_cat_if_available(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        dist = _keeper().get_tensor_distribution("torch.cat")
        assert -1 in dist, f"torch.cat should have TensorList in distribution, got {dist}"

    def test_singleton_returns_same_instance(self):
        assert _keeper() is _keeper()
