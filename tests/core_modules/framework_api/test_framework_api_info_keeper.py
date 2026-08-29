#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""
Tests for ttk.core_modules.framework_api.framework_api_info_keeper:
cache, register, get.
"""

import pytest

from ttk.core_modules.framework_api.framework_api_info_keeper import FrameworkApiInfoKeeper
from ttk.utilities.simple_param_extractor import (
    _MANUAL_OVERRIDES,
    APIParamInfo,
    ParamInfo,
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
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")
        keeper = _keeper()
        info = keeper.get("torch.add")
        assert info is not None
        assert info.tensor_count >= 2
        cached = keeper.get("torch.add")
        assert cached is info
