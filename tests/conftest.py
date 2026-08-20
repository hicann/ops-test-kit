# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Shared pytest fixtures for the test suite."""
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_ttk_environment(monkeypatch):
    """Ensure tests do not accidentally pick up real device or config."""
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_TOOLKIT_HOME", raising=False)
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    monkeypatch.delenv("ASCEND_ROOT", raising=False)
    # 用户级 ~/.config/ttk/ttk.conf.yaml 会被 loader 读入(真实机器上常配着远端
    # endpoints),使断言"未配 endpoints"的测试随环境红。只保留 cwd 相对路径,
    # 测试用 tmp_path + chdir 自己铺配置。
    import ttk.config.loader as _loader
    monkeypatch.setattr(_loader, "_CONFIG_PATHS", [Path("ttk.conf.yaml")])


@pytest.fixture
def make_testcase():
    """Factory: TestcaseAclnn instances (shared global; test_testcase_e2e.py
    overrides locally with TestcaseE2e)."""
    def _make(api_name="aclnnDummy", **kwargs):
        from ttk.core_modules.testcase_manager.testcase_aclnn import TestcaseAclnn
        case = TestcaseAclnn()
        case.api_name = api_name
        case.is_valid = True
        case.fail_reason = None
        case.attributes = kwargs.pop("attributes", {})
        for k, v in kwargs.items():
            setattr(case, k, v)
        return case
    return _make


@pytest.fixture(scope="session", autouse=True)
def _load_default_config():
    """所有测试前加载默认 config 一次（替代被删的 get_config lazy fallback）。

    删 lazy fallback 后 get_config() 不再自动加载；这个 session fixture 保证
    _config 已加载默认配置链（default.yaml + ~/.config/ttk + ./ttk.conf.yaml），
    测试可直接 get_config()。需要自定义 config 的测试在测试体内显式
    load_config(yaml)，测完 load_config() 恢复默认。
    """
    from ttk.config.loader import load_config
    load_config()
    yield
