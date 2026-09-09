# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for discovering the CANN root from Ascend environment paths."""

from pathlib import Path

import pytest

from ttk._env import _find_ascend_root


@pytest.fixture(autouse=True)
def isolate_ascend_root_environment(monkeypatch):
    for env_var in ("ASCEND_CUSTOM_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        monkeypatch.delenv(env_var, raising=False)


def _make_ascend_root(root: Path) -> Path:
    (root / "compiler").mkdir(parents=True)
    (root / "opp").mkdir()
    return root


@pytest.mark.parametrize("root_name", ["foo", "pop"])
def test_find_ascend_root_preserves_non_opp_directory_name(monkeypatch, tmp_path, root_name):
    ascend_root = _make_ascend_root(tmp_path / root_name)
    monkeypatch.setenv("ASCEND_OPP_PATH", str(ascend_root))

    assert _find_ascend_root() == str(ascend_root)


@pytest.mark.parametrize("trailing_separator", [False, True])
def test_find_ascend_root_uses_parent_of_opp_directory(monkeypatch, tmp_path, trailing_separator):
    ascend_root = _make_ascend_root(tmp_path / "ascend-toolkit" / "latest")
    opp_path = str(ascend_root / "opp")
    if trailing_separator:
        opp_path += "/"
    monkeypatch.setenv("ASCEND_OPP_PATH", opp_path)

    assert _find_ascend_root() == str(ascend_root)
