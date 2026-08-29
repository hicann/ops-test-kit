# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for server-side _resolve_3party_api (KERNEL dual-form port).

torch/tf are available on the XPU server. Skip when torch_npu + tensorflow
coexist: torch_npu._C and tensorflow C extensions conflict on import → segfault.
"""

import importlib.util

import pytest

_has_torch_npu = importlib.util.find_spec("torch_npu") is not None
_has_tf = importlib.util.find_spec("tensorflow") is not None
if _has_torch_npu and _has_tf:
    pytestmark = pytest.mark.skip(reason="torch_npu._C + tensorflow C extension conflict → segfault")


def test_torch_resolve_snake():
    """torch snake op_name 命中（torch.add）。"""
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("add", "Add", "torch")
    assert callable(f)
    assert name == "add"


def test_tf_resolve_camel():
    """tf resolves relu（op_name first）。"""
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("relu", "Relu", "tf")
    assert callable(f)


def test_aclnn_resolve_strips_prefix_and_finds_torch():
    """aclnnAdd → strip aclnn → add → torch.add。"""
    from ttk.remote.server import executor

    f, name = executor._resolve_3party_api("Add", "aclnnAdd", "aclnn")
    assert callable(f)
