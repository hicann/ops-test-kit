# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--core-limit 传递链测试：解析格式、kernel 侧按 core_type 生效分量、启动期物理上限校验、aclgraph 下发。"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("8", (8, None)),
        ("8,48", (8, 48)),
        (",48", (None, 48)),
    ],
)
def test_parse_core_limit_formats(raw, expected):
    from ttk.cli.bridge import _parse_core_limit

    assert _parse_core_limit(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "8|48"])
def test_parse_core_limit_rejects_invalid(raw):
    from ttk.cli.bridge import _parse_core_limit

    with pytest.raises(ValueError, match="--core-limit"):
        _parse_core_limit(raw)


def _make_opc_with_core_limit(core_type, core_limit):
    from ttk.core_modules.operator.tbe_interface import Opc
    from ttk.utilities.classes import SWITCHES
    from ttk.utilities.container_utils import set_global_storage

    sw = SWITCHES()
    sw.core_limit = core_limit
    set_global_storage(sw)
    opc = object.__new__(Opc)
    opc._core_type = core_type
    calls = []
    opc._all_opc_invoke = lambda func, *a, **k: calls.append((func, a))
    return opc, calls


@pytest.mark.parametrize(
    ("core_type", "core_limit", "expected_res"),
    [
        ("AiCore", (8, 48), {"ai_core_cnt": "8"}),
        ("VectorCore", (8, 48), {"vector_core_cnt": "48"}),
        ("MIX", (None, 48), {"vector_core_cnt": "48"}),
        ("AiCore", (None, 48), None),
    ],
)
def test_apply_core_limit_per_core_type(core_type, core_limit, expected_res):
    opc, calls = _make_opc_with_core_limit(core_type, core_limit)
    opc._apply_core_limit()
    if expected_res is None:
        assert calls == []
    else:
        assert calls == [("set_platform_info_res", (0, expected_res))]


_HW = {"full_soc_version": "Ascend910B3", "ai_core_cnt": 20, "vector_core_cnt": 40}


def test_validate_core_limit_bounds():
    from ttk.utilities.platform import validate_core_limit

    validate_core_limit((20, 40), _HW)
    validate_core_limit((8, None), _HW)
    with pytest.raises(ValueError, match="ai_core_cnt"):
        validate_core_limit((21, 32), _HW)


def test_e2e_instance_validates_core_limit(monkeypatch):
    """e2e get_device_platform 在启动期校验 core_limit；CPU 后端跳过。"""
    from ttk.core_modules.framework_api import instance as fai
    from ttk.utilities.classes import SWITCHES
    from ttk.utilities.container_utils import set_global_storage

    sw = SWITCHES()
    sw.dev_plat = "AUTO"
    sw.core_limit = (8, 99)
    set_global_storage(sw)

    backend = SimpleNamespace(
        device_name=lambda: "Ascend910B3",
        soc_series=lambda: "Ascend910B",
        is_npu=lambda: True,
    )
    monkeypatch.setattr(fai, "get_backend", lambda *a, **k: backend)
    monkeypatch.setattr(
        "ttk.utilities.platform.get_npu_hw_info",
        lambda plat: dict(_HW),
    )
    inst = fai.FrameworkApiInstance.__new__(fai.FrameworkApiInstance)
    inst.backend = backend
    with pytest.raises(ValueError, match="vector_core_cnt"):
        inst.get_device_platform()

    inst.backend = SimpleNamespace(device_name=lambda: "cpu", soc_series=lambda: "cpu", is_npu=lambda: False)
    inst.get_device_platform()  # CPU: 不校验不抛错


def test_aclgraph_core_limit_scopes_preprocess_and_captured_model(monkeypatch):
    from ttk.core_modules.framework_api import graph_execution

    events = []

    class FakeScope:
        @staticmethod
        @contextmanager
        def limit_core_num(ai_limit, vec_limit):
            events.append(("enter", ai_limit, vec_limit))
            yield
            events.append(("exit", ai_limit, vec_limit))

    monkeypatch.setattr(graph_execution, "_get_aclgraph_scope", lambda: FakeScope)
    monkeypatch.setattr(
        graph_execution.torch.npu,
        "get_device_properties",
        lambda _dev_id: SimpleNamespace(cube_core_num=20, vector_core_num=40),
    )

    with graph_execution._aclgraph_core_limit_context((1, None), 0):
        events.append(("preprocess",))

    model = graph_execution._AclGraphCoreLimitModel(torch.nn.Identity(), (1, 2), 0)
    assert torch.equal(model(torch.tensor([3])), torch.tensor([3]))
    assert events == [
        ("enter", 1, 40),
        ("preprocess",),
        ("exit", 1, 40),
        ("enter", 1, 2),
        ("exit", 1, 2),
    ]
