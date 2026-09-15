# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""--super-kernel 传递链测试：旗标接线、启动期校验、aclgraph options 透传、GE 模式 scope 包装。"""

from types import SimpleNamespace

import pytest
import torch


def _args(**kwargs):
    base = {
        "dynamic": None,
        "const": None,
        "cpu": False,
        "fullgraph": 0,
        "aclgraph": False,
        "super_kernel": False,
        "core_limit": None,
    }
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_super_kernel_flag_wires_switch():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.utilities.classes import SWITCHES

    sw = SWITCHES()
    apply_e2e_args(sw, _args(const=True, super_kernel=True))
    assert sw.super_kernel_enabled is True
    assert sw.cst_switches.enabled is True


def test_super_kernel_requires_graph_mode():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.utilities.classes import SWITCHES

    sw = SWITCHES()
    sw.dyn_switches.enabled = False
    with pytest.raises(ValueError, match="graph mode"):
        apply_e2e_args(sw, _args(super_kernel=True))


def test_super_kernel_rejects_cpu():
    from ttk.cli.bridge import apply_e2e_args
    from ttk.utilities.classes import SWITCHES

    sw = SWITCHES()
    sw.dyn_switches.enabled = False
    with pytest.raises(ValueError, match="--cpu"):
        apply_e2e_args(sw, _args(const=True, cpu=True, super_kernel=True))


def test_super_kernel_scope_model_wraps_forward(monkeypatch):
    import sys

    from ttk.core_modules.framework_api.graph_execution import _SuperKernelScopeModel

    entered = []

    class _scope_ctx:
        def __init__(self, name, options):
            entered.append((name, options))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake_torchair = SimpleNamespace(scope=SimpleNamespace(super_kernel=lambda name, options: _scope_ctx(name, options)))
    monkeypatch.setitem(sys.modules, "torchair", fake_torchair)

    calls = []

    class _Inner(torch.nn.Module):
        def forward(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "ok"

    wrapper = _SuperKernelScopeModel(_Inner(), "Add")
    assert wrapper(1, x=2) == "ok"
    assert calls == [((1,), {"x": 2})]
    assert entered == [("Add", "")]


def test_compile_model_aclgraph_passes_super_kernel_options(monkeypatch):
    import torch

    from ttk.core_modules.framework_api import graph_execution

    captured = {}

    def fake_compile(model, *, fullgraph, backend, dynamic, **kwargs):
        captured.update(model=model, backend=backend, fullgraph=fullgraph, dynamic=dynamic, kwargs=kwargs)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)

    m = object()
    graph_execution._compile_model_aclgraph(m, "npugraph_ex", 1, True)
    assert captured["kwargs"]["options"] == graph_execution._ACLGRAPH_SUPER_KERNEL_OPTIONS
    assert captured["kwargs"]["options"]["super_kernel_optimize"] is True
    assert captured["kwargs"]["options"]["super_kernel_debug_options"] == {"debug_per_op_max_core_num": 1}

    graph_execution._compile_model_aclgraph(m, "npugraph_ex", 0, False)
    assert "options" not in captured["kwargs"]
