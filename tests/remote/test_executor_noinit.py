# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""_invoke class 分支契约: __init__/__call__ 参数(除kwargs)并集 ⊆ inputs∪attrs;
input/attr 喂给声明它的方法(都声明则都喂)。device 保留注入, 有默认值用默认。
"""

import importlib.util

import pytest

if importlib.util.find_spec("torch_npu") is not None and importlib.util.find_spec("tensorflow") is not None:
    pytestmark = pytest.mark.skip(reason="torch_npu._C + tensorflow C extension conflict → segfault")
else:
    pytest.importorskip("torch")
    from ttk.remote.server import executor  # noqa: E402


def test_invoke_no_init_class_with_attrs():
    """无自定义 __init__ 的类 + 非空 attrs 不应 TypeError(object.__init__ 不收任意 kwarg)。"""

    class NoInit:
        def __call__(self, x):
            return [x]

    out = executor._invoke(
        NoInit,
        named={"x": 1},
        attrs={"axis": 9, "eps": 1e-5},
        provider="torch",
        device_id="cpu",
        use_device=False,
    )
    assert out == [1]
