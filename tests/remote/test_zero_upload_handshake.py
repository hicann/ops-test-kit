# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""免传输(zero-upload)的能力握手。

回归的是一次真实事故：客户端单方面启用免传输(不发 body、只发配方头), 打到
未升级的服务端 —— 它不认这两个头, 看到 Content-Length: 0 却有 input_count>0,
于是绑不上第一个参数回 400。整整 400+ 例三方腿因此全空, 而判据带 close 的档
照报 PASS, 事后数 device_us 才发现。

故：服务端没声明 zero_upload 能力时, 客户端必须回落成整包上传。
"""

import numpy as np
import pytest

from ttk.remote import PERF, dispatcher


class _FakeResponse:
    status = 200

    def __init__(self):
        self.length = 0

    def getheader(self, name, default=None):
        return {"X-Perf": "", "X-API": "FakeSpec"}.get(name, default)

    def read(self, *a):
        return b""


class _FakeConn:
    """记录请求头; 不带 body 的 PERF 请求走 200 早退路径。"""

    def __init__(self, captured):
        self.captured = captured
        self.sent_bytes = 0

    def putrequest(self, *a):
        pass

    def putheader(self, key, value):
        self.captured[key] = value

    def endheaders(self):
        pass

    def send(self, chunk):
        self.sent_bytes += len(chunk)

    def getresponse(self):
        return _FakeResponse()

    def close(self):
        pass


@pytest.fixture
def captured(monkeypatch):
    cap = {}
    conn = _FakeConn(cap)
    monkeypatch.setattr(dispatcher, "_acquire_connection", lambda h, p, t: (conn, False))
    monkeypatch.setattr(dispatcher, "_settle_connection", lambda *a, **k: None)
    cap["_conn"] = conn
    return cap


def _recipe_for(arr):
    """配方表按叶子的内容指纹索引 —— 键必须是真指纹, 不能手编。"""
    from ttk.remote.input_recipe import digest_of

    return {digest_of(arr): {"kind": "zeros", "shape": list(arr.shape), "dtype": str(arr.dtype)}}


def _dispatch(with_recipes=True):
    x = np.zeros(8, dtype=np.float32)
    recipes = _recipe_for(x) if with_recipes else None
    return dispatcher.dispatch_to_remote(
        op_name="FakeOp",
        op_type="FakeOp",
        inputs=[x],
        input_names=["pred"],
        provider="torch",
        attrs={},
        input_recipes=recipes,
        endpoint_host="127.0.0.1",
        endpoint_port=9099,
        tenant_id="t",
        mode=PERF,
        return_result=True,
    )


def test_falls_back_to_full_upload_when_server_lacks_feature(monkeypatch, captured):
    monkeypatch.setattr(dispatcher, "_server_features", lambda h, p, t: [])  # 旧服务端
    _dispatch()
    assert "X-Input-Recipes" not in captured, "旧服务端不认配方头, 必须整包上传"
    assert captured["_conn"].sent_bytes > 0, "回落后必须真的把输入发出去"
    assert int(captured["Content-Length"]) > 0


def test_uses_zero_upload_when_server_advertises_feature(monkeypatch, captured):
    monkeypatch.setattr(dispatcher, "_server_features", lambda h, p, t: ["zero_upload"])
    _dispatch()
    assert "X-Input-Recipes" in captured, "服务端声明支持时应免传输"
    assert captured["_conn"].sent_bytes == 0
    assert captured["Content-Length"] == "0"


def test_probe_failure_is_treated_as_unsupported(monkeypatch, captured):
    # 探活本身失败(连不上/超时)时, 绝不能"乐观假设支持"
    monkeypatch.setattr(dispatcher, "_create_connection", _boom)
    dispatcher._FEATURE_CACHE.clear()
    _dispatch()
    assert "X-Input-Recipes" not in captured


def _boom(*a, **k):
    raise OSError("connection refused")
