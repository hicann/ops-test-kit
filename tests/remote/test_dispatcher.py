# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for remote dispatcher: serialization, dispatch, backoff, round-trip."""
import io
import json
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(scope="module")
def xpu_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", "19092", "--dry-run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(20):
        time.sleep(0.5)
        try:
            import http.client
            conn = http.client.HTTPConnection("127.0.0.1", 19092, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            conn.getresponse().read()
            conn.close()
            break
        except (ConnectionRefusedError, OSError):
            continue
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestDispatchApiMode:
    def test_dispatch_api_string(self, xpu_server):
        from ttk.remote.dispatcher import dispatch_to_remote
        inputs = [np.random.randn(4, 8).astype(np.float32)]
        outputs = dispatch_to_remote(
            op_name="softmax_v2",
            inputs=inputs,
            provider="torch",
            attrs={"axis": -1},
            endpoint_host="127.0.0.1",
            endpoint_port=19092,
            tenant_id="disp_test_001",
        )
        assert isinstance(outputs, list)
        assert len(outputs) >= 1
        assert isinstance(outputs[0], np.ndarray)


class TestSerializationRoundtrip:
    def test_numpy_roundtrip(self):
        from ttk.remote.dispatcher import _load_npz_outputs, _serialize_to_file
        original = [
            np.random.randn(4, 8).astype(np.float32),
            np.random.randn(3, 5).astype(np.int64),
        ]
        tmp = _serialize_to_file(original)
        try:
            schema = [{"index": 0, "dtype": "float32"}, {"index": 1, "dtype": "int64"}]
            outputs = _load_npz_outputs(tmp, schema)
            assert len(outputs) == 2
            np.testing.assert_array_equal(outputs[0], original[0])
            np.testing.assert_array_equal(outputs[1], original[1])
        finally:
            import os
            os.unlink(tmp)


class TestInputSchema:
    def test_build_schema_tensor_list(self):
        # G2 守卫：list slot 发 indices + 真实 dtype（卡 index/dtype 错配）
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema
        a, b, c = np.array([1.0]), np.array([2.0]), np.array([3.0])
        schema = _build_input_schema(inputs=[[a, b], c], input_names=["x", "y"])
        assert schema == [
            {"name": "x", "indices": [0, 1], "dtype": "float64"},
            {"name": "y", "index": 2, "dtype": "float64"},
        ]


class TestSerialize:
    def test_serialize_flattens_nested_and_skips_none(self):
        # G1：npz 叶子数守卫（卡 merge corruption —— 原 bug 把 [a,b] 堆成 (2,1)）
        import os

        import numpy as np

        from ttk.remote.dispatcher import _serialize_to_file
        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        tmp = _serialize_to_file([[a, b], c, [d, e], None])
        try:
            npz = np.load(tmp)
            assert len(npz.files) == 5                       # a0..a4,不是 3
            for i in range(5):
                assert npz[f"a{i}"].shape == (1,)           # 每个 (1,),不是 (2,1)
        finally:
            os.unlink(tmp)


class TestSchemaLeafCount:
    def test_count_equals_npz_and_schema(self):
        # G5：两路不变量（§4.4）—— _schema_leaf_count(schema) == npz 叶子数
        import os

        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema, _schema_leaf_count, _serialize_to_file
        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        inputs = [[a, b], c, [d, e], None]
        names = ["p0", "p1", "p2", "p3"]
        schema = _build_input_schema(inputs, names)
        npz_path = _serialize_to_file(inputs)
        try:
            npz_n = len(np.load(npz_path).files)
            assert _schema_leaf_count(schema) == npz_n == 5
        finally:
            os.unlink(npz_path)


class TestErrorHandling:
    def test_connection_refused(self):
        from ttk.remote.dispatcher import RemoteConnectionError, dispatch_to_remote
        with pytest.raises(RemoteConnectionError):
            dispatch_to_remote(
                op_name="test",
                inputs=[np.array([1.0])],
                endpoint_host="127.0.0.1",
                endpoint_port=19999,  # nobody listening
                tenant_id="err_test",
                timeout=2,
            )


class TestSyncSemaphore:
    def test_acquire_true_syncs_and_sets_ok(self, tmp_path):
        from ttk.remote import dispatcher

        class FakeCtx:
            def __init__(self):
                self.set_calls = []

            def acquire_semaphore(self, name):
                return True

            def set_semaphore(self, name, value):
                self.set_calls.append((name, value))

            def get_semaphore(self, name):
                raise AssertionError("should not poll when acquired")

        fake = FakeCtx()
        with patch("ttk.remote.dispatcher._do_http_sync", return_value=True) as do_sync, \
             patch("ttk.core_modules.tbe_multiprocessing.pool.get_process_context",
                   return_value=fake):
            ok = dispatcher._sync_missing_dependency(
                "m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is True
        assert do_sync.called
        assert fake.set_calls == [("xpu_sync_127.0.0.1:19090:m", "ok")]


class TestClientHelpers:
    def test_backoff_delay_caps_and_jitters(self):
        from ttk.remote.dispatcher import _backoff_delay
        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: b) == 0.5 * 1.25
        assert _backoff_delay(5, 0.5, 10.0, 0.25, lambda a, b: b) == 10.0 * 1.25
        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: a) == 0.5 * 0.75


class _FakeResp:
    def __init__(self, status, headers=None, body=b""):
        self.status = status
        self._h = headers or {}
        self._body = body

    def getheader(self, name, default=None):
        return self._h.get(name, default)

    def read(self):
        return self._body


class _ScriptedConn:
    script = []
    sent_modes = []

    def __init__(self, host, port, timeout=None):
        self.timeout = timeout

    def putrequest(self, *a, **k):
        pass

    def putheader(self, k, v):
        if k == "X-Mode":
            _ScriptedConn.sent_modes.append(v)

    def endheaders(self):
        pass

    def send(self, chunk):
        pass

    def getresponse(self):
        return _ScriptedConn.script.pop(0) if _ScriptedConn.script else _FakeResp(500)

    def close(self):
        pass


@pytest.fixture
def scripted(monkeypatch):
    _ScriptedConn.script = []
    _ScriptedConn.sent_modes = []
    monkeypatch.setattr("ttk.remote.dispatcher.http.client.HTTPConnection", _ScriptedConn)
    monkeypatch.setattr("ttk.remote.dispatcher.time.sleep", lambda *a, **k: None)
    yield _ScriptedConn


def _npz_body(*arrs):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(arrs)})
    return buf.getvalue()


class TestDispatchBackoff:
    def test_503_then_200(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote
        scripted.script = [_FakeResp(503), _FakeResp(503),
                           _FakeResp(200, {"X-Output-Count": "1",
                                           "X-Output-Schema": json.dumps([{"index": 0, "dtype": "float64"}])},
                                     _npz_body(np.array([2.0])))]
        out = dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                                 mode=DATA, endpoint_port=9, tenant_id="t")
        assert len(out) == 1

    def test_500_no_retry(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import RemoteExecutionError, dispatch_to_remote
        scripted.script = [_FakeResp(500, body=b"boom")]
        with pytest.raises(RemoteExecutionError):
            dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                               mode=DATA, endpoint_port=9, tenant_id="t")
        assert len(scripted.script) == 0      # only one attempt; no retry


class TestRoundTrip:
    def test_nested_round_trips_through_match_params_v1(self):
        # G3：端到端 schema→serialize→server match_params_v1 结构对等
        # 未修代码上必红（p0 是 merged (2,1) 数组,不是 list）
        import numpy as np

        from ttk.remote.dispatcher import _build_input_schema, _serialize_to_file
        from ttk.remote.server.execution_container import match_params_v1
        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        inputs = [[a, b], c, [d, e], None]
        names = ["p0", "p1", "p2", "p3"]
        schema = _build_input_schema(inputs, names)
        npz_path = _serialize_to_file(inputs)
        try:
            npz = np.load(npz_path)
            flat = [npz[k] for k in npz.files]
            named = match_params_v1(schema, flat)            # 真实 server 函数
            # p0/p2 是 list（不是 merged 数组）,p3 是 None
            assert isinstance(named["p0"], list) and len(named["p0"]) == 2
            np.testing.assert_array_equal(named["p0"][0], a)
            np.testing.assert_array_equal(named["p0"][1], b)
            assert isinstance(named["p2"], list) and len(named["p2"]) == 2
            assert named["p3"] is None
        finally:
            import os
            os.unlink(npz_path)
