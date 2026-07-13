import io
import json
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

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
        from ttk.remote.dispatcher import _serialize_to_file, _load_npz_outputs
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

    def test_single_tensor_roundtrip(self):
        from ttk.remote.dispatcher import _serialize_to_file, _load_npz_outputs
        original = [np.array([1.0, 2.0, 3.0])]
        tmp = _serialize_to_file(original)
        try:
            schema = [{"index": 0, "dtype": "float64"}]
            outputs = _load_npz_outputs(tmp, schema)
            np.testing.assert_array_equal(outputs[0], original[0])
        finally:
            import os
            os.unlink(tmp)


class TestPerfMode:
    def test_perf_mode_no_output(self, xpu_server):
        from ttk.remote.dispatcher import dispatch_to_remote
        inputs = [np.random.randn(4, 8).astype(np.float32)]
        outputs = dispatch_to_remote(
            op_name="softmax_v2",
            inputs=inputs,
            provider="torch",
            attrs={},
            endpoint_host="127.0.0.1",
            endpoint_port=19092,
            tenant_id="disp_perf_001",
            mode="perf",
        )
        assert outputs == []


class TestInputSchema:
    def test_build_schema_single_tensors(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        schema = _build_input_schema(
            inputs=[np.array([1.0]), np.array([2.0]), np.array([3.0])],
            input_names=["x", "y", "z"],
        )
        assert schema == [
            {"name": "x", "index": 0, "dtype": "float64"},
            {"name": "y", "index": 1, "dtype": "float64"},
            {"name": "z", "index": 2, "dtype": "float64"},
        ]

    def test_build_schema_top_level_none(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        schema = _build_input_schema(
            inputs=[np.array([1.0]), None, np.array([3.0])],
            input_names=["x", "y", "z"],
        )
        assert schema == [
            {"name": "x", "index": 0, "dtype": "float64"},
            {"name": "y", "index": None, "dtype": None},
            {"name": "z", "index": 1, "dtype": "float64"},
        ]

    def test_build_schema_tensor_list(self):
        # G2 守卫：list slot 发 indices + 真实 dtype（卡 index/dtype 错配）
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        a, b, c = np.array([1.0]), np.array([2.0]), np.array([3.0])
        schema = _build_input_schema(inputs=[[a, b], c], input_names=["x", "y"])
        assert schema == [
            {"name": "x", "indices": [0, 1], "dtype": "float64"},
            {"name": "y", "index": 2, "dtype": "float64"},
        ]

    def test_build_schema_multi_tensor_list_with_none(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        schema = _build_input_schema(
            inputs=[[a, b], c, [d, e], None],
            input_names=["p0", "p1", "p2", "p3"],
        )
        assert schema == [
            {"name": "p0", "indices": [0, 1], "dtype": "float64"},
            {"name": "p1", "index": 2, "dtype": "float64"},
            {"name": "p2", "indices": [3, 4], "dtype": "float64"},
            {"name": "p3", "index": None, "dtype": None},
        ]

    def test_build_schema_none_inside_list_slot(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        a, b = np.array([1.0]), np.array([2.0])
        schema = _build_input_schema(
            inputs=[a, None, [b, None]], input_names=["x", "y", "z"],
        )
        assert schema == [
            {"name": "x", "index": 0, "dtype": "float64"},
            {"name": "y", "index": None, "dtype": None},
            {"name": "z", "indices": [1], "dtype": "float64"},
        ]

    def test_build_schema_empty_list_slot(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        schema = _build_input_schema(inputs=[[], np.array([1.0])],
                                     input_names=["x", "y"])
        assert schema == [
            {"name": "x", "indices": [], "dtype": None},
            {"name": "y", "index": 0, "dtype": "float64"},
        ]

    def test_build_schema_names_more_than_inputs(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        schema = _build_input_schema(inputs=[np.array([1.0])],
                                     input_names=["x", "y"])
        assert schema == [
            {"name": "x", "index": 0, "dtype": "float64"},
            {"name": "y", "index": None, "dtype": None},
        ]

    def test_build_schema_inputs_more_than_names_asserts(self):
        from ttk.remote.dispatcher import _build_input_schema
        import numpy as np
        with pytest.raises(AssertionError):
            _build_input_schema(inputs=[np.array([1.0]), np.array([2.0])],
                                 input_names=["x"])

    def test_build_schema_empty_inputs(self):
        from ttk.remote.dispatcher import _build_input_schema
        assert _build_input_schema([], []) == []

    def test_build_schema_bfloat16_single(self):
        import numpy as _np
        try:
            import ml_dtypes
        except ImportError:
            pytest.skip("ml_dtypes not installed")
        arr = _np.array([1.5, 2.5], dtype=ml_dtypes.bfloat16)
        from ttk.remote.dispatcher import _build_input_schema
        schema = _build_input_schema(inputs=[arr], input_names=["x"])
        assert schema[0]["dtype"] == "bfloat16"

    def test_build_schema_bfloat16_list(self):
        # bf16 list-slot：卡 dtype=None corruption（原 bug 下 list slot dtype 丢失）
        import numpy as _np
        try:
            import ml_dtypes
        except ImportError:
            pytest.skip("ml_dtypes not installed")
        arr = _np.array([1.5, 2.5], dtype=ml_dtypes.bfloat16)
        from ttk.remote.dispatcher import _build_input_schema
        schema = _build_input_schema(inputs=[[arr, arr]], input_names=["x"])
        assert schema == [{"name": "x", "indices": [0, 1], "dtype": "bfloat16"}]

    def test_build_schema_2d_ndarray_not_split(self):
        # 守卫：2D ndarray slot 必须是单叶子（index:0），不能被 deep_flatten
        # 按行切成两片。卡 "deep_flatten everything" 回归（回退后会得到
        # indices=[0,1] 或 dtype 跟随首行）。
        from ttk.remote.dispatcher import _build_input_schema
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        schema = _build_input_schema(inputs=[arr], input_names=["x"])
        assert schema == [{"name": "x", "index": 0, "dtype": "float64"}]

    def test_build_schema_tuple_slot(self):
        # 守卫：tuple slot 与 list slot 等价走 indices 分支。
        # 卡 isinstance(slot,(list,tuple)) 漏掉 tuple 的回退。
        from ttk.remote.dispatcher import _build_input_schema
        schema = _build_input_schema(
            inputs=[np.array([1.0]), (np.array([2.0]), np.array([3.0]))],
            input_names=["x", "y"],
        )
        assert schema == [
            {"name": "x", "index": 0, "dtype": "float64"},
            {"name": "y", "indices": [1, 2], "dtype": "float64"},
        ]

    def test_serialize_skips_none(self):
        from ttk.remote.dispatcher import _serialize_to_file
        import os
        inputs = [np.array([1.0]), None, np.array([2.0, 3.0])]
        tmp = _serialize_to_file(inputs)
        try:
            npz = np.load(tmp)
            assert len(npz.files) == 2
            np.testing.assert_array_equal(npz["a0"], np.array([1.0]))
            np.testing.assert_array_equal(npz["a1"], np.array([2.0, 3.0]))
        finally:
            os.unlink(tmp)


class TestSerialize:
    def test_serialize_flattens_nested_and_skips_none(self):
        # G1：npz 叶子数守卫（卡 merge corruption —— 原 bug 把 [a,b] 堆成 (2,1)）
        from ttk.remote.dispatcher import _serialize_to_file
        import numpy as np, os
        a, b, c, d, e = [np.array([float(i)]) for i in range(1, 6)]
        tmp = _serialize_to_file([[a, b], c, [d, e], None])
        try:
            npz = np.load(tmp)
            assert len(npz.files) == 5                       # a0..a4,不是 3
            for i in range(5):
                assert npz[f"a{i}"].shape == (1,)           # 每个 (1,),不是 (2,1)
        finally:
            os.unlink(tmp)

    def test_serialize_keeps_dir_param(self):
        # 回归：dir= 参数必须保留（dispatch_to_remote L463 传 req_dir）
        from ttk.remote.dispatcher import _serialize_to_file
        import numpy as np, os, tempfile, shutil
        d = tempfile.mkdtemp()
        tmp = _serialize_to_file([np.array([1.0])], dir=d)
        try:
            assert os.path.dirname(tmp) == d
        finally:
            os.unlink(tmp)
            shutil.rmtree(d, ignore_errors=True)


class TestSchemaLeafCount:
    def test_count_equals_npz_and_schema(self):
        # G5：两路不变量（§4.4）—— _schema_leaf_count(schema) == npz 叶子数
        from ttk.remote.dispatcher import (_build_input_schema,
            _serialize_to_file, _schema_leaf_count)
        import numpy as np, os
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

    def test_count_none_inside_list_slot(self):
        from ttk.remote.dispatcher import _build_input_schema, _schema_leaf_count
        import numpy as np
        a, b = np.array([1.0]), np.array([2.0])
        schema = _build_input_schema([a, None, [b, None]], ["x", "y", "z"])
        assert _schema_leaf_count(schema) == 2               # list 内 None 不计

    def test_count_all_none(self):
        from ttk.remote.dispatcher import _build_input_schema, _schema_leaf_count
        schema = _build_input_schema([None, None], ["x", "y"])
        assert _schema_leaf_count(schema) == 0


class TestErrorHandling:
    def test_connection_refused(self):
        from ttk.remote.dispatcher import dispatch_to_remote, RemoteConnectionError
        with pytest.raises(RemoteConnectionError):
            dispatch_to_remote(
                op_name="test",
                inputs=[np.array([1.0])],
                endpoint_host="127.0.0.1",
                endpoint_port=19999,  # nobody listening
                tenant_id="err_test",
                timeout=2,
            )


class TestFindSpecFile:
    def test_finds_by_full_module_name(self, tmp_path):
        from ttk.remote.dispatcher import _find_spec_file

        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "util.py").write_text("# util")

        result = _find_spec_file("sub.util", [str(tmp_path)])
        assert result == str(tmp_path / "sub" / "util.py")

    def test_finds_by_shallow_name(self, tmp_path):
        from ttk.remote.dispatcher import _find_spec_file

        (tmp_path / "mylib.py").write_text("# mylib")

        result = _find_spec_file("mylib", [str(tmp_path)])
        assert result == str(tmp_path / "mylib.py")

    def test_returns_none_when_not_found(self, tmp_path):
        from ttk.remote.dispatcher import _find_spec_file

        result = _find_spec_file("nonexistent", [str(tmp_path)])
        assert result is None

    def test_searches_multiple_roots(self, tmp_path):
        from ttk.remote.dispatcher import _find_spec_file

        root2 = tmp_path / "other"
        root2.mkdir()
        (root2 / "found.py").write_text("# found")

        result = _find_spec_file("found", [str(tmp_path), str(root2)])
        assert result == str(root2 / "found.py")


class TestReadFileWithHash:
    def test_returns_content_and_hash(self, tmp_path):
        import base64
        import hashlib
        from ttk.remote.dispatcher import _read_file_with_hash
        content = b"def foo():\n    return 42\n"
        f = tmp_path / "mod.py"
        f.write_bytes(content)
        b64, sha = _read_file_with_hash(str(f))
        assert b64 == base64.b64encode(content).decode()
        assert sha == hashlib.sha256(content).hexdigest()

    def test_empty_file(self, tmp_path):
        import base64
        import hashlib
        from ttk.remote.dispatcher import _read_file_with_hash
        f = tmp_path / "empty.py"
        f.write_bytes(b"")
        b64, sha = _read_file_with_hash(str(f))
        assert b64 == ""
        assert sha == hashlib.sha256(b"").hexdigest()


class TestDoHttpSync:
    def test_sends_real_hash_in_body(self, tmp_path):
        import base64
        import hashlib
        import json
        from ttk.remote import dispatcher
        content = b"x = 1\n"
        (tmp_path / "missing_mod.py").write_bytes(content)
        expected_hash = hashlib.sha256(content).hexdigest()

        captured = {}

        class FakeResp:
            status = 200
            def read(self):
                return b"{}"

        class FakeConn:
            def __init__(self, *a, **k):
                pass
            def request(self, method, path, body=None, headers=None):
                captured["body"] = body
                captured["headers"] = headers
            def getresponse(self):
                return FakeResp()
            def close(self):
                pass

        with patch("ttk.remote.dispatcher.http.client.HTTPConnection", FakeConn):
            ok = dispatcher._do_http_sync(
                "missing_mod", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is True
        entry = json.loads(captured["body"])["files"]["missing_mod.py"]
        assert entry["hash"] == expected_hash
        assert entry["hash"] != ""

    def test_file_not_found_returns_false(self, tmp_path):
        from ttk.remote import dispatcher
        ok = dispatcher._do_http_sync(
            "nope", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is False

    def test_non_200_returns_false(self, tmp_path):
        from ttk.remote import dispatcher
        (tmp_path / "m.py").write_bytes(b"x=1\n")

        class FakeResp:
            status = 500
            def read(self):
                return b"{}"

        class FakeConn:
            def __init__(self, *a, **k):
                pass
            def request(self, *a, **k):
                pass
            def getresponse(self):
                return FakeResp()
            def close(self):
                pass

        with patch("ttk.remote.dispatcher.http.client.HTTPConnection", FakeConn):
            ok = dispatcher._do_http_sync(
                "m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is False


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

    def test_acquire_false_polls_and_reuses(self, tmp_path):
        from ttk.remote import dispatcher

        results = iter([None, None, "ok"])

        class FakeCtx:
            def acquire_semaphore(self, name):
                return False
            def set_semaphore(self, name, value):
                raise AssertionError("should not set when not acquired")
            def get_semaphore(self, name):
                return next(results)

        fake = FakeCtx()
        with patch("ttk.core_modules.tbe_multiprocessing.pool.get_process_context",
                   return_value=fake), \
             patch("ttk.remote.dispatcher.time.sleep"):
            ok = dispatcher._sync_missing_dependency(
                "m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is True

    def test_acquire_false_failure_propagates(self, tmp_path):
        from ttk.remote import dispatcher

        results = iter([None, "fail"])

        class FakeCtx:
            def acquire_semaphore(self, name):
                return False
            def set_semaphore(self, name, value):
                raise AssertionError
            def get_semaphore(self, name):
                return next(results)

        fake = FakeCtx()
        with patch("ttk.core_modules.tbe_multiprocessing.pool.get_process_context",
                   return_value=fake), \
             patch("ttk.remote.dispatcher.time.sleep"):
            ok = dispatcher._sync_missing_dependency(
                "m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert ok is False

    def test_acquire_true_exception_sets_err_and_reraises(self, tmp_path):
        from ttk.remote import dispatcher

        class FakeCtx:
            def __init__(self):
                self.set_calls = []
            def acquire_semaphore(self, name):
                return True
            def set_semaphore(self, name, value):
                self.set_calls.append((name, value))
            def get_semaphore(self, name):
                raise AssertionError

        fake = FakeCtx()
        with patch("ttk.remote.dispatcher._do_http_sync",
                   side_effect=RuntimeError("boom")), \
             patch("ttk.core_modules.tbe_multiprocessing.pool.get_process_context",
                   return_value=fake):
            with pytest.raises(RuntimeError):
                dispatcher._sync_missing_dependency(
                    "m", [str(tmp_path)], "127.0.0.1", 19090, "t1", 5)
        assert fake.set_calls[0][1].startswith("err:")


class TestClientHelpers:
    def test_remote_busy_is_remote_execution_error(self):
        from ttk.remote.dispatcher import RemoteBusyError, RemoteExecutionError
        assert issubclass(RemoteBusyError, RemoteExecutionError)

    def test_remote_result_defaults(self):
        from ttk.remote.dispatcher import RemoteResult
        r = RemoteResult(outputs=[1])
        assert r.outputs == [1] and r.perf is None

    def test_parse_client_mode(self):
        from ttk.remote.dispatcher import _parse_client_mode
        from ttk.remote import DATA, PERF
        assert _parse_client_mode(DATA) == DATA
        assert _parse_client_mode("data") == DATA
        assert _parse_client_mode("PERF") == PERF
        assert _parse_client_mode(None) == DATA

    def test_backoff_delay_caps_and_jitters(self):
        from ttk.remote.dispatcher import _backoff_delay
        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: b) == 0.5 * 1.25
        assert _backoff_delay(5, 0.5, 10.0, 0.25, lambda a, b: b) == 10.0 * 1.25
        assert _backoff_delay(0, 0.5, 10.0, 0.25, lambda a, b: a) == 0.5 * 0.75

    def test_env_defaults(self, monkeypatch):
        from ttk.remote.dispatcher import _cfg
        # Default config (session fixture) has no remote endpoints, so
        # get_remote_config() returns None and _cfg falls back to defaults.
        for k in ("TTK_XPU_BACKOFF_BASE_S", "TTK_XPU_BACKOFF_MAX_S",
                  "TTK_XPU_BACKOFF_JITTER", "TTK_XPU_503_MAX_RETRIES",
                  "TTK_XPU_CONN_MAX_RETRIES", "TTK_XPU_DISPATCH_DEADLINE_S"):
            monkeypatch.delenv(k, raising=False)
        assert _cfg('backoff_base_s', 0.5) == 0.5
        assert _cfg('backoff_max_s', 10.0) == 10.0
        assert _cfg('backoff_jitter', 0.25) == 0.25
        assert _cfg('max_503_retries', 10.0, int) == 10
        assert _cfg('max_conn_retries', 5.0, int) == 5
        assert _cfg('dispatch_deadline_s', 300.0, int) == 300


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
    def test_x_mode_sent_as_int(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote
        scripted.script = [_FakeResp(200, {"X-Output-Count": "1"},
                                     _npz_body(np.array([1.0])))]
        dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                           mode=DATA, endpoint_port=9, tenant_id="t")
        assert scripted.sent_modes == ["1"]

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

    def test_503_exhausted_raises_busy(self, scripted, monkeypatch, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote, RemoteBusyError
        from ttk.config import loader as loader

        # Set max_503_retries=2 via a yaml config (was set_remote_config before)
        (tmp_path / "ttk.conf.yaml").write_text(
            "remote:\n"
            "  endpoints:\n"
            "    - {host: '127.0.0.1', port: 9}\n"
            "  max_503_retries: 2\n"
        )
        loader._config = None
        loader.load_config(str(tmp_path / "ttk.conf.yaml"))

        scripted.script = [_FakeResp(503) for _ in range(10)]
        with pytest.raises(RemoteBusyError):
            dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                               mode=DATA, endpoint_port=9, tenant_id="t")

    def test_503_does_not_erode_424(self, scripted, monkeypatch, tmp_path):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote
        from ttk.config import loader as loader

        monkeypatch.setattr("ttk.remote.dispatcher._sync_missing_dependency",
                            lambda *a, **k: True)

        # Set max_503_retries=3 via a yaml config (test expects 3 retries to be allowed)
        (tmp_path / "ttk.conf.yaml").write_text(
            "remote:\n"
            "  endpoints:\n"
            "    - {host: '127.0.0.1', port: 9}\n"
            "  max_503_retries: 3\n"
        )
        loader._config = None
        loader.load_config(str(tmp_path / "ttk.conf.yaml"))

        scripted.script = [_FakeResp(503), _FakeResp(503), _FakeResp(503),
                           _FakeResp(424, body=b'{"missing": "m"}'),
                           _FakeResp(200, {"X-Output-Count": "1",
                                           "X-Output-Schema": json.dumps([{"index": 0, "dtype": "float64"}])},
                                     _npz_body(np.array([3.0])))]
        out = dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                                 mode=DATA, endpoint_port=9, tenant_id="t", max_retries=2)
        assert len(out) == 1

    def test_500_no_retry(self, scripted):
        from ttk.remote import DATA
        from ttk.remote.dispatcher import dispatch_to_remote, RemoteExecutionError
        scripted.script = [_FakeResp(500, body=b"boom")]
        with pytest.raises(RemoteExecutionError):
            dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                               mode=DATA, endpoint_port=9, tenant_id="t")
        assert len(scripted.script) == 0      # only one attempt; no retry

    def test_read_perf_with_return_result(self, scripted):
        from ttk.remote import DATA, PERF
        from ttk.remote.dispatcher import dispatch_to_remote, RemoteResult
        scripted.script = [_FakeResp(200, {"X-Output-Count": "1",
                                           "X-Output-Schema": json.dumps([{"index": 0, "dtype": "float64"}]),
                                           "X-Perf": '{"device_us": 2500.0}'},
                                     _npz_body(np.array([1.0])))]
        res = dispatch_to_remote(op_name="add", inputs=[np.array([1.0])], input_names=["x"],
                                 mode=DATA | PERF, endpoint_port=9, tenant_id="t",
                                 return_result=True)
        assert isinstance(res, RemoteResult)
        assert res.perf == {"device_us": 2500.0}
        assert len(res.outputs) == 1


class TestMissingModuleFastFail:
    """424 for a module not in spec_search_roots must fail immediately,
    not burn through the retry budget.

    Server-side ``ImportError`` on e.g. ``import scipy`` (an env dep) produces
    a 424 — but client has no scipy.py to upload, so retrying is pointless.
    """

    def test_missing_env_dep_fails_immediately(self, scripted, tmp_path):
        from ttk.remote.dispatcher import dispatch_to_remote, RemoteExecutionError

        scripted.script = [_FakeResp(424, body=b'{"missing": "scipy"}')]
        (tmp_path / "spec.py").write_text("# spec")  # not scipy

        with pytest.raises(RemoteExecutionError) as exc_info:
            dispatch_to_remote(
                op_name="add",
                inputs=[np.array([1.0])],
                input_names=["x"],
                endpoint_port=9,
                tenant_id="t",
                spec_search_roots=[str(tmp_path)],
                max_retries=5,
            )
        assert "environment" in str(exc_info.value).lower()
        assert "scipy" in str(exc_info.value)
        # Server only got ONE request (424), no retries were issued
        assert len(scripted.script) == 0 or scripted.script[0].status != 424

    def test_missing_spec_file_retries_if_sync_possible(self, scripted, tmp_path):
        """Regression: a module that IS in search_roots must still attempt sync."""
        from ttk.remote.dispatcher import dispatch_to_remote

        (tmp_path / "util.py").write_text("# util")
        scripted.script = [
            _FakeResp(424, body=b'{"missing": "util"}'),
            _FakeResp(200, {"X-Output-Count": "1",
                            "X-Output-Schema": json.dumps([{"index": 0, "dtype": "float64"}])},
                      _npz_body(np.array([1.0]))),
        ]
        with patch("ttk.remote.dispatcher._sync_missing_dependency", return_value=True):
            res = dispatch_to_remote(
                op_name="add",
                inputs=[np.array([1.0])],
                input_names=["x"],
                endpoint_port=9,
                tenant_id="t",
                spec_search_roots=[str(tmp_path)],
            )
        assert len(res) == 1


def test_dispatch_sends_x_op_name_header(monkeypatch):
    """op_name/op_type must be sent as X-Op-Name/X-Op-Type; api=None omits X-API."""
    captured = {}

    class FakeResp:
        status = 200

        def getheader(self, n, d=None):
            if n == "X-Output-Count":
                return "0"
            return d

        def read(self):
            return b""

    class FakeConn:
        def putrequest(self, method, path):
            captured["headers"] = {}

        def putheader(self, key, value):
            captured["headers"][key] = value

        def endheaders(self):
            pass

        def send(self, chunk):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setattr("ttk.remote.dispatcher._create_connection",
                        lambda *a, **k: FakeConn())
    from ttk.remote.dispatcher import dispatch_to_remote
    dispatch_to_remote(op_name="add", inputs=[], op_type="Add", provider="torch",
                       execution_type="api", api=None)  # no explicit api
    assert captured["headers"].get("X-Op-Name") == "add"
    assert captured["headers"].get("X-Op-Type") == "Add"
    # api was None -> no X-API (server derives)
    assert "X-API" not in captured["headers"] or captured["headers"].get("X-API") in (None, "")


def test_dispatch_sends_leaf_count_x_input_count(monkeypatch):
    """X-Input-Count 派生自 schema 叶子数，不是 slot 数。

    inputs=[[a,b], c] → 2 slot，但 3 个 npz 叶子（a,b,c）。
    守卫 line 407 `effective_count = _schema_leaf_count(schema)`：若回退到
    `sum(1 for x in inputs if x is not None)`，header 会变成 "2"，server 静默截断。
    """
    captured = {}

    class FakeResp:
        status = 200

        def getheader(self, n, d=None):
            if n == "X-Output-Count":
                return "0"
            return d

        def read(self):
            return b""

    class FakeConn:
        def putrequest(self, method, path):
            captured["headers"] = {}

        def putheader(self, key, value):
            captured["headers"][key] = value

        def endheaders(self):
            pass

        def send(self, chunk):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setattr("ttk.remote.dispatcher._create_connection",
                        lambda *a, **k: FakeConn())
    from ttk.remote.dispatcher import dispatch_to_remote
    a, b, c = np.array([1.0]), np.array([2.0]), np.array([3.0])
    dispatch_to_remote(
        op_name="add",
        inputs=[[a, b], c],
        input_names=["x", "y"],
        endpoint_port=9,
        tenant_id="t",
        mode="data",
    )
    # 3 leaves (a, b, c), NOT 2 slots —— slot-count 回退会让此断言红
    assert captured["headers"].get("X-Input-Count") == "3"


def test_dispatch_sends_x_runtime_header_default(monkeypatch):
    """X-Runtime header 默认 = str(3)（switches.run_time 默认 3）。"""
    captured = {}

    class FakeResp:
        status = 200

        def getheader(self, n, d=None):
            if n == "X-Output-Count":
                return "0"
            return d

        def read(self):
            return b""

    class FakeConn:
        def putrequest(self, method, path):
            captured["headers"] = {}

        def putheader(self, key, value):
            captured["headers"][key] = value

        def endheaders(self):
            pass

        def send(self, chunk):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setattr("ttk.remote.dispatcher._create_connection",
                        lambda *a, **k: FakeConn())
    from ttk.remote.dispatcher import dispatch_to_remote
    dispatch_to_remote(op_name="add", inputs=[], endpoint_port=9, tenant_id="t")
    assert captured["headers"].get("X-Runtime") == "3"


def test_dispatch_sends_x_runtime_header_explicit(monkeypatch):
    """X-Runtime header 透传显式 runtime 值。"""
    captured = {}

    class FakeResp:
        status = 200

        def getheader(self, n, d=None):
            if n == "X-Output-Count":
                return "0"
            return d

        def read(self):
            return b""

    class FakeConn:
        def putrequest(self, method, path):
            captured["headers"] = {}

        def putheader(self, key, value):
            captured["headers"][key] = value

        def endheaders(self):
            pass

        def send(self, chunk):
            pass

        def getresponse(self):
            return FakeResp()

        def close(self):
            pass

    monkeypatch.setattr("ttk.remote.dispatcher._create_connection",
                        lambda *a, **k: FakeConn())
    from ttk.remote.dispatcher import dispatch_to_remote
    dispatch_to_remote(op_name="add", inputs=[], endpoint_port=9, tenant_id="t",
                       runtime=42)
    assert captured["headers"].get("X-Runtime") == "42"


class TestRoundTrip:
    def test_nested_round_trips_through_match_params_v1(self):
        # G3：端到端 schema→serialize→server match_params_v1 结构对等
        # 未修代码上必红（p0 是 merged (2,1) 数组,不是 list）
        from ttk.remote.dispatcher import _build_input_schema, _serialize_to_file
        from ttk.remote.server.execution_container import match_params_v1
        import numpy as np
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
            import os; os.unlink(npz_path)
