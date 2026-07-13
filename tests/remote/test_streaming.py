"""Tests for tempfile-based streaming serialization."""
import os
import io
import numpy as np
import pytest


class TestServerSideStreaming:
    """Test xpu_server streaming receive (tempfile instead of BytesIO)."""

    def test_receive_body_to_file(self, tmp_path):
        """_receive_body_to_file should write request body to a temp file."""
        from ttk.remote.server.xpu_server import _receive_body_to_file

        # Simulate: write numpy data to a BytesIO buffer as if it were a request body
        original = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(original)})
        body_data = buf.getvalue()

        # Create a minimal mock handler with rfile and headers
        class MockHandler:
            class MockHeaders:
                def get(self, name, default=0):
                    if name == "Content-Length":
                        return str(len(body_data))
                    return default
            def __init__(self):
                self.headers = self.MockHeaders()
                self.rfile = io.BytesIO(body_data)

        handler = MockHandler()
        tmp_path = _receive_body_to_file(handler)
        try:
            assert tmp_path is not None
            assert os.path.isfile(tmp_path)
            assert os.path.getsize(tmp_path) > 0

            # Verify data roundtrips (the child executor reads tmp_in with np.load
            # directly; _restore_inputs_from_file was removed as dead code).
            npz = np.load(tmp_path)
            restored = [npz[f"a{i}"] for i in range(2)]
            assert len(restored) == 2
            np.testing.assert_array_equal(restored[0], original[0])
            np.testing.assert_array_equal(restored[1], original[1])
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_receive_empty_body(self):
        """_receive_body_to_file returns None when Content-Length is 0."""
        from ttk.remote.server.xpu_server import _receive_body_to_file

        class MockHandler:
            class MockHeaders:
                def get(self, name, default=0):
                    return "0"
            def __init__(self):
                self.headers = self.MockHeaders()
                self.rfile = io.BytesIO(b"")

        result = _receive_body_to_file(MockHandler())
        assert result is None


class TestStreamingSerialization:
    def test_save_to_tempfile_and_reload(self):
        from ttk.remote.dispatcher import _serialize_to_file, _load_npz_outputs

        original = [
            np.random.randn(100, 200).astype(np.float32),
            np.random.randn(50, 80).astype(np.int64),
        ]
        tmp = _serialize_to_file(original)
        try:
            assert os.path.isfile(tmp)
            assert os.path.getsize(tmp) > 0
            schema = [{"index": 0, "dtype": "float32"}, {"index": 1, "dtype": "int64"}]
            outputs = _load_npz_outputs(tmp, schema)
            assert len(outputs) == 2
            np.testing.assert_array_almost_equal(outputs[0], original[0])
            np.testing.assert_array_equal(outputs[1], original[1])
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_skip_none_in_streaming(self):
        from ttk.remote.dispatcher import _serialize_to_file, _load_npz_outputs

        inputs = [np.array([1.0, 2.0]), None, np.array([3.0, 4.0])]
        tmp = _serialize_to_file(inputs)
        try:
            schema = [{"index": 0, "dtype": "float64"}, {"index": 1, "dtype": "float64"}]
            outputs = _load_npz_outputs(tmp, schema)
            np.testing.assert_array_equal(outputs[0], np.array([1.0, 2.0]))
            np.testing.assert_array_equal(outputs[1], np.array([3.0, 4.0]))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def test_cleanup_after_use(self):
        from ttk.remote.dispatcher import _serialize_to_file
        tmp = _serialize_to_file([np.array([1.0])])
        assert os.path.exists(tmp)
        os.unlink(tmp)
        assert not os.path.exists(tmp)

    def test_roundtrip_single_tensor(self):
        from ttk.remote.dispatcher import _serialize_to_file, _load_npz_outputs

        original = [np.array([1.0, 2.0, 3.0], dtype=np.float64)]
        tmp = _serialize_to_file(original)
        try:
            schema = [{"index": 0, "dtype": "float64"}]
            outputs = _load_npz_outputs(tmp, schema)
            np.testing.assert_array_equal(outputs[0], original[0])
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)


class _FakeResp:
    """Fake HTTPResponse: getheader + read(amt) for streaming."""
    def __init__(self, body, content_length=None):
        self._body = body
        self._cl = content_length
        self._pos = 0

    def getheader(self, name, default=None):
        if name == "Content-Length" and self._cl is not None:
            return str(self._cl)
        return default

    def read(self, amt=-1):
        if amt is None or amt < 0 or amt >= len(self._body) - self._pos:
            chunk = self._body[self._pos:]
        else:
            chunk = self._body[self._pos:self._pos + amt]
        self._pos += len(chunk)
        return chunk


class TestRespToNpzSource:
    """Adaptive response: Content-Length > threshold -> file, else BytesIO."""
    def test_small_cl_returns_bytesio(self, tmp_path):
        from ttk.remote.dispatcher import _resp_to_npz_source
        body = b"x" * 100
        src = _resp_to_npz_source(_FakeResp(body, content_length=100), str(tmp_path))
        assert isinstance(src, io.BytesIO)
        assert src.read() == body

    def test_large_cl_streams_to_file(self, tmp_path):
        from ttk.remote.dispatcher import _resp_to_npz_source, RESP_MEM_THRESHOLD
        buf = io.BytesIO()
        arr = np.array([1.0, 2.0, 3.0])
        np.savez_compressed(buf, a0=arr)
        src = _resp_to_npz_source(
            _FakeResp(buf.getvalue(), content_length=RESP_MEM_THRESHOLD + 1),
            str(tmp_path))
        assert isinstance(src, str)            # path, not BytesIO
        assert src.endswith("resp.npz")
        npz = np.load(src)
        np.testing.assert_array_equal(npz["a0"], arr)

    def test_missing_cl_falls_back_to_memory(self, tmp_path):
        from ttk.remote.dispatcher import _resp_to_npz_source
        body = b"no-content-length"
        src = _resp_to_npz_source(_FakeResp(body, content_length=None), str(tmp_path))
        assert isinstance(src, io.BytesIO)
        assert src.read() == body
