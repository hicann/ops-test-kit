import json
import os
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="module")
def xpu_server():
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", "19090", "--dry-run"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for server to become ready (framework detection may be slow)
    import http.client
    for _ in range(30):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19090, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                break
        except (ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture
def http_conn():
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", 19090, timeout=5)
    yield conn
    conn.close()


class TestV1HeartbeatEndpoint:
    """Merged /v1/heartbeat: old /health + /v1/detect + /heartbeat in one."""

    def test_v1_heartbeat_merges_health_detect_register(self, xpu_server, http_conn):
        conn = http_conn
        conn.request("GET", "/v1/heartbeat?tenant_id=t1")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 200
        assert body["status"] == "ok"
        assert isinstance(body["providers"], list)
        assert "hardware" in body and "device_count" in body

    def test_v1_heartbeat_without_tenant_still_returns_capabilities(self, xpu_server, http_conn):
        conn = http_conn
        conn.request("GET", "/v1/heartbeat")
        resp = conn.getresponse()
        body = json.loads(resp.read())
        assert resp.status == 200
        assert "providers" in body

    def test_old_endpoints_are_gone(self, xpu_server, http_conn):
        for path in ("/health", "/v1/detect", "/heartbeat"):
            conn = http_conn
            conn.request("GET", path)
            resp = conn.getresponse()
            resp.read()
            assert resp.status == 404


class TestTenantCleanup:
    def test_delete_tenant(self, xpu_server, http_conn):
        http_conn.request("DELETE", "/v1/tenant/test_001")
        resp = http_conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "cleaned" in data


class TestDryRunRun:
    def test_dry_run_returns_random(self, xpu_server, http_conn):
        import numpy as np
        import io

        inputs = [np.random.randn(4, 8).astype(np.float32)]
        buf = io.BytesIO()
        np.savez_compressed(buf, **{f"a{i}": a for i, a in enumerate(inputs)})
        body = buf.getvalue()

        http_conn.request("POST", "/v1/run", body=body,
                          headers={"X-Execution-Type": "api",
                                   "X-Provider": "torch",
                                   "X-Input-Count": "1",
                                   "X-Mode": "data",
                                   "X-Tenant-ID": "dry_run_test",
                                   "Content-Type": "application/octet-stream"})
        resp = http_conn.getresponse()
        assert resp.status == 200
        assert int(resp.getheader("X-Output-Count", "0")) >= 1
        resp_body = resp.read()
        assert len(resp_body) > 0


class TestResolveClass:
    def test_simple_class(self):
        from ttk.remote.server.xpu_server import _resolve_class
        import types
        class Simple: pass
        mod = types.ModuleType("test_mod")
        mod.Simple = Simple
        assert _resolve_class(mod, "Simple") is Simple

    def test_nested_class(self):
        from ttk.remote.server.xpu_server import _resolve_class
        import types
        class Outer:
            class Inner: pass
        mod = types.ModuleType("test_mod")
        mod.Outer = Outer
        result = _resolve_class(mod, "Outer.Inner")
        assert result is Outer.Inner

    def test_deeply_nested(self):
        from ttk.remote.server.xpu_server import _resolve_class
        import types
        class A:
            class B:
                class C: pass
        mod = types.ModuleType("test_mod")
        mod.A = A
        result = _resolve_class(mod, "A.B.C")
        assert result is A.B.C

    def test_missing_attr_raises(self):
        from ttk.remote.server.xpu_server import _resolve_class
        import pytest
        import types
        mod = types.ModuleType("test_mod")
        with pytest.raises(AttributeError):
            _resolve_class(mod, "NonExistent")


class TestAtomicWriteFile:
    def test_writes_content_creates_dirs(self, tmp_path):
        from ttk.remote.server.xpu_server import _atomic_write_file
        target = tmp_path / "sub" / "mod.py"
        _atomic_write_file(str(target), b"hello")
        assert target.read_bytes() == b"hello"

    def test_overwrites_existing(self, tmp_path):
        from ttk.remote.server.xpu_server import _atomic_write_file
        target = tmp_path / "mod.py"
        target.write_bytes(b"old")
        _atomic_write_file(str(target), b"new")
        assert target.read_bytes() == b"new"

    def test_no_tmp_residue(self, tmp_path):
        from ttk.remote.server.xpu_server import _atomic_write_file
        target = tmp_path / "mod.py"
        _atomic_write_file(str(target), b"x")
        leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []


@pytest.fixture(scope="module")
def sync_server(tmp_path_factory):
    sync_dir = tmp_path_factory.mktemp("sync_root")
    tmp_dir = tmp_path_factory.mktemp("tmp_root")
    
    # Create a config file for the server
    import yaml
    config_file = tmp_path_factory.mktemp("config") / "xpu_server.yaml"
    config_data = {
        "storage": {
            "sync_dir": str(sync_dir),
            "tmp_dir": str(tmp_dir),
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", "19095", "--dry-run", "--config", str(config_file)],
        env=os.environ.copy(),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import http.client
    for _ in range(30):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19095, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            conn.close()
            if resp.status == 200:
                break
        except (ConnectionRefusedError, OSError):
            pass
        time.sleep(0.5)
    yield proc
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestSyncHashSkip:
    def test_second_sync_with_same_hash_is_skipped(self, sync_server):
        import base64
        import hashlib
        import http.client
        content = b"def f():\n    return 1\n"
        file_hash = hashlib.sha256(content).hexdigest()
        body = json.dumps({"files": {
            "mod.py": {
                "content": base64.b64encode(content).decode(),
                "hash": file_hash,
            }
        }})

        def post():
            conn = http.client.HTTPConnection("127.0.0.1", 19095, timeout=5)
            conn.request("POST", "/v1/sync", body=body,
                         headers={"Content-Type": "application/json",
                                  "X-Tenant-ID": "skip_test"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            conn.close()
            return data

        first = post()
        second = post()
        assert first["synced"] == 1
        assert first["skipped"] == 0
        assert second["synced"] == 0
        assert second["skipped"] == 1


def test_resolve_api_string_removed():
    """Regression test: _resolve_api_string was dead code and should be deleted."""
    import ttk.remote.server.xpu_server as xs
    assert not hasattr(xs, "_resolve_api_string"), "_resolve_api_string should be deleted (dead code)"
