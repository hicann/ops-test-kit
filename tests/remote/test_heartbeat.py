import json
import http.client
import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def xpu_server():
    port = _find_free_port()
    sync_dir = tempfile.mkdtemp(prefix="ttk_hb_test_")
    tmp_dir = tempfile.mkdtemp(prefix="ttk_hb_tmp_")
    
    # Create a config file for the server
    import yaml
    config_file = tempfile.mktemp(prefix="xpu_server_hb_", suffix=".yaml")
    config_data = {
        "storage": {
            "sync_dir": sync_dir,
            "tmp_dir": tmp_dir,
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", str(port), "--dry-run", "--config", config_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=os.environ.copy())
    ready = False
    for _ in range(40):
        time.sleep(0.5)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            continue
    if not ready:
        proc.terminate()
        proc.wait(timeout=5)
        pytest.skip("xpu_server failed to start")
    yield proc, port
    proc.terminate()
    proc.wait(timeout=5)
    import shutil
    shutil.rmtree(sync_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def port(xpu_server):
    return xpu_server[1]


@pytest.fixture(scope="module")
def proc(xpu_server):
    return xpu_server[0]


class TestHeartbeatLifecycle:
    def test_send_heartbeat(self, port):
        tenant_id = "hb_test_001"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", f"/v1/heartbeat?tenant_id={tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()
        conn.close()

    def test_heartbeat_then_cleanup(self, port):
        tenant_id = "hb_test_002"
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

        # Heartbeat
        conn.request("GET", f"/v1/heartbeat?tenant_id={tenant_id}")
        conn.getresponse().read()

        # Cleanup
        conn.request("DELETE", f"/v1/tenant/{tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["cleaned"] is True
        conn.close()


class TestHeartbeatModule:
    def test_heartbeat_loop_import(self):
        """Verify heartbeat module is importable and has correct interfaces."""
        from ttk.remote.heartbeat import (
            heartbeat_loop, _probe_one, _cleanup_all)
        assert callable(heartbeat_loop)
        assert callable(_probe_one)
        assert callable(_cleanup_all)

    def test_probe_one_to_server(self, port):
        """Test _probe_one helper directly against the merged endpoint."""
        from ttk.remote.heartbeat import _probe_one
        from ttk.remote.config import Endpoint

        ep = Endpoint(host="127.0.0.1", port=port)
        out = {}
        _probe_one(ep, "hb_direct_test", out, None)  # should not raise
        ep_key = f"127.0.0.1:{port}"
        assert ep_key in out
        assert out[ep_key]["alive"] is True

    def test_cleanup_all(self, port):
        """Test _cleanup_all helper directly."""
        from ttk.remote.heartbeat import _cleanup_all
        from ttk.remote.config import Endpoint

        ep = Endpoint(host="127.0.0.1", port=port)

        # Create tenant first
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/v1/heartbeat?tenant_id=hb_cleanup_test")
        conn.getresponse().read()
        conn.close()

        # Cleanup via helper
        _cleanup_all([ep], "hb_cleanup_test", None)

        # Verify cleaned — DELETE returns cleaned=False on second attempt
        conn2 = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn2.request("DELETE", "/v1/tenant/hb_cleanup_test")
        resp = conn2.getresponse()
        data = json.loads(resp.read())
        assert data["cleaned"] is False  # already cleaned
        conn2.close()
