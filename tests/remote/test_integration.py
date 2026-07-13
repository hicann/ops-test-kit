"""Full-pipeline integration tests using xpu_server --dry-run.

Validates end-to-end flow: config -> heartbeat -> dispatch -> verify -> cleanup.
Uses port 19094 to avoid conflicts with other test suites.
"""
import base64
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

# Isolated sync directory so this suite does not clash with test_sync.py
_INTEGRATION_SYNC_DIR = os.path.join(tempfile.gettempdir(), "ttk_sync_integration")


@pytest.fixture(scope="module")
def xpu_server():
    # Clean sync directory
    if os.path.exists(_INTEGRATION_SYNC_DIR):
        shutil.rmtree(_INTEGRATION_SYNC_DIR, ignore_errors=True)
    os.makedirs(_INTEGRATION_SYNC_DIR, exist_ok=True)

    # Create a config file for the server
    import yaml
    tmp_dir = tempfile.mkdtemp(prefix="ttk_tmp_integration_")
    config_file = os.path.join(tempfile.gettempdir(), "xpu_server_integration.yaml")
    config_data = {
        "storage": {
            "sync_dir": _INTEGRATION_SYNC_DIR,
            "tmp_dir": tmp_dir,
        },
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", "19094", "--dry-run", "--config", config_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    # Wait for server readiness
    for _ in range(20):
        time.sleep(0.5)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 19094, timeout=1)
            conn.request("GET", "/v1/heartbeat")
            resp = conn.getresponse()
            resp.read()
            conn.close()
            if resp.status == 200:
                break
        except (ConnectionRefusedError, OSError):
            continue
    yield proc
    proc.terminate()
    proc.wait(timeout=5)
    shutil.rmtree(_INTEGRATION_SYNC_DIR, ignore_errors=True)


class TestFullPipeline:
    """End-to-end: config -> heartbeat -> dispatch -> verify -> cleanup."""

    def test_api_mode_full_flow(self, xpu_server):
        """API mode: config -> heartbeat -> /run -> verify -> cleanup."""
        from ttk.remote.dispatcher import dispatch_to_remote
        from ttk.remote.config import RemoteConfig

        # 1. Config
        config = RemoteConfig.from_dict({"endpoints": [{"host": "127.0.0.1", "port": 19094}]})
        tenant_id = "integration_test_001"

        # 2. Heartbeat
        conn = http.client.HTTPConnection("127.0.0.1", 19094, timeout=5)
        conn.request("GET", f"/v1/heartbeat?tenant_id={tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.read()

        # 3. Dispatch
        inputs = [np.random.randn(8, 16).astype(np.float32)]
        outputs = dispatch_to_remote(
            op_name="softmax_v2",
            inputs=inputs,
            provider="torch",
            attrs={"axis": -1},
            endpoint_host="127.0.0.1",
            endpoint_port=19094,
            tenant_id=tenant_id,
        )

        # 4. Verify
        assert len(outputs) >= 1
        assert isinstance(outputs[0], np.ndarray)

        # 5. Cleanup
        conn.request("DELETE", f"/v1/tenant/{tenant_id}")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["cleaned"] is True
        conn.close()

    def test_multi_input(self, xpu_server):
        """Multiple inputs, verify serialization roundtrip."""
        from ttk.remote.dispatcher import dispatch_to_remote

        inputs = [
            np.random.randn(4, 8).astype(np.float32),
            np.random.randn(4, 8).astype(np.float32),
        ]
        outputs = dispatch_to_remote(
            op_name="add",
            inputs=inputs,
            provider="torch",
            attrs={},
            endpoint_host="127.0.0.1",
            endpoint_port=19094,
            tenant_id="integration_test_002",
        )
        assert len(outputs) >= 1


class TestRemoteConfigIntegration:
    def test_tenant_id_available(self):
        from ttk.remote import get_tenant_id
        tid = get_tenant_id()
        assert isinstance(tid, str)
        assert len(tid) == 12


class TestSyncIntegration:
    def test_sync_then_run(self, xpu_server):
        """Sync spec files then run -- dry-run mode ignores spec."""
        tenant_id = "integration_sync_001"

        # Sync a dummy spec
        content = base64.b64encode(b"# dummy spec").decode()
        h = hashlib.sha256(b"# dummy spec").hexdigest()
        body = json.dumps({
            "files": {"nn/dummy.py": {"content": content, "hash": f"sha256:{h}"}}
        })
        conn = http.client.HTTPConnection("127.0.0.1", 19094, timeout=5)
        conn.request("POST", "/v1/sync", body=body,
                     headers={"Content-Type": "application/json",
                              "X-Tenant-ID": tenant_id})
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 200

        # Run (dry-run ignores spec, just returns random)
        from ttk.remote.dispatcher import dispatch_to_remote
        outputs = dispatch_to_remote(
            op_name="dummy",
            inputs=[np.array([1.0])],
            endpoint_host="127.0.0.1",
            endpoint_port=19094,
            tenant_id=tenant_id,
        )
        assert len(outputs) >= 1
        conn.close()
