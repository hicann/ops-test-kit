"""E2E mTLS handshake test: gen cert + subprocess tls server + client real connect.

Verifies the full mTLS path (server wrap_socket CERT_REQUIRED + client
HTTPSConnection) that unit tests (config loading / HTTP-vs-HTTPS selection)
don't cover. See spec 2026-06-19-mtls-e2e-test-design.
"""
import os
import socket
import ssl
import subprocess
import sys
import time

import pytest

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "gen_tls_certs.sh")
_SERVER_MOD = "server.xpu_server"


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


_TENANT = "mtls_test"   # client cert tenant → <TENANT>.client.{crt,key}


def _gen_certs(d):
    """Generate a full CA + server + client cert set via the subcommand API.

    gen_tls_certs.sh is subcommand-based (init-ca/server/client); server SAN
    IP:127.0.0.1 matches the test's loopback connect (TLS hostname check).
    """
    script = os.path.abspath(_SCRIPT)
    d = str(d)
    subprocess.run(["bash", script, "init-ca", d], check=True, capture_output=True)
    subprocess.run(["bash", script, "server", d, "IP:127.0.0.1"], check=True, capture_output=True)
    subprocess.run(["bash", script, "client", d, _TENANT], check=True, capture_output=True)


def _wait_server(host, port, timeout=30):
    """Two-phase readiness: TCP reachable, then a short settle for TLS wrap."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.3)
    else:
        raise RuntimeError(f"server not reachable on {host}:{port}")
    time.sleep(1)   # let the TLS socket settle


def _set_tls(ca, cert="", key=""):
    """Set mTLS client certs via load_config (自管 _config)。

    原 set_remote_config(RemoteConfig(...)) 被删 lazy fallback 后不再适用；
    改为写临时 yaml（remote.endpoints 占位 + tls 段）再 load_config。
    _create_connection 实参用 mtls_env 的 host/port，endpoints 仅占位满足
    get_remote_config 预检（endpoints 非空）。
    """
    import os
    import tempfile
    import ttk.config.loader as loader
    import yaml
    remote = {
        "endpoints": [{"host": "127.0.0.1", "port": 0}],   # placeholder; _create_connection uses args
        "tls_ca": ca, "tls_cert": cert, "tls_key": key,
    }
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False) as f:
        yaml.dump({"remote": remote}, f)
        path = f.name
    try:
        loader._config = None
        loader.load_config(path)
    finally:
        os.unlink(path)   # load_config 已读完，清掉临时 yaml


def _reset_tls():
    """恢复默认 config（清缓存后重新 load 默认链）。"""
    import ttk.config.loader as loader
    loader._config = None
    loader.load_config()  # 恢复默认


def _https_get(host, port):
    from ttk.remote.dispatcher import _create_connection
    conn = _create_connection(host, port, timeout=5)
    try:
        conn.request("GET", "/v1/heartbeat")
        resp = conn.getresponse()
        resp.read()
        return resp.status
    finally:
        conn.close()


@pytest.fixture(scope="module")
def mtls_env(tmp_path_factory):
    import yaml
    certs = tmp_path_factory.mktemp("certs")
    wrong_certs = tmp_path_factory.mktemp("wrong_certs")
    _gen_certs(certs)
    _gen_certs(wrong_certs)        # independent CA for the wrong-CA negative test

    port = _free_port()
    sync = tmp_path_factory.mktemp("sync")
    tmp = tmp_path_factory.mktemp("tmp")
    yaml_path = tmp_path_factory.mktemp("config") / "tls.yaml"
    yaml.dump({
        "server": {"bind": "127.0.0.1", "port": port, "max_concurrent": 4},
        "execution": {"sandbox": "none"},
        "storage": {"sync_dir": str(sync), "tmp_dir": str(tmp)},
        "tls": {"enabled": True,
                "ca_cert": str(certs / "ca.crt"),
                "server_cert": str(certs / "server.crt"),
                "server_key": str(certs / "server.key")},
    }, open(yaml_path, "w"))

    proc = subprocess.Popen(
        [sys.executable, "-m", _SERVER_MOD, "--port", str(port),
         "--devices", "cpu", "--config", str(yaml_path)],
        env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        _wait_server("127.0.0.1", port)
        yield {"host": "127.0.0.1", "port": port, "proc": proc,
               "ca": str(certs / "ca.crt"),
               "client_crt": str(certs / f"{_TENANT}.client.crt"),
               "client_key": str(certs / f"{_TENANT}.client.key"),
               "wrong_ca": str(wrong_certs / "ca.crt")}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


class TestMTLSE2E:
    @pytest.fixture(autouse=True)
    def _reset_tls_after(self):
        yield
        _reset_tls()

    def test_handshake_succeeds(self, mtls_env):
        # positive: full client cert + matching CA → mTLS handshake → 200
        _set_tls(mtls_env["ca"], mtls_env["client_crt"], mtls_env["client_key"])
        assert _https_get(mtls_env["host"], mtls_env["port"]) == 200

    def test_missing_client_cert_rejected(self, mtls_env):
        # negative: no client cert → server CERT_REQUIRED rejects → SSLError
        _set_tls(mtls_env["ca"])   # tls_ca only, no client cert
        with pytest.raises(ssl.SSLError):
            _https_get(mtls_env["host"], mtls_env["port"])

    def test_wrong_ca_rejected(self, mtls_env):
        # negative: client trusts a different CA → verifies server cert fails (client-side)
        _set_tls(mtls_env["wrong_ca"], mtls_env["client_crt"], mtls_env["client_key"])
        with pytest.raises(ssl.SSLError):
            _https_get(mtls_env["host"], mtls_env["port"])
