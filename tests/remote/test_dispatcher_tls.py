"""Test _create_connection delegates TLS construction to the shared tls module.

The TLS gate (ca / cert+key pairing, RuntimeError on mismatch) and the
SSLContext/HTTP(S)Connection wiring are covered by tests/remote/test_tls.py
(Task 4). This file only asserts the delegation seam: dispatcher's
`_create_connection` calls `tls_from_config(get_remote_config())` and forwards
the result to `build_tls_connection` with the right args.
"""


def test_create_connection_delegates_to_build_tls_connection(monkeypatch, tmp_path):
    """_create_connection forwards (host, port, timeout, tls_dict) to build_tls_connection."""
    from ttk.remote import dispatcher
    from ttk.config import loader as loader

    # Configure TLS via a yaml config (was set_remote_config before)
    (tmp_path / "ttk.conf.yaml").write_text(
        "remote:\n"
        "  endpoints:\n"
        "    - {host: '127.0.0.1', port: 9090}\n"
        "  tls_ca: '/path/to/ca.pem'\n"
        "  tls_cert: '/path/to/cert.pem'\n"
        "  tls_key: '/path/to/key.pem'\n"
    )
    loader._config = None
    loader.load_config(str(tmp_path / "ttk.conf.yaml"))

    captured = {}

    def fake_build(host, port, timeout, tls):
        captured["host"] = host
        captured["port"] = port
        captured["timeout"] = timeout
        captured["tls"] = tls
        return "sentinel-conn"

    # Patch the names _create_connection imports — it does a function-level
    # `from ttk.remote.tls import tls_from_config, build_tls_connection`, so
    # patch at the source module.
    monkeypatch.setattr("ttk.remote.tls.build_tls_connection", fake_build)

    conn = dispatcher._create_connection("127.0.0.1", 9090, timeout=5)

    assert conn == "sentinel-conn"
    assert captured == {
        "host": "127.0.0.1",
        "port": 9090,
        "timeout": 5,
        "tls": {"ca_cert": "/path/to/ca.pem",
                "cert": "/path/to/cert.pem",
                "key": "/path/to/key.pem",
                "skip_verify": False},
    }
