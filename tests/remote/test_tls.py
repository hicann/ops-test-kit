import http.client
import types
from unittest.mock import MagicMock

import pytest

from ttk.remote.tls import build_tls_connection, tls_from_config


def test_build_https_when_cert_and_key(monkeypatch):
    fake = MagicMock()
    monkeypatch.setattr("ssl.SSLContext", lambda *a, **k: fake)
    conn = build_tls_connection("x", 9090, 5, {"ca_cert": "ca", "cert": "c", "key": "k"})
    assert isinstance(conn, http.client.HTTPSConnection)
    fake.load_verify_locations.assert_called_once_with("ca")
    fake.load_cert_chain.assert_called_once_with("c", "k")


def test_build_https_when_ca_only(monkeypatch):
    # CA-only（单向 TLS，无 client cert）仍走 HTTPS
    fake = MagicMock()
    monkeypatch.setattr("ssl.SSLContext", lambda *a, **k: fake)
    conn = build_tls_connection("x", 9090, 5, {"ca_cert": "ca"})
    assert isinstance(conn, http.client.HTTPSConnection)
    fake.load_verify_locations.assert_called_once_with("ca")
    fake.load_cert_chain.assert_not_called()


def test_build_http_when_cert_without_key():
    # 严谨：cert 无 key -> 不走 HTTPS
    assert isinstance(
        build_tls_connection("x", 9090, 5, {"cert": "c"}),
        http.client.HTTPConnection,
    )


def test_build_http_when_no_tls():
    assert isinstance(build_tls_connection("x", 9090, 5, None), http.client.HTTPConnection)
    assert isinstance(build_tls_connection("x", 9090, 5, {}), http.client.HTTPConnection)


def _cfg(**kw):
    return types.SimpleNamespace(
        tls_ca=kw.get("ca", ""),
        tls_cert=kw.get("cert", ""),
        tls_key=kw.get("key", ""),
        tls_skip_verify=kw.get("skip_verify", False),
    )


def test_tls_from_config_raises_on_cert_without_key():
    with pytest.raises(RuntimeError, match="必须成对"):
        tls_from_config(_cfg(cert="c"))          # cert 无 key


def test_tls_from_config_raises_on_key_without_cert():
    with pytest.raises(RuntimeError, match="必须成对"):
        tls_from_config(_cfg(key="k"))           # key 无 cert


def test_tls_from_config_empty_when_no_tls():
    assert tls_from_config(_cfg()) == {}


def test_tls_from_config_full():
    assert tls_from_config(_cfg(ca="ca", cert="c", key="k")) == \
        {"ca_cert": "ca", "cert": "c", "key": "k", "skip_verify": False}


def test_tls_from_config_skip_verify_passthrough():
    """tls_skip_verify=True 透传到 tls dict（client 关 server hostname 校验）。"""
    assert tls_from_config(_cfg(ca="ca", skip_verify=True))["skip_verify"] is True


def test_build_tls_skip_verify_disables_hostname(monkeypatch):
    """skip_verify=True → ctx.check_hostname=False（关 server hostname 校验, CA 验证仍开）。"""
    import ssl
    monkeypatch.setattr("ssl.SSLContext.load_verify_locations", lambda self, *a: None)
    fake = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr("ssl.SSLContext", lambda *a, **k: fake)
    build_tls_connection("x", 9090, 5, {"ca_cert": "ca", "skip_verify": True})
    assert fake.check_hostname is False


def test_build_tls_default_keeps_hostname(monkeypatch):
    """默认（无 skip_verify）保持 check_hostname=True（严格）。"""
    import ssl
    monkeypatch.setattr("ssl.SSLContext.load_verify_locations", lambda self, *a: None)
    fake = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr("ssl.SSLContext", lambda *a, **k: fake)
    build_tls_connection("x", 9090, 5, {"ca_cert": "ca"})
    assert fake.check_hostname is True
