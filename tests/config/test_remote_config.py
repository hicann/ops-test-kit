"""Tests for RemoteConfig dataclass field extensions."""

import pytest
from ttk.config.loader import RemoteConfig, Endpoint


def test_remote_config_backoff_fields():
    """Verify backoff configuration fields exist and parse correctly."""
    config = RemoteConfig.from_dict({
        "endpoints": [{"host": "127.0.0.1", "port": 9090}],
        "backoff_base_s": 0.5,
        "backoff_max_s": 60.0,
        "backoff_jitter": 0.1,
    })
    
    assert config.backoff_base_s == 0.5
    assert config.backoff_max_s == 60.0
    assert config.backoff_jitter == 0.1


def test_remote_config_retry_fields():
    """Verify retry limit fields exist and parse correctly."""
    config = RemoteConfig.from_dict({
        "endpoints": [{"host": "127.0.0.1", "port": 9090}],
        "max_503_retries": 15,
        "max_conn_retries": 8,
        "dispatch_deadline_s": 600,
    })
    
    assert config.max_503_retries == 15
    assert config.max_conn_retries == 8
    assert config.dispatch_deadline_s == 600


def test_remote_config_tls_fields():
    """Verify TLS certificate path fields exist and parse correctly."""
    config = RemoteConfig.from_dict({
        "endpoints": [{"host": "127.0.0.1", "port": 9090}],
        "tls_ca": "/path/to/ca.pem",
        "tls_cert": "/path/to/cert.pem",
        "tls_key": "/path/to/key.pem",
    })
    
    assert config.tls_ca == "/path/to/ca.pem"
    assert config.tls_cert == "/path/to/cert.pem"
    assert config.tls_key == "/path/to/key.pem"


def test_remote_config_defaults():
    """Verify all new fields have sensible defaults when not specified."""
    config = RemoteConfig.from_dict({
        "endpoints": [{"host": "127.0.0.1", "port": 9090}],
    })
    
    # Backoff defaults (match current code in dispatcher.py for backward compat)
    assert config.backoff_base_s == 0.5
    assert config.backoff_max_s == 10.0
    assert config.backoff_jitter == 0.25
    
    # Retry defaults
    assert config.max_503_retries == 10
    assert config.max_conn_retries == 5
    assert config.dispatch_deadline_s == 300
    
    # TLS defaults (empty strings, not None, to match YAML pattern)
    assert config.tls_ca == ""
    assert config.tls_cert == ""
    assert config.tls_key == ""


def test_remote_config_all_fields_together():
    """Verify all fields work together in a complete configuration."""
    config = RemoteConfig.from_dict({
        "endpoints": [
            {"host": "127.0.0.1", "port": 9090},
            {"host": "192.168.1.1", "port": 8080},
        ],
        "backoff_base_s": 1.0,
        "backoff_max_s": 30.0,
        "backoff_jitter": 0.05,
        "max_503_retries": 20,
        "max_conn_retries": 10,
        "dispatch_deadline_s": 900,
        "tls_ca": "/etc/ssl/ca.crt",
        "tls_cert": "/etc/ssl/client.crt",
        "tls_key": "/etc/ssl/client.key",
    })
    
    assert len(config.endpoints) == 2
    assert config.backoff_base_s == 1.0
    assert config.backoff_max_s == 30.0
    assert config.backoff_jitter == 0.05
    assert config.max_503_retries == 20
    assert config.max_conn_retries == 10
    assert config.dispatch_deadline_s == 900
    assert config.tls_ca == "/etc/ssl/ca.crt"
    assert config.tls_cert == "/etc/ssl/client.crt"
    assert config.tls_key == "/etc/ssl/client.key"
