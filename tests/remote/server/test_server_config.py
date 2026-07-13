"""Test that server config reads all fields from YAML, not env vars."""

import os
import tempfile
import yaml
import pytest


def test_server_config_reads_all_fields(monkeypatch):
    """Server config should read all fields from YAML, not env vars."""
    # Ensure no env vars are set
    monkeypatch.delenv("TTK_XPU_MAX_CONCURRENT", raising=False)
    monkeypatch.delenv("TTK_XPU_GATE_WAIT_S", raising=False)
    monkeypatch.delenv("TTK_XPU_RUN_DEADLINE_S", raising=False)
    
    # Create a temporary YAML file with non-default values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "server": {
                "bind": "0.0.0.0",
                "port": 9999,
                "max_concurrent": 20,
                "run_deadline_s": 600,
            },
            "execution": {
                "gate_wait_s": 5.0,
            },
            "storage": {
                "sync_dir": "/custom/sync",
                "tmp_dir": "/custom/tmp",
            },
        }, f)
        config_path = f.name

    try:
        from ttk.remote.server.config import load_server_config

        config = load_server_config(config_path)

        # Verify all fields are read from YAML
        assert config["bind"] == "0.0.0.0"
        assert config["port"] == 9999
        assert config["max_concurrent"] == 20
        assert config["run_deadline_s"] == 600
        assert config["gate_wait_s"] == 5.0
        assert config["sync_dir"] == "/custom/sync"
        assert config["tmp_dir"] == "/custom/tmp"
    finally:
        os.unlink(config_path)


def test_server_config_uses_defaults_when_yaml_missing():
    """Server config should use built-in defaults when YAML file doesn't exist."""
    from ttk.remote.server.config import load_server_config
    
    config = load_server_config("/nonexistent/path.yaml")
    
    # Verify defaults are used
    assert config["bind"] == "127.0.0.1"
    assert config["port"] == 9090
    assert isinstance(config["max_concurrent"], int)
    assert config["max_concurrent"] > 0
    assert isinstance(config["run_deadline_s"], int)
    assert config["run_deadline_s"] > 0
    assert isinstance(config["gate_wait_s"], (int, float))
    assert config["gate_wait_s"] > 0
    assert isinstance(config["sync_dir"], str)
    assert isinstance(config["tmp_dir"], str)


def test_server_config_no_env_var_override(monkeypatch):
    """Verify that env vars do NOT override YAML values (per unified config design)."""
    # Set env vars that should be ignored
    monkeypatch.setenv("TTK_XPU_MAX_CONCURRENT", "999")
    monkeypatch.setenv("TTK_XPU_GATE_WAIT_S", "99.9")
    monkeypatch.setenv("TTK_XPU_RUN_DEADLINE_S", "999")
    
    # Create YAML with specific values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "server": {
                "max_concurrent": 20,
                "run_deadline_s": 600,
            },
            "execution": {
                "gate_wait_s": 5.0,
            },
        }, f)
        config_path = f.name

    try:
        from ttk.remote.server.config import load_server_config

        config = load_server_config(config_path)

        # Env vars should NOT override YAML
        assert config["max_concurrent"] == 20, \
            "max_concurrent should come from YAML, not TTK_XPU_MAX_CONCURRENT env var"
        assert config["run_deadline_s"] == 600, \
            "run_deadline_s should come from YAML, not TTK_XPU_RUN_DEADLINE_S env var"
        assert config["gate_wait_s"] == 5.0, \
            "gate_wait_s should come from YAML, not TTK_XPU_GATE_WAIT_S env var"
    finally:
        os.unlink(config_path)
