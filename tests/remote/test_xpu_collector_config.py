"""EndpointView reads endpoints (incl. providers) from load_config (remote.endpoints).

The legacy ``restore_endpoints`` (env-based, TTK_XPU_ENDPOINTS) was deleted when
config loading was consolidated. EndpointView now pulls endpoints from
get_remote_config() (fed by load_config). The health file path still comes from
TTK_XPU_HEALTH_PATH env (worker-side).
"""
import json
import os

import pytest
import yaml


def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _load_endpoints(tmp_path, endpoints_list):
    """Write a remote.endpoints yaml and load_config it (reset module cache)."""
    import ttk.config.loader as loader
    loader._config = None
    config_path = tmp_path / "endpoints.yaml"
    config_path.write_text(yaml.safe_dump({"remote": {"endpoints": endpoints_list}}))
    loader.load_config(str(config_path))


def test_endpoints_read_from_config_with_providers(monkeypatch, tmp_path):
    """yaml endpoints (incl. providers) flow into EndpointView via get_remote_config."""
    from ttk.utilities.singleton import Singleton
    Singleton._instances.clear()
    _load_endpoints(tmp_path, [
        {"host": "127.0.0.1", "port": 9090, "providers": ["torch"]},
        {"host": "192.168.1.1", "port": 8080, "providers": ["tf"]},
    ])
    hp = str(tmp_path / "h.json")
    _write_health(hp, {
        "127.0.0.1:9090": {"alive": True, "providers": ["torch"]},
        "192.168.1.1:8080": {"alive": True, "providers": ["tf"]},
    })
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", hp)

    from ttk.remote.endpoint_view import EndpointView
    ev = EndpointView()

    # providers declared in yaml flow into the endpoint objects
    ep_by_host = {e.host: e for e in ev._endpoints}
    assert ep_by_host["127.0.0.1"].providers == ["torch"]
    assert ep_by_host["192.168.1.1"].providers == ["tf"]
    # both providers resolvable -> sorted union
    assert ev.resolve_providers() == ["tf", "torch"]
    assert ev.pick_endpoint("torch").host == "127.0.0.1"
    assert ev.pick_endpoint("tf").host == "192.168.1.1"


def test_no_endpoints_config_returns_empty(monkeypatch, tmp_path):
    """No remote.endpoints in config -> get_remote_config() is None -> empty endpoints.

    load_config with no remote.endpoints: get_remote_config returns None,
    EndpointView._endpoints is []. resolve_providers raises (nothing alive).
    """
    from ttk.utilities.singleton import Singleton
    Singleton._instances.clear()
    import ttk.config.loader as loader
    loader._config = None
    # load only default.yaml (no endpoints override) by passing no cli path
    loader.load_config()
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", str(tmp_path / "h.json"))

    from ttk.remote.config import get_remote_config
    assert get_remote_config() is None  # no endpoints configured

    from ttk.remote.endpoint_view import EndpointView
    ev = EndpointView()
    assert ev._endpoints == []
    with pytest.raises(RuntimeError):
        ev.resolve_providers()
