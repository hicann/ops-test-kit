import json
import os
import pytest
import yaml
from ttk.remote.endpoint_view import EndpointView, _parse_provider_filter


def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _endpoints_yaml(endpoints_list):
    """Build a remote.endpoints yaml snippet from a list of endpoint dicts."""
    return yaml.safe_dump({"remote": {"endpoints": endpoints_list}})


def _make_ev(monkeypatch, tmp_path, health_data, endpoints_list, clear_singleton=True):
    """Wire config + health file, then construct a fresh EndpointView.

    Endpoints now come from load_config(yaml) (remote.endpoints) instead of
    TTK_XPU_ENDPOINTS env. Health file path still read from TTK_XPU_HEALTH_PATH.
    """
    if clear_singleton:
        from ttk.utilities.singleton import Singleton
        Singleton._instances.clear()
    import ttk.config.loader as loader
    loader._config = None
    config_path = tmp_path / "endpoints.yaml"
    config_path.write_text(_endpoints_yaml(endpoints_list))
    loader.load_config(str(config_path))
    health_path = str(tmp_path / "h.json")
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    _write_health(health_path, health_data)
    return EndpointView()


# --- resolve_providers ---

def test_resolve_no_spec_no_cli_returns_all_effective_sorted(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.resolve_providers() == ["tf", "torch"]  # sorted union

def test_resolve_spec_intersects(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.resolve_providers(spec_providers=["torch"]) == ["torch"]


def test_resolve_spec_order_preserves_priority_not_sorted(monkeypatch, tmp_path):
    """spec insertion order = app-declared priority (spec §③).

    effective={torch,tf}: resolve follows spec order, not sorted order, so the
    caller's first spec entry wins priority. Two non-sorted spec orders produce
    two different result orders.
    """
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.resolve_providers(spec_providers=["tf", "torch"]) == ["tf", "torch"]
    assert ev.resolve_providers(spec_providers=["torch", "tf"]) == ["torch", "tf"]

def test_resolve_cli_intersects(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.resolve_providers(cli_providers=["tf"]) == ["tf"]

def test_resolve_yaml_filters(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "10.0.0.1", "port": 9090, "providers": ["torch"]}])  # yaml=[torch]
    assert ev.resolve_providers() == ["torch"]

def test_resolve_dead_ep_skipped(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": False, "providers": ["torch"]},
                   "10.0.0.2:9090": {"alive": True, "providers": ["tf"]}},
                  [{"host": "10.0.0.1", "port": 9090}, {"host": "10.0.0.2", "port": 9090}])
    assert ev.resolve_providers() == ["tf"]

def test_resolve_all_dead_raises(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": False, "providers": ["torch"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    with pytest.raises(RuntimeError):
        ev.resolve_providers()

def test_resolve_spec_not_in_detect_raises(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    with pytest.raises(RuntimeError):
        ev.resolve_providers(spec_providers=["tf"])

# --- pick_endpoint round-robin ---

def test_pick_endpoint_round_robin(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]},
                   "10.0.0.2:9090": {"alive": True, "providers": ["torch"]}},
                  [{"host": "10.0.0.1", "port": 9090}, {"host": "10.0.0.2", "port": 9090}])
    ep1 = ev.pick_endpoint("torch")
    ep2 = ev.pick_endpoint("torch")
    assert ep1 is not ep2  # load-balanced across the two

def test_pick_endpoint_none_when_dead(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": False, "providers": ["torch"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.pick_endpoint("torch") is None

def test_pick_endpoint_none_when_provider_absent(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}},
                  [{"host": "10.0.0.1", "port": 9090}])
    assert ev.pick_endpoint("tf") is None

# --- Singleton + parse_provider_filter ---

def test_singleton_same_instance(monkeypatch, tmp_path):
    ev1 = _make_ev(monkeypatch, tmp_path,
                   {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}},
                   [{"host": "10.0.0.1", "port": 9090}], clear_singleton=True)
    ev2 = EndpointView()
    assert ev1 is ev2

def test_parse_provider_filter():
    assert _parse_provider_filter("torch, tf") == ["torch", "tf"]

def test_parse_provider_filter_empty_returns_none():
    assert _parse_provider_filter("") is None
    assert _parse_provider_filter(None) is None


# --- tf-leak root-cause verification (refactor headline test, spec §13) ---

def test_tf_not_dispatched_when_server_lacks_tf(monkeypatch, tmp_path):
    """A server that only detects ["torch"] must NOT have tf dispatched to it.

    Root cause of the pre-refactor tf leak: providers were resolved from yaml/spec
    without intersecting the server's actual detect set, so tf could be dispatched
    to an endpoint that lacks it. After the refactor, EndpointView intersects
    detect∩yaml∩alive before any dispatch decision.

    server detect=[torch] (no tf); spec third_party=[torch, tf].
    Expected: resolve_providers -> ["torch"] (tf filtered out),
              pick_endpoint("tf") -> None (tf never dispatched).
    """
    from ttk.utilities.singleton import Singleton
    Singleton._instances.clear()
    import ttk.config.loader as loader
    loader._config = None
    config_path = tmp_path / "ep.yaml"
    config_path.write_text(_endpoints_yaml([{"host": "10.0.0.1", "port": 9090}]))
    loader.load_config(str(config_path))
    hp = str(tmp_path / "h.json")
    _write_health(hp, {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}})
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", hp)
    ev = EndpointView()
    # spec wants torch+tf, but detect has no tf -> only torch survives (leak fixed)
    assert ev.resolve_providers(spec_providers=["torch", "tf"]) == ["torch"]
    assert ev.pick_endpoint("tf") is None  # tf is never dispatched
