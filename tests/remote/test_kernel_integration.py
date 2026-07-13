"""Tests for KERNEL profile_process XPU integration: ev.pick_endpoint dispatch."""
import json
import os
import pytest


def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _make_ev(monkeypatch, health_path, health_data, endpoints_list,
             clear_singleton=True):
    """Wire health-path env + write health file + load remote endpoints from yaml
    (load_config), then construct a FRESH EndpointView (Singleton cleared).

    Remote endpoints come from yaml (load_config), not TTK_XPU_ENDPOINTS env
    (env-based config removed in config-loading consolidation).
    """
    if clear_singleton:
        from ttk.utilities.singleton import Singleton
        Singleton._instances.clear()
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    # Remote config now comes from yaml (load_config), not TTK_XPU_ENDPOINTS env.
    lines = ["remote:", "  endpoints:"]
    for e in endpoints_list:
        lines.append(f"    - host: {e['host']}")
        lines.append(f"      port: {e['port']}")
        if e.get("providers"):
            lines.append("      providers: ["
                         + ", ".join(repr(p) for p in e["providers"]) + "]")
    yaml_path = os.path.join(os.path.dirname(health_path), "ttk.conf.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    from ttk.config.loader import load_config
    load_config(yaml_path)
    _write_health(health_path, health_data)
    from ttk.remote.endpoint_view import EndpointView
    return EndpointView()


class TestPickEndpoint:
    def test_fallback_first_when_no_yaml_providers(self, monkeypatch, tmp_path):
        # No yaml providers -> detect∩yaml == detect; first alive endpoint wins.
        ev = _make_ev(monkeypatch, str(tmp_path / "h.json"),
                      {"10.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                      [{"host": "10.0.0.1", "port": 9090}])
        ep = ev.pick_endpoint("torch")
        assert ep.host == "10.0.0.1"
