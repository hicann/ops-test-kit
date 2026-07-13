"""_extract_spec_providers (dict/str/None) + EndpointView.resolve_providers
+ _do_xpu_profiling fail-loud (Task 9). No server fixture; EV constructed
in-process with a health file (ref Task 3 test pattern)."""
import json
import logging
import os
import types

import pytest

from ttk.core_modules.npu.op import profiling as prof


# ---- health-file + EndpointView construction (mirrors Task 3 test pattern) ----

def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _make_ev(monkeypatch, tmp_path, health_data, endpoints_list,
             health_name="h.json"):
    """Wire TTK_XPU_HEALTH_PATH env + write health file + load remote endpoints
    from yaml (load_config), then construct a FRESH EndpointView (Singleton cleared)."""
    from ttk.remote.endpoint_view import EndpointView
    from ttk.utilities.singleton import Singleton
    from ttk.config.loader import load_config
    Singleton._instances.clear()
    health_path = str(tmp_path / health_name)
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    # Remote config now comes from yaml (load_config), not TTK_XPU_ENDPOINTS env.
    lines = ["remote:", "  endpoints:"]
    for e in endpoints_list:
        lines.append(f"    - host: {e['host']}")
        lines.append(f"      port: {e['port']}")
        if e.get("providers"):
            lines.append("      providers: ["
                         + ", ".join(repr(p) for p in e["providers"]) + "]")
    yaml_path = tmp_path / "ttk.conf.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    load_config(str(yaml_path))
    _write_health(health_path, health_data)
    return EndpointView()


# ---- _extract_spec_providers ----

def test_extract_spec_providers_dict_returns_keys():
    tp = {"torch": "torch.add", "tf": "tf.raw_ops.Add"}
    assert prof._extract_spec_providers(tp) == ["torch", "tf"]


def test_extract_spec_providers_str_derives_single():
    # str -> single provider derived from API prefix (fallback "torch")
    derived = prof._extract_spec_providers("torch.add")
    assert isinstance(derived, list) and len(derived) == 1
    assert derived[0] in ("torch", "tf")     # _derive_provider_from_api result


def test_extract_spec_providers_none_is_empty():
    assert prof._extract_spec_providers(None) == []


def test_extract_spec_providers_empty_dict_is_empty():
    assert prof._extract_spec_providers({}) == []


# ---- resolve_providers via real EndpointView ----

def test_resolve_spec_intersects_effective(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"127.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "127.0.0.1", "port": 9090}])
    assert ev.resolve_providers(spec_providers=["torch"]) == ["torch"]


def test_resolve_cli_narrows_effective(monkeypatch, tmp_path):
    ev = _make_ev(monkeypatch, tmp_path,
                  {"127.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "127.0.0.1", "port": 9090}])
    assert ev.resolve_providers(cli_providers=["tf"]) == ["tf"]


# ---- CONTRACT CHANGE: cli as detect FILTER (not provider SOURCE) ----
# Old semantics "no-spec + cli -> cli is provider SOURCE" no longer holds;
# cli is now a filter: resolve_providers([], cli) = effective ∩ cli, so it is
# empty unless a health file has an alive EP detecting that provider.

def test_resolve_no_spec_cli_filter_needs_alive_ep_detecting_provider(monkeypatch, tmp_path):
    """no spec + cli='torch' -> effective(torch from alive EP) ∩ {torch} = ['torch'].

    Without a health file detecting torch this would raise; with one it returns ['torch'].
    This replaces the old test_resolve_no_spec_cli_provider_is_source semantics.
    """
    ev = _make_ev(monkeypatch, tmp_path,
                  {"127.0.0.1:9090": {"alive": True, "providers": ["torch", "tf"]}},
                  [{"host": "127.0.0.1", "port": 9090}])
    assert ev.resolve_providers(spec_providers=[], cli_providers=["torch"]) == ["torch"]


def test_resolve_no_spec_cli_filter_empty_when_no_alive_ep_detects(monkeypatch, tmp_path):
    """no spec + cli='torch' but only dead EPs -> raises (cli alone is NOT a source)."""
    ev = _make_ev(monkeypatch, tmp_path,
                  {"127.0.0.1:9090": {"alive": False, "providers": ["torch"]}},
                  [{"host": "127.0.0.1", "port": 9090}])
    with pytest.raises(RuntimeError):
        ev.resolve_providers(spec_providers=[], cli_providers=["torch"])


# ---- fail-loud: _do_xpu_profiling survives resolve failure ----

def test_do_xpu_profiling_resolve_failure_sets_empty_and_survives(monkeypatch, tmp_path, caplog):
    """resolve_providers RuntimeError -> context.xpu_results = {} + error log,
    no exception propagated (worker survives)."""
    from ttk.utilities.singleton import Singleton
    Singleton._instances.clear()
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", str(tmp_path / "h.json"))
    # Remote endpoints come from yaml (load_config), not TTK_XPU_ENDPOINTS env.
    # Load a yaml with the dead endpoint so the resolve-failure path is driven
    # by "alive=False" rather than "no endpoints configured".
    yaml_path = tmp_path / "ttk.conf.yaml"
    yaml_path.write_text(
        "remote:\n  endpoints:\n    - host: 127.0.0.1\n      port: 9090\n",
        encoding="utf-8")
    from ttk.config.loader import load_config
    load_config(str(yaml_path))
    _write_health(str(tmp_path / "h.json"),
                  {"127.0.0.1:9090": {"alive": False, "providers": ["torch"]}})

    # Stub OpInfoKeeper so the NPU-env-dependent lookups (ASCEND_OPP_PATH etc.)
    # don't blow up before we reach the resolve. fail-loud path must run in any env.
    class _FakeKeeper:
        def info_of(self, op_name):
            return None
        def op_type_of(self, op_name):
            return None
    monkeypatch.setattr(prof, "OpInfoKeeper", _FakeKeeper)

    # Minimal fake context — _do_xpu_profiling reads op_name, testcase_name,
    # input_arrays, attributes; resolve fails before any of the collector path.
    context = types.SimpleNamespace(
        op_name="add", testcase_name="add_case_0",
        input_arrays=(), attributes={})

    # OpInfoKeeper / TestSpecManager lookups must not break the fail-loud path;
    # the resolve failure happens after them, before collect_xpu_results.
    with caplog.at_level(logging.ERROR):
        priority = prof._do_xpu_profiling(context, xpu_mode=0b11)  # must NOT raise

    assert context.xpu_results == {}           # empty: no provider dispatched
    assert priority is None                    # resolve failed -> None priority
    assert any("XPU resolve failed" in r.message for r in caplog.records)
