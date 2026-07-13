"""E2E: KERNEL → XPU dispatch chain against a local CPU XPU-Server stub.

Tests the FULL dispatch chain via direct function calls (no subprocess, no NPU):
  _extract_spec_providers → EndpointView.resolve_providers → _build_spec
  → collect_xpu_results → dispatch_to_remote → standalone server → executor
  → _resolve_3party_api / spec-mode → result.

Covers:
  - No spec (discovery): server-advertised providers (torch+tf).
  - No spec + --provider torch: narrowed to torch only.
  - Spec third_party = "torch.add" (single API string).
  - Spec third_party = {"torch": ..., "tf": ...} (dict of API strings).
  - Spec third_party = {"torch": AddImpl} (dict of impl class, spec mode).
  - Spec third_party = None (spec exists, no third_party → fallback resolve).

No NPU needed — this tests the XPU dispatch chain, not the kernel compile.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

from ttk.core_modules.npu.op.profiling import _build_spec, _extract_spec_providers
from ttk.remote import DATA, PERF
from ttk.remote.endpoint_view import EndpointView
from ttk.remote.xpu_collector import collect_xpu_results
from ttk.test_spec import get_spec_attr, get_spec_class_meta
from ttk.utilities.singleton import Singleton

# Dynamic capability check (find_spec = no import, no TF flood).
# Run partial E2E based on what's installed; CI/CD (has both) runs full guard.
import importlib.util as _ilu
has_torch = _ilu.find_spec("torch") is not None
has_tf = _ilu.find_spec("tensorflow") is not None
if has_tf:
    # find_spec 只看包存在；CI 的 tf 可能装了但 import 崩溃（protobuf 不兼容 / C 扩展 segfault）
    try:
        has_tf = subprocess.run([sys.executable, "-c", "import tensorflow"],
                                capture_output=True, timeout=90).returncode == 0
    except Exception:
        has_tf = False

needs_torch = pytest.mark.skipif(not has_torch, reason="torch not installed")
needs_tf = pytest.mark.skipif(not has_tf, reason="tensorflow not installed")
needs_both = pytest.mark.skipif(not (has_torch and has_tf), reason="needs both torch+tf")

pytestmark = pytest.mark.e2e

_SPEC_DIR = os.path.join(os.path.dirname(__file__), "specs")
_TENANT_ID = "e2e_xpu"


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(scope="module")
def xpu_server():
    """Start a standalone XPU-Server (CPU stub, same as production deploy)."""
    port = _free_port()
    env = dict(os.environ)
    ttk_remote = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ttk", "remote"))
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (ttk_remote, env.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen(
        [sys.executable, "-m", "server.xpu_server",
         "--port", str(port), "--devices", "cpu"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Task 6 removed /health; readiness now probes GET /v1/heartbeat (merged
    # health+detect+register). tenant_id=e2e_xpu matches collect_xpu_results.
    for _ in range(20):
        time.sleep(0.5)
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            c.request("GET", f"/v1/heartbeat?tenant_id={_TENANT_ID}")
            c.getresponse().read()
            c.close()
            break
        except (ConnectionRefusedError, OSError):
            continue
    yield port
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _probe_heartbeat(xpu_port):
    """GET /v1/heartbeat?tenant_id=e2e_xpu once: registers tenant + returns
    {providers, hardware}. Used to seed the health file for EndpointView."""
    c = http.client.HTTPConnection("127.0.0.1", xpu_port, timeout=5)
    c.request("GET", f"/v1/heartbeat?tenant_id={_TENANT_ID}")
    resp = c.getresponse()
    body = json.loads(resp.read())
    c.close()
    return body


def _dispatch(xpu_port, tmp_path, monkeypatch, spec_subdir=None, cli_providers=None):
    """Build specs (via real EndpointView resolution) + dispatch to server.

    Returns (results_dict, inputs_tuple).

    `_dispatch` is a plain helper (no pytest fixtures of its own): tmp_path +
    monkeypatch are passed in by the 6 caller tests so the health-file path +
    env wiring are auto-cleaned per test.
    """
    # 1. Register tenant + discover providers via /v1/heartbeat (must match
    #    collect_xpu_results tenant_id below so /v1/run finds the tenant).
    body = _probe_heartbeat(xpu_port)
    providers = body.get("providers", [])
    hardware = body.get("hardware", "")

    # 2. Write health file: alive EP detecting the server's providers.
    health_path = str(tmp_path / "xpu_health.json")
    ep_key = f"127.0.0.1:{xpu_port}"
    os.makedirs(os.path.dirname(health_path), exist_ok=True)
    with open(health_path, "w") as f:
        json.dump({"endpoints": {
            ep_key: {"alive": True, "providers": providers,
                     "hardware": hardware, "ts": time.time()}}}, f)

    # 3. Wire env + RemoteConfig + clear Singleton EVERY call so the
    #    EndpointView constructed inside collect_xpu_results binds to THIS
    #    test's health file + endpoint (not a stale prior Singleton).
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    # Remote config now comes from yaml (load_config), not env vars or
    # set_remote_config override (both removed in Task 6/7). Write a temp yaml
    # with this endpoint and load it so get_remote_config() sees it.
    yaml_path = tmp_path / "ttk.conf.yaml"
    yaml_path.write_text(
        f"remote:\n  endpoints:\n    - host: 127.0.0.1\n      port: {xpu_port}\n")
    import ttk.config.loader as loader
    from ttk.config.loader import load_config
    load_config(str(yaml_path))
    # cli_providers is passed directly to resolve_providers below (and into
    # collect_xpu_results); TTK_XPU_PROVIDER env is no longer read by production
    # (provider filter now lives in SWITCHES.provider_filter), so no env wiring.
    Singleton._instances.clear()

    # Load spec (if spec_subdir given)
    search = (os.path.join(_SPEC_DIR, spec_subdir),) if spec_subdir else ()
    tp = get_spec_attr("add", "third_party", search) if search else None
    meta = get_spec_class_meta("add", search) if search else None
    spec_file = meta["spec_file"] if meta else None
    spec_class = meta["class_name"] if meta else None

    # 4. Resolve providers via real EndpointView (detect∩yaml∩alive ∩ spec ∩ cli).
    #    With a spec, resolve_providers returns survivors in spec insertion order
    #    (app-declared priority, spec §③); priority = resolved[0]. Without a spec
    #    it returns sorted survivors (deterministic neutral).
    ev = EndpointView()
    resolved = ev.resolve_providers(_extract_spec_providers(tp), cli_providers)

    # 5. Build specs + dispatch.
    specs = [_build_spec(p, tp, spec_file, spec_class, "add", "Add") for p in resolved]

    x = np.random.randn(4, 8).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)
    results = collect_xpu_results(
        specs, inputs=[x, y], input_names=["x", "y"],
        mode=DATA | PERF, tenant_id=_TENANT_ID,
        op_name="add", op_type="Add",
    )
    # priority = first resolved provider (spec order when a spec is given); DATA goes to it.
    return results, (x, y), (resolved[0] if resolved else None)


def _assert_pass(results, provider):
    assert provider in results, f"{provider} missing from results {list(results)}"
    assert results[provider]["status"] == "PASS", f"{provider}: {results[provider]}"


def _assert_priority_output(results, x, y, priority):
    """Priority provider (specs[0]) gets DATA in DATA|PERF → verify its output.

    resolve_providers returns survivors in spec insertion order when a spec is
    given (app-declared priority, spec §③), or sorted when no spec. priority is
    resolved[0] in either case; we assert on the actual priority, not a hard-
    coded provider name.
    """
    assert priority is not None, "no provider resolved (priority unknown)"
    outs = results[priority].get("outputs")
    assert outs, f"priority {priority} has no outputs (expected DATA for priority)"
    np.testing.assert_allclose(outs[0], np.add(x, y), rtol=1e-5,
                                err_msg=f"priority {priority} output mismatch")


# ---- No-spec scenarios ----

@needs_both
def test_no_spec_discovery_both_providers(xpu_server, tmp_path, monkeypatch):
    """No spec, no --provider: both torch+tf dispatched + priority output verified."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch)
    _assert_pass(results, "torch")
    _assert_pass(results, "tf")
    _assert_priority_output(results, x, y, priority)


@needs_torch
def test_no_spec_provider_torch_only(xpu_server, tmp_path, monkeypatch):
    """--provider torch on no-spec op: only torch dispatched (torch is priority)."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          cli_providers=["torch"])
    _assert_pass(results, "torch")
    assert "tf" not in results, f"tf should be absent (--provider torch): {list(results)}"
    _assert_priority_output(results, x, y, priority)


@needs_tf
def test_no_spec_provider_tf_only(xpu_server, tmp_path, monkeypatch):
    """--provider tf on no-spec op: only tf dispatched (tf is priority)."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          cli_providers=["tf"])
    _assert_pass(results, "tf")
    assert "torch" not in results, f"torch should be absent (--provider tf): {list(results)}"
    _assert_priority_output(results, x, y, priority)


# ---- Spec scenarios: all third_party styles ----

@needs_torch
def test_spec_tp_str(xpu_server, tmp_path, monkeypatch):
    """third_party = 'torch.add' (single API string); torch is priority."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_str")
    _assert_pass(results, "torch")
    _assert_priority_output(results, x, y, priority)


@needs_both
def test_spec_tp_dict_api(xpu_server, tmp_path, monkeypatch):
    """third_party = {'torch': '...', 'tf': '...'} (dict of API strings)."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_dict_api")
    _assert_pass(results, "torch")
    _assert_pass(results, "tf")
    _assert_priority_output(results, x, y, priority)


@needs_torch
def test_spec_tp_dict_impl(xpu_server, tmp_path, monkeypatch):
    """third_party = {'torch': AddImpl} (dict of impl class, spec mode)."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_dict_impl")
    _assert_pass(results, "torch")
    _assert_priority_output(results, x, y, priority)


@needs_both
def test_spec_tp_none_fallback(xpu_server, tmp_path, monkeypatch):
    """third_party = None: spec exists but no third_party → fallback resolve."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_none")
    _assert_pass(results, "torch")
    _assert_pass(results, "tf")
    _assert_priority_output(results, x, y, priority)


# ---- TF-specific E2E guards ----

@needs_both
def test_spec_tp_tensorflow_key(xpu_server, tmp_path, monkeypatch):
    """third_party = {'tensorflow': ..., 'torch': ...} → tensorflow 归一化为 tf."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_tensorflow_key")
    _assert_pass(results, "torch")
    _assert_pass(results, "tf")
    _assert_priority_output(results, x, y, priority)


@needs_tf
def test_spec_tp_impl_tf(xpu_server, tmp_path, monkeypatch):
    """third_party = {'tf': AddTfImpl} (TF spec mode)."""
    results, (x, y), priority = _dispatch(xpu_server, tmp_path, monkeypatch,
                                          spec_subdir="tp_impl_tf")
    _assert_pass(results, "tf")
    _assert_priority_output(results, x, y, priority)


@needs_both
def test_hardware_profile_penetration(xpu_server, tmp_path, monkeypatch):
    """heartbeat hardware field flows through health file → EndpointView."""
    body = _probe_heartbeat(xpu_server)
    assert "hardware" in body, f"heartbeat missing hardware: {body}"
    assert body["hardware"] == "cpu", f"expected cpu (no device), got {body['hardware']}"
    assert "providers" in body, f"heartbeat missing providers: {body}"
    assert "torch" in body["providers"], f"torch not in providers {body['providers']}"
