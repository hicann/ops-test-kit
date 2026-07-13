"""Tests for xpu_collector + _format_xpu_metrics + ev.pick_endpoint dispatch."""
import json
import os

import yaml


class TestFormatXpuMetrics:
    def test_empty_returns_empty_dict(self):
        from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
        assert _format_xpu_metrics(None) == {}
        assert _format_xpu_metrics({}) == {}

    def test_pass_entry(self):
        from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
        result = _format_xpu_metrics({
            "torch": {"status": "PASS", "api": "torch.add",
                      "outputs": [], "perf": {"device_us": 120.0,
                                              "peak_memory_mb": 8.5}},
        })
        assert result["torch"]["status"] == "PASS"
        assert result["torch"]["api"] == "torch.add"
        assert result["torch"]["device_us"] == 120.0
        assert result["torch"]["peak_memory_mb"] == 8.5
        assert "error" not in result["torch"]

    def test_fail_entry(self):
        from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
        result = _format_xpu_metrics({
            "tf": {"status": "FAIL", "api": "tf.raw.ops.Add",
                   "error": "import failed"},
        })
        assert result["tf"]["status"] == "FAIL"
        assert result["tf"]["error"] == "import failed"
        # FAIL 条目（无 perf）不应有 3 个 perf 列
        assert "device_us" not in result["tf"]
        assert "peak_memory_mb" not in result["tf"]

    def test_missing_status_defaults_to_fail(self):
        from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
        result = _format_xpu_metrics({"x": {}})
        assert result["x"]["status"] == "FAIL"

    def test_perf_none_not_included(self):
        from ttk.core_modules.npu.op.profiling_structure import _format_xpu_metrics
        result = _format_xpu_metrics({
            "torch": {"status": "PASS", "api": "torch.add", "perf": None},
        })
        assert "device_us" not in result["torch"]
        assert "peak_memory_mb" not in result["torch"]


def _write_health(path, endpoints_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"endpoints": endpoints_data}, f)


def _make_ev(monkeypatch, tmp_path, health_data, endpoints_list,
             clear_singleton=True):
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
    config_path.write_text(yaml.safe_dump({"remote": {"endpoints": endpoints_list}}))
    loader.load_config(str(config_path))
    health_path = str(tmp_path / "h.json")
    monkeypatch.setenv("TTK_XPU_HEALTH_PATH", health_path)
    _write_health(health_path, health_data)
    from ttk.remote.endpoint_view import EndpointView
    return EndpointView()


class TestPickEndpointEV:
    """collect_xpu_results now routes endpoint selection through
    EndpointView.pick_endpoint (the single decision point)."""

    def test_select_by_provider(self, monkeypatch, tmp_path):
        ev = _make_ev(monkeypatch, tmp_path,
                      {"10.0.0.1:9090": {"alive": True, "providers": ["tf"]},
                       "10.0.0.2:9090": {"alive": True, "providers": ["torch"]}},
                      [{"host": "10.0.0.1", "port": 9090},
                       {"host": "10.0.0.2", "port": 9090}])
        ep = ev.pick_endpoint("torch")
        assert ep.host == "10.0.0.2"

    def test_none_when_provider_absent(self, monkeypatch, tmp_path):
        ev = _make_ev(monkeypatch, tmp_path,
                      {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]}},
                      [{"host": "10.0.0.1", "port": 9090}])
        assert ev.pick_endpoint("tf") is None

    def test_none_when_all_dead(self, monkeypatch, tmp_path):
        ev = _make_ev(monkeypatch, tmp_path,
                      {"10.0.0.1:9090": {"alive": False, "providers": ["torch"]}},
                      [{"host": "10.0.0.1", "port": 9090}])
        assert ev.pick_endpoint("torch") is None

    def test_round_robin_load_balancing(self, monkeypatch, tmp_path):
        ev = _make_ev(monkeypatch, tmp_path,
                      {"10.0.0.1:9090": {"alive": True, "providers": ["torch"]},
                       "10.0.0.2:9090": {"alive": True, "providers": ["torch"]}},
                      [{"host": "10.0.0.1", "port": 9090},
                       {"host": "10.0.0.2", "port": 9090}])
        ep1 = ev.pick_endpoint("torch")
        ep2 = ev.pick_endpoint("torch")
        assert ep1 is not ep2


class TestSelectRunSpecs:
    """§③ mode-driven count dispatch + per-provider mode.

    Availability filtering moved upstream to EndpointView.resolve_providers
    (profiling._do_xpu_profiling); all incoming specs are already resolvable,
    so _select_run_specs no longer takes a has_endpoint predicate.
    """

    def _spec(self, provider):
        from ttk.remote import ExecutionSpec
        return ExecutionSpec(provider=provider, type="api", api=f"torch.{provider}")

    def test_data_mode_picks_priority_only(self):
        from ttk.remote import DATA
        from ttk.remote.xpu_collector import _select_run_specs
        specs = [self._spec("torch"), self._spec("tf")]  # torch = priority (order)
        run, priority = _select_run_specs(specs, DATA)
        assert [s.provider for s in run] == ["torch"]
        assert priority.provider == "torch"

    def test_perf_mode_runs_all(self):
        from ttk.remote import PERF
        from ttk.remote.xpu_collector import _select_run_specs
        specs = [self._spec("torch"), self._spec("tf")]
        run, priority = _select_run_specs(specs, PERF)
        assert [s.provider for s in run] == ["torch", "tf"]

    def test_dataperf_runs_all_priority_gets_data(self):
        from ttk.remote import DATA, PERF
        from ttk.remote.xpu_collector import _select_run_specs, _per_spec_mode
        specs = [self._spec("torch"), self._spec("tf")]
        run, priority = _select_run_specs(specs, DATA | PERF)
        assert [s.provider for s in run] == ["torch", "tf"]
        assert _per_spec_mode(specs[0], priority, DATA | PERF) == (DATA | PERF)  # priority
        assert _per_spec_mode(specs[1], priority, DATA | PERF) == PERF            # non-priority

    def test_empty_specs(self):
        from ttk.remote import DATA, PERF
        from ttk.remote.xpu_collector import _select_run_specs
        run, priority = _select_run_specs([], DATA)
        assert run == []
        assert priority is None
        run, priority = _select_run_specs([], PERF)
        assert run == []
        assert priority is None
