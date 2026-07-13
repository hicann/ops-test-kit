"""Test that yaml config flows providers into endpoints (pure yaml, no --xpu override).

The --xpu CLI override mechanism was removed (Task 7). Remote config now comes
exclusively from yaml (default.yaml + ~/.config/ttk + ./ttk.conf.yaml + --config).
These tests verify that yaml providers flow into get_remote_config().endpoints
and that --provider lands on sw.provider_filter (a test filter, not a config
override).
"""
import os
from unittest.mock import MagicMock


def _make_args(**kwargs):
    """Create args mock with all required fields."""
    args = MagicMock()
    args.provider = kwargs.get('provider', None)
    args.run = 1
    args.output = None
    args.input = []
    args.log_level = 'INFO'
    args.compile_only = False
    args.no_compile = False
    args.no_golden = False
    args.no_compare = False
    args.dump_dir = None
    args.worker_count = 1
    args.proc_per_dev = None
    args.replay = None
    args.device_id = None
    args.tiling_key = None
    args.benchmark_iters = 10
    args.warmup_iters = 3
    args.sync_mode = 'auto'
    args.skip_unsupported = False
    args.case_filter = None
    return args


def test_yaml_providers_flow_into_endpoints(tmp_path, monkeypatch):
    """yaml endpoint providers flow through get_remote_config().endpoints[0].providers."""
    # yaml config: endpoint with providers=["tf"]
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  endpoints:
    - host: "10.0.0.1"
      port: 9090
      providers: ["tf"]
""")
    monkeypatch.chdir(tmp_path)

    import ttk.config.loader as loader
    loader._config = None

    from ttk.config.loader import load_config, get_remote_config
    load_config()  # picks up ./ttk.conf.yaml via standard search path

    config = get_remote_config()
    assert config is not None
    assert len(config.endpoints) == 1
    assert config.endpoints[0].host == "10.0.0.1"
    assert config.endpoints[0].port == 9090
    assert config.endpoints[0].providers == ["tf"]


def test_yaml_without_endpoints_uses_defaults(tmp_path, monkeypatch):
    """User yaml without endpoints: get_remote_config() returns None (no endpoints)."""
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  backoff_base_s: 1.0
""")
    monkeypatch.chdir(tmp_path)

    import ttk.config.loader as loader
    loader._config = None

    from ttk.config.loader import load_config, get_remote_config
    load_config()

    # No endpoints in yaml → get_remote_config returns None (per loader semantics).
    assert get_remote_config() is None


def test_provider_lands_on_switches_provider_filter(tmp_path, monkeypatch):
    """--provider CLI lands on sw.provider_filter (test filter), not on RemoteConfig.

    args_to_switches sets sw.provider_filter from args.provider; it does NOT
    patch RemoteConfig.provider (which doesn't exist as a field anyway).
    """
    user_config = tmp_path / "ttk.conf.yaml"
    user_config.write_text("""
remote:
  backoff_base_s: 1.0
""")
    monkeypatch.chdir(tmp_path)

    import ttk.config.loader as loader
    loader._config = None

    args = _make_args(provider="torch")
    # args_to_switches reads getattr(args, 'config', None) and getattr(args, 'provider', None)
    args.config = None
    args.input = "x.csv"
    args.output = None

    from ttk.cli.bridge import args_to_switches
    sw = args_to_switches(args)

    # --provider lands on sw.provider_filter (a test filter for workers)
    assert sw.provider_filter == "torch"
