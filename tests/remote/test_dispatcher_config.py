"""Test that dispatcher reads backoff/retry config from RemoteConfig dataclass."""

from ttk.config import loader as loader


def _reset_and_load(yaml_text, tmp_path):
    """Reset loader._config then load_config from a yaml file written to tmp_path."""
    p = tmp_path / "ttk.conf.yaml"
    p.write_text(yaml_text)
    loader._config = None
    return loader.load_config(str(p))


def test_dispatcher_reads_backoff_from_config(tmp_path):
    """Dispatcher should read backoff params from RemoteConfig via _cfg() helper."""
    _reset_and_load(
        """
remote:
  endpoints:
    - {host: "127.0.0.1", port: 9090}
  backoff_base_s: 2.0
  backoff_max_s: 60.0
  backoff_jitter: 0.3
""",
        tmp_path,
    )
    from ttk.remote.dispatcher import _cfg

    assert _cfg('backoff_base_s', 0.5) == 2.0
    assert _cfg('backoff_max_s', 10.0) == 60.0
    assert _cfg('backoff_jitter', 0.25) == 0.3


def test_dispatcher_reads_retry_limits_from_config(tmp_path):
    """Dispatcher should read retry limits from RemoteConfig via _cfg() helper."""
    _reset_and_load(
        """
remote:
  endpoints:
    - {host: "127.0.0.1", port: 9090}
  max_503_retries: 20
  max_conn_retries: 15
  dispatch_deadline_s: 600
""",
        tmp_path,
    )
    from ttk.remote.dispatcher import _cfg

    assert _cfg('max_503_retries', 10.0, int) == 20
    assert _cfg('max_conn_retries', 5.0, int) == 15
    assert _cfg('dispatch_deadline_s', 300.0, int) == 600


def test_dispatcher_uses_code_defaults_when_no_config(tmp_path):
    """Dispatcher should fall back to code defaults when no RemoteConfig is set.

    Loading a yaml without endpoints => get_remote_config() returns None =>
    _cfg falls back to its per-call default arg.
    """
    # No remote.endpoints => get_remote_config() returns None (no config set)
    _reset_and_load("foo: bar\n", tmp_path)
    from ttk.remote.dispatcher import _cfg

    assert _cfg('backoff_base_s', 0.5) == 0.5
    assert _cfg('backoff_max_s', 10.0) == 10.0
    assert _cfg('backoff_jitter', 0.25) == 0.25
    assert _cfg('max_503_retries', 10.0, int) == 10
    assert _cfg('max_conn_retries', 5.0, int) == 5
    assert _cfg('dispatch_deadline_s', 300.0, int) == 300
