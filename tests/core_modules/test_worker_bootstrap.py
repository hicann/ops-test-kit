import pytest


def test_worker_bootstrap_loads_config_from_switches(tmp_path, monkeypatch):
    """worker_bootstrap 从 SWITCHES.config_path 加载 yaml（含 TLS）——根因测试。"""
    import ttk.config.loader as loader
    from ttk.utilities.classes import SWITCHES
    # 桩掉 default_logging_config：它有全局副作用（清 root logger handler +
    # numpy.seterr('ignore') + warnings.filterwarnings），会污染同 session 其它
    # 测试。worker_bootstrap 的核心是 load_config，logging 非本测试关注点。
    import ttk.core_modules.tbe_multiprocessing.pool as pool
    monkeypatch.setattr(pool, "default_logging_config", lambda **kw: None)

    yaml_path = tmp_path / "worker-test.yaml"
    yaml_path.write_text(
        'remote:\n'
        '  tls_ca: "/fake/ca.crt"\n'
        '  endpoints:\n'
        '    - host: 10.0.0.1\n'
        '      port: 9090\n'
    )
    saved = loader._config
    loader._config = None  # 模拟 forkserver worker：干净的 _config
    try:
        sw = SWITCHES()
        sw.config_path = str(yaml_path)
        sw.logging_to_file = False

        from ttk.core_modules.tbe_multiprocessing.pool import worker_bootstrap
        worker_bootstrap(sw)

        rc = loader.get_remote_config()
        assert rc is not None
        assert rc.tls_ca == "/fake/ca.crt"
        assert len(rc.endpoints) == 1
    finally:
        loader._config = saved
        loader.load_config()  # 恢复默认 config，别污染后续测试


def test_worker_bootstrap_without_config_path_loads_defaults(monkeypatch):
    """config_path=None 时 worker_bootstrap 走标准路径（不 crash）。"""
    import ttk.config.loader as loader
    from ttk.utilities.classes import SWITCHES
    import ttk.core_modules.tbe_multiprocessing.pool as pool
    monkeypatch.setattr(pool, "default_logging_config", lambda **kw: None)
    saved = loader._config
    loader._config = None
    try:
        sw = SWITCHES()
        sw.config_path = None
        sw.logging_to_file = False
        from ttk.core_modules.tbe_multiprocessing.pool import worker_bootstrap
        worker_bootstrap(sw)
        cfg = loader.get_config()
        assert "remote" in cfg  # default.yaml 有 remote 段
    finally:
        loader._config = saved
        loader.load_config()
