import pytest
from ttk.config.loader import load_config
from ttk.remote.config import RemoteConfig, Endpoint


def _parse_from_yaml(yaml_path):
    cfg = load_config(yaml_path)
    remote = cfg.get("remote")
    return RemoteConfig.from_dict(remote) if remote else None


class TestEndpoint:
    def test_minimal_endpoint(self):
        ep = Endpoint(host="10.0.0.100", port=9090)
        assert ep.host == "10.0.0.100"
        assert ep.port == 9090
        assert ep.providers is None
        assert ep.hardware is None

    def test_full_endpoint(self):
        ep = Endpoint(host="10.0.0.101", port=9090,
                      providers=["tf"], hardware="gpu")
        assert ep.providers == ["tf"]
        assert ep.hardware == "gpu"


class TestYamlParsing:
    def test_parse_yaml(self, tmp_path):
        yaml_file = tmp_path / "ttk.conf.yaml"
        yaml_file.write_text("""
remote:
  endpoints:
    - host: 10.0.0.100
      port: 9091
    - host: 10.0.0.101
      port: 9090
      hardware: gpu
""")
        config = _parse_from_yaml(str(yaml_file))
        assert len(config.endpoints) == 2
        assert config.endpoints[1].hardware == "gpu"

    def test_missing_yaml(self):
        """YAML 路径不存在时，load_config 会 fallback 到 default.yaml。"""
        config = _parse_from_yaml("/nonexistent/ttk.conf.yaml")
        # default.yaml 有 remote 配置，所以不会是 None
        assert config is not None
        # 但应该使用 default.yaml 的默认值
        assert isinstance(config, RemoteConfig)

    def test_yaml_no_remote_section(self, tmp_path):
        """YAML 中无 remote section 时，load_config 会合并 default.yaml。"""
        yaml_file = tmp_path / "ttk.conf.yaml"
        yaml_file.write_text("other: value\n")
        config = _parse_from_yaml(str(yaml_file))
        # 即使指定的 yaml 没有 remote section，default.yaml 提供了默认 remote
        assert config is not None
        assert isinstance(config, RemoteConfig)

    def test_yaml_empty_endpoints(self, tmp_path):
        yaml_file = tmp_path / "ttk.conf.yaml"
        yaml_file.write_text("remote:\n  endpoints: []\n")
        config = _parse_from_yaml(str(yaml_file))
        assert config is not None
        assert config.endpoints == []


class TestNewConfigModule:
    """直接测试 ttk.config 模块（非 shim）。"""

    def test_default_yaml_is_parseable(self):
        """default.yaml 存在且可被 yaml 解析。"""
        from ttk.config.loader import _load_yaml, _default_config_path
        cfg = _load_yaml(_default_config_path())
        assert isinstance(cfg, dict)

    def test_deep_merge_dict_recursive(self):
        from ttk.config.loader import deep_merge

        base = {"remote": {"timeout": 30, "endpoints": []}}
        override = {"remote": {"timeout": 300}}
        result = deep_merge(base, override)
        assert result["remote"]["timeout"] == 300
        assert result["remote"]["endpoints"] == []  # 来自 base

    def test_deep_merge_list_replace(self):
        from ttk.config.loader import deep_merge

        base = {"remote": {"endpoints": [{"host": "a", "port": 1}]}}
        override = {"remote": {"endpoints": [{"host": "b", "port": 2}]}}
        result = deep_merge(base, override)
        # list 整体替换，不做并集
        assert result["remote"]["endpoints"] == [{"host": "b", "port": 2}]

    def test_deep_merge_new_key(self):
        from ttk.config.loader import deep_merge

        base = {"remote": {"timeout": 30}}
        override = {"remote": {"extra": "value"}}
        result = deep_merge(base, override)
        assert result["remote"]["timeout"] == 30
        assert result["remote"]["extra"] == "value"

    def test_load_config_returns_dict(self):
        from ttk.config.loader import load_config
        # 生产环境可能没有 ttk.conf.yaml，返回 default 或空
        cfg = load_config()
        assert isinstance(cfg, dict)

    def test_get_remote_config_no_config(self):
        """config 未加载（_config=None）时 get_remote_config 抛 RuntimeError。

        Task 3 删了 lazy fallback，无 config → raise。"无 config → raise" 的
        核心断言由 Task 3 `test_get_config_raises_before_load` 守；这里只验证
        get_remote_config 同样不静默回退。
        """
        import ttk.config.loader as loader
        import ttk.remote.config as remote_shim

        saved = loader._config
        loader._config = None
        try:
            with pytest.raises(RuntimeError):
                remote_shim.get_remote_config()
        finally:
            loader._config = saved
