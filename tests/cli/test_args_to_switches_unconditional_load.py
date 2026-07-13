import types


def test_args_to_switches_unconditional_load():
    """不传 --config 时 args_to_switches 仍无条件 load_config（删 lazy fallback 的硬约束）。"""
    import ttk.config.loader as loader
    from ttk.cli.bridge import args_to_switches
    saved = loader._config
    loader._config = None
    try:
        args = types.SimpleNamespace(config=None, provider=None, input="x.csv", output=None)
        sw = args_to_switches(args)         # config=None
        assert loader._config is not None   # 仍无条件加载（走标准路径）
        assert sw.config_path is None       # CLI 值正确落入 SWITCHES
        assert sw.provider_filter is None
    finally:
        loader._config = saved
