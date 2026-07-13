import pytest


def test_get_config_raises_before_load():
    """get_config() 在 load_config() 之前调 → raise RuntimeError（不再 lazy fallback）。"""
    import ttk.config.loader as loader
    saved = loader._config
    loader._config = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            loader.get_config()
    finally:
        loader._config = saved  # 恢复，别污染同 session 其它测试
