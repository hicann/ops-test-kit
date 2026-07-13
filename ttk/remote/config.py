"""
向后兼容 shim — 所有配置相关功能已移至 ttk.config。

新代码请使用:
    from ttk.config.loader import Endpoint, RemoteConfig, get_remote_config, get_config
"""
from ttk.config.loader import (          # re-export
    Endpoint,
    RemoteConfig,
    get_config,
    load_config,
    get_remote_config,
)

