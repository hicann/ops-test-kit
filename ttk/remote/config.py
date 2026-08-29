"""
向后兼容 shim — 所有配置相关功能已移至 ttk.config。

新代码请使用:
    from ttk.config.loader import Endpoint, RemoteConfig, get_remote_config, get_config
"""

from ttk.config.loader import (  # noqa: F401  re-export
    Endpoint as Endpoint,
)
from ttk.config.loader import (
    RemoteConfig as RemoteConfig,
)
from ttk.config.loader import (
    get_config as get_config,
)
from ttk.config.loader import (
    get_remote_config as get_remote_config,
)
from ttk.config.loader import (
    load_config as load_config,
)
