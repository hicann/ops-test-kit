"""
TTK 远端执行 —— 跨机器 XPU 算子执行。

Public API:
    get_tenant_id() — 获取当前 TTK 实例的租户 ID
    is_remote_configured() — 检查是否配置了远端执行
    dispatch_to_remote() — 发送输入到远端 xpu_server，返回输出
"""
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

# --- X-Mode bitmask. Shared protocol between client (dispatcher)
# and server (xpu_server / executor). Keep these bit-stable across versions.
DATA = 0b01  # return remote xpu outputs
PERF = 0b10  # collect performance


def has_data(mode: int) -> bool:
    return bool(mode & DATA)


def has_perf(mode: int) -> bool:
    return bool(mode & PERF)


# 实例级 tenant_id，fork 前生成，子进程继承同一值
_TENANT_ID: str = uuid.uuid4().hex[:12]


def get_tenant_id() -> str:
    return _TENANT_ID


def is_remote_configured() -> bool:
    """Check if remote execution is configured (has endpoints in yaml).

    This answers "is it configured", not "is the server reachable".
    删了 TTK_XPU_ENDPOINTS env 检查——endpoints 现从 yaml + get_remote_config() 来。
    """
    try:
        from ttk.remote.config import get_remote_config
        config = get_remote_config()
        return config is not None and bool(config.endpoints)
    except RuntimeError:
        # 防御性兜底：config 未加载时返回 False（而非崩）
        return False


@dataclass
class ExecutionSpec:
    """Per-provider remote execution specification.

    type == 'api':      call ``api`` (a dotted callable string) on the server.
    type == 'spec':  sync ``spec_file`` and invoke ``spec_class``
                        (whose ``third_party[provider]`` is the impl) on the server.
    """
    provider: str
    type: str = "api"
    api: Optional[str] = None
    spec_module: Optional[str] = None
    spec_file: Optional[str] = None
    spec_class: Optional[str] = None


def _derive_provider_from_api(api: Optional[str], fallback: str) -> str:
    """Derive provider from API module prefix ('torch.add' → 'torch').

    Known frameworks (torch/tf/tensorflow/numpy/np) are normalized:
    tensorflow→tf, np→numpy. Unknown prefixes pass through verbatim —
    resolve_providers filters what the server doesn't detect.
    """
    if not api or "." not in api:
        return fallback
    prefix = api.split(".", 1)[0]
    return {"tensorflow": "tf", "np": "numpy"}.get(prefix, prefix)
