# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Server config loader: yaml -> built-in defaults.

Deployment constraint: stdlib + PyYAML only. No ttk.* imports.
"""

import logging
import os
import re


def _load_yaml(path):
    try:
        import yaml  # lazy（与现状一致，防 PyYAML 未装）

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}  # 首次部署无 yaml 合法，静默（启动期）
    except Exception as e:
        logging.warning(f"_load_yaml: parse failed {path}: {e}")  # yaml 语法错/权限
        return {}


def detect_hardware(hw_config):
    """探测硬件。返回 (role, [ids])——role=段名，[ids]=物理节点 id 列表
    (int，regex 提 /dev/{prefix}{N} 的数字，sorted(set)，稀疏保留如 [0,5])。
    只 /dev 探测（快速，不 import torch/tf）。torch/tf 可用性由 executor lazy
    import 验证（不预探，崩则异常响应）。全未命中（/dev 无设备）→ cpu 兜底。
    profile 由 handler 用 role 查 hardware_config[role] 得（cpu 兜底 {}）。
    """
    try:
        dev_names = os.listdir("/dev")
    except OSError as e:
        logging.warning(f"detect_hardware: /dev not readable: {e}")
        dev_names = []
    for role, profile in hw_config.items():
        prefix = profile.get("dev_prefix")
        if prefix:
            ids = _scan_dev_ids(prefix, dev_names)
            if ids:
                return role, ids
    logging.warning("detect_hardware: no segment matched /dev; falling back to cpu (check dev_prefix in yaml)")
    return "cpu", ["cpu"]


def _scan_dev_ids(prefix, dev_names):
    """提取 /dev/{prefix}{N} 物理节点 id（int），sorted(set)；控制设备（无数字尾）不匹配。"""
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    ids = []
    for n in dev_names:
        m = pat.match(n)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def get_framework(provider, provider_framework=None):
    """Derive framework from provider. Default: framework = provider.

    Args:
        provider: e.g. "torch", "tf", "flash_attn"
        provider_framework: dict of overrides, e.g. {"flash_attn": "torch"}

    Returns:
        framework string
    """
    if provider_framework is None:
        provider_framework = {}
    return provider_framework.get(provider, provider)


def load_server_config(yaml_path=None):
    """Load xpu_server config from yaml with built-in defaults.

    Args:
        yaml_path: Path to xpu_server.yaml. None = look next to this file.

    Returns:
        dict with all config values resolved.
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(__file__), "xpu_server.yaml")

    yaml_cfg = _load_yaml(yaml_path)

    def _get(path, default):
        d = yaml_cfg
        for k in path:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return default
        return d if d not in ("", {}, None) else default

    cfg = {
        "bind": _get(("server", "bind"), "127.0.0.1"),
        "port": _get(("server", "port"), 9090),
        "max_concurrent": _get(("server", "max_concurrent"), 16),
        "run_deadline_s": _get(("server", "run_deadline_s"), 300),
        "sandbox": _get(("execution", "sandbox"), "none"),
        "gate_wait_s": _get(("execution", "gate_wait_s"), 1.0),
        "device_unhealthy_detect_interval_s": _get(("execution", "device_unhealthy_detect_interval_s"), 60.0),
        "providers": _get(("providers",), None),
        "provider_framework": _get(("provider_framework",), {}),
        "sync_dir": _get(("storage", "sync_dir"), os.path.join(os.path.dirname(__file__), "ttk_xpu_sync")),
        "tmp_dir": _get(("storage", "tmp_dir"), os.path.join(os.path.dirname(__file__), "ttk_tmp_dir")),
        "tls_enabled": _get(("tls", "enabled"), False),
        "tls_ca_cert": _get(("tls", "ca_cert"), ""),
        "tls_server_cert": _get(("tls", "server_cert"), ""),
        "tls_server_key": _get(("tls", "server_key"), ""),
        "docker_images": _get(("docker", "images"), {}),
        "docker_memory": _get(("docker", "memory"), "8g"),
        "docker_network": _get(("docker", "network"), "none"),
    }

    hw = _get(("hardware",), {})
    if not isinstance(hw, dict):
        raise ValueError(f"hardware must be a mapping, got {type(hw).__name__}")
    for name, seg in hw.items():
        if name != name.lower():
            raise ValueError(f"hardware segment name not lowercase: {name}")
        if not isinstance(seg, dict):
            raise ValueError(f"hardware segment '{name}' must be a mapping")
        for f in ("dev_prefix", "torch_lib", "torch_profiler"):
            if not seg.get(f):
                raise ValueError(f"hardware segment '{name}' missing required field: {f}")
    cfg["hardware_config"] = hw

    return cfg
