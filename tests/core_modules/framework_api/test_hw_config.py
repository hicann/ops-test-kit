from __future__ import annotations

"""Task 4: hardware config in default.yaml + get_hardware_config + profile validation.

- frameworks.torch.npu segment ships in default.yaml (torch_lib=npu, profiler=builtin).
- get_hardware_config() returns {} when config empty (only-cpu legal default;
  distinct from remote's None).
- _build(fw, name, profile) fail-fast validates torch_lib + profiler before
  instantiating/injecting the backend.
"""
import pytest

from ttk.config.loader import load_config, get_hardware_config
from ttk.core_modules.framework_api.backends import _build


def test_hardware_config_returns_frameworks():
    load_config()
    hw = get_hardware_config()
    assert hw["torch"]["npu"]["torch_lib"] == "npu"


def test_get_hardware_config_empty_returns_dict(monkeypatch):
    monkeypatch.setattr("ttk.config.loader.get_config", lambda: {})
    assert get_hardware_config() == {}


def test_build_missing_torch_lib_raises():
    with pytest.raises(ValueError, match="torch_lib"):
        _build("torch", "npu", {"profiler": "builtin"})


def test_build_invalid_profiler_raises():
    with pytest.raises(ValueError, match="profiler"):
        _build("torch", "xpu", {"torch_lib": "cuda", "profiler": 123})  # 非 builtin/dict
