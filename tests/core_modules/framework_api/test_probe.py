from __future__ import annotations

"""Task 5: _probe (hardware detection D-scheme) + get_backend(force_cpu).

_probe: cuda lib skips import_module; non-cuda import_module + getattr
is_available; whole body wrapped in try/except Exception (covers
ImportError / RuntimeError / AttributeError) returning False + warning.
get_backend(force_cpu=True) -> CpuTorchBackend; else iterate _hw_profiles
in order, _probe each non-cpu profile, build first hit; cpu fallback.
"""
import importlib

from ttk.core_modules.framework_api.backends import get_backend, _probe
from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend


def test_probe_cuda_skips_import(monkeypatch):
    """cuda native skips import: importlib.import_module must not be called."""
    called = []

    def spy(name):
        called.append(name)
        raise AssertionError("should not import for cuda")

    monkeypatch.setattr(importlib, "import_module", spy)
    _probe({"torch_lib": "cuda"})  # no raise + import not called
    assert called == []


def test_probe_missing_torch_lib_returns_false(caplog):
    """A profile missing torch_lib is rejected up front (not swallowed as
    "device not available"). Both empty {} and {profiler: ...} -> False +
    warning naming torch_lib."""
    for bad in ({}, {"profiler": "builtin"}):
        caplog.clear()
        assert _probe(bad) is False
        assert any("torch_lib" in rec.message for rec in caplog.records)


def test_force_cpu_returns_cpu_backend():
    assert isinstance(get_backend(force_cpu=True), CpuTorchBackend)


def test_probe_catches_runtime_error(monkeypatch):
    """PrivateUse1 single-slot RuntimeError caught by except."""

    def boom(name):
        raise RuntimeError("rename_privateuse1_backend already set")

    monkeypatch.setattr(importlib, "import_module", boom)
    assert _probe({"torch_lib": "mlu"}) is False


def test_auto_detect_falls_back_to_cpu(monkeypatch):
    """All miss -> cpu fallback."""
    monkeypatch.setattr("ttk.core_modules.framework_api.backends._hw_profiles", lambda fw: {})
    assert isinstance(get_backend(force_cpu=False), CpuTorchBackend)
