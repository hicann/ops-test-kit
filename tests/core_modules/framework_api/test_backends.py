from __future__ import annotations

"""Backend ABC + TorchBackend + three hardware backends.

Task 7: device_name() returns the hardware MODEL (torch.<lib>.get_device_name);
alias() carries the segment name; soc_version removed (merged into device_name).
"""
from ttk.core_modules.framework_api.backends.base import Backend
from ttk.core_modules.framework_api.backends.torch_backend import TorchBackend
from ttk.core_modules.framework_api.backends.npu_torch_backend import NpuTorchBackend
from ttk.core_modules.framework_api.backends.xpu_torch_backend import XpuTorchBackend
from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend


def test_backend_abc_has_new_methods():
    assert all(hasattr(Backend, m) for m in ["device_name", "device_type", "has_device", "is_npu"])


def test_torchbackend_is_available_via_getattr(monkeypatch):
    class _FakeMod:
        def is_available(self):
            return True

        def device_count(self):
            return 1

    import torch

    monkeypatch.setattr(torch, "cuda", _FakeMod(), raising=False)
    tb = TorchBackend()
    tb.torch_lib = "cuda"
    tb.profile = {}
    assert tb.is_available() is True
    assert tb.device_count() == 1


def test_torchbackend_device_type_default_is_segment_name():
    """device_type() returns _segment_name (config-driven); empty until injected.
    TorchBackend itself is never built by _build, so its _segment_name stays ''
    until a test (or _build) sets it."""
    tb = TorchBackend()
    assert tb.device_type() == ""


def test_torchbackend_device_name_is_model_via_get_device_name(monkeypatch):
    """Task 7: device_name() returns the hardware MODEL via
    torch.<torch_lib>.get_device_name (no longer the torch_lib segment)."""
    import torch

    class _FakeCuda:
        @staticmethod
        def get_device_name(dev_id):
            return "FakeGPU-Model"

    monkeypatch.setattr(torch, "cuda", _FakeCuda(), raising=False)
    tb = TorchBackend()
    tb.torch_lib = "cuda"
    tb.profile = {}
    assert tb.device_name(0) == "FakeGPU-Model"


def test_torchbackend_is_npu_default_false():
    tb = TorchBackend()
    assert tb.is_npu() is False


def test_torchbackend_has_device_default_true():
    tb = TorchBackend()
    assert tb.has_device() is True


# --- Task 3: three hardware backends onto TorchBackend ---
# device_type() is now config-driven: _build injects _segment_name = the yaml segment
# key. Tests that build backends directly must set _segment_name to mimic _build.


def test_npu_is_npu_device_type_soc_series(monkeypatch):
    nb = NpuTorchBackend()
    nb.torch_lib = "npu"
    nb.profile = {}
    nb._segment_name = "npu"  # mimic _build injection
    assert nb.is_npu() is True
    assert nb.device_type() == "npu"


def test_xpu_device_type_uses_device():
    xb = XpuTorchBackend()
    xb.torch_lib = "cuda"
    xb.profile = {}
    xb._segment_name = "xpu"  # mimic _build injection
    assert xb.device_type() == "xpu"
    assert xb.has_device() is True


def test_cpu_no_device():
    cb = CpuTorchBackend()
    cb.torch_lib = "cpu"
    cb.profile = {}
    # CPU never goes through _build; _segment_name = 'cpu' is a class attribute.
    assert cb.device_type() == "cpu"
    assert cb.has_device() is False


def test_npu_is_available_uses_torch_npu(monkeypatch):
    """NPU asys 去除：is_available 走 torch.npu.is_available（非 asys）。"""
    import torch

    class _FakeNpu:
        def is_available(self):
            return True

        def device_count(self):
            return 2

    monkeypatch.setattr(torch, "npu", _FakeNpu(), raising=False)
    nb = NpuTorchBackend()
    nb.torch_lib = "npu"
    nb.profile = {}
    assert nb.is_available() is True
    assert nb.device_count() == 2


def test_npu_soc_series_uses_model_not_segment(monkeypatch):
    """I2: soc_series() derives short series from device_name() (MODEL) via
    get_npu_hw_info, NOT the segment name 'npu'. Asserts the model is passed
    to get_npu_hw_info and short_soc_version is returned verbatim."""
    import torch
    from ttk.core_modules.framework_api.backends import npu_torch_backend

    class _FakeNpu:
        @staticmethod
        def get_device_name(dev_id):
            return "Ascend910B3"

    monkeypatch.setattr(torch, "npu", _FakeNpu(), raising=False)

    captured = {}

    def _fake_hw_info(full_soc_version):
        captured["arg"] = full_soc_version
        return {"short_soc_version": "Ascend910B"}

    monkeypatch.setattr(npu_torch_backend, "get_npu_hw_info", _fake_hw_info)

    nb = NpuTorchBackend()
    nb.torch_lib = "npu"
    nb.profile = {}
    assert nb.soc_series() == "Ascend910B"
    # get_npu_hw_info received the MODEL ('Ascend910B3'), not the segment 'npu'.
    assert captured["arg"] == "Ascend910B3"


# --- device_type = config-driven segment name (not hardcoded per subclass) ---


def test_build_device_type_is_segment_name_not_hardcoded():
    """_build injects _segment_name = the yaml segment key; device_type() returns it
    verbatim. A 'gpu' segment with torch_lib='cuda' yields device_type() == 'gpu'
    (NOT the hardcoded 'xpu' the old override returned)."""
    from ttk.core_modules.framework_api.backends import _build
    from ttk.core_modules.framework_api.backends.xpu_torch_backend import XpuTorchBackend

    profile = {"torch_lib": "cuda", "profiler": {"activities": ["CPU", "CUDA"]}}
    b = _build("torch", "gpu", profile)
    assert isinstance(b, XpuTorchBackend)  # cuda -> generic accelerator class
    assert b.torch_lib == "cuda"
    assert b.device_type() == "gpu"  # segment-name driven, not hardcoded "xpu"


def test_build_arbitrary_segment_name_carried_through():
    """Segment names are arbitrary: 'custom' segment -> device_type() == 'custom'."""
    from ttk.core_modules.framework_api.backends import _build
    from ttk.core_modules.framework_api.backends.xpu_torch_backend import XpuTorchBackend

    profile = {"torch_lib": "mlu", "profiler": "builtin"}
    b = _build("torch", "custom", profile)
    assert isinstance(b, XpuTorchBackend)
    assert b.device_type() == "custom"


def test_build_npu_torch_lib_routes_to_npu_backend():
    """torch_lib='npu' -> NpuTorchBackend regardless of segment name."""
    from ttk.core_modules.framework_api.backends import _build

    profile = {"torch_lib": "npu", "profiler": "builtin"}
    b = _build("torch", "ascend", profile)
    assert isinstance(b, NpuTorchBackend)
    assert b.device_type() == "ascend"  # segment name, not "npu"


def test_build_cpu_torch_lib_routes_to_cpu_backend():
    """torch_lib='cpu' -> CpuTorchBackend; _segment_name still injected."""
    from ttk.core_modules.framework_api.backends import _build

    profile = {"torch_lib": "cpu", "profiler": {"activities": ["CPU"]}}
    b = _build("torch", "cpu", profile)
    assert isinstance(b, CpuTorchBackend)
    assert b.device_type() == "cpu"
