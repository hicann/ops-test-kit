from __future__ import annotations

"""TorchProfiler internal data-driven activities + result _device_time fallback.

Task 8: TorchProfiler.__init__ reads activities from backend.profile['profiler']
(data-driven, no =='gpu'/torch_lib=='cuda' string compares); result() uses
_device_acts (non-CPU activities) + _device_time 3-candidate pure-self fallback
covering torch 2.7+ (self_device_time_total) and legacy
(self_{device}_time_total, e.g. self_cuda_time_total / self_mlu_time_total).
"""
import pytest

from ttk.core_modules.framework_api.profiler import TorchProfiler, get_profiler


class _Evt:
    """Minimal stand-in for a torch.profiler.Event for fallback tests."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.key = "op"
        self.count = 1


class _FakeBackend:
    """Minimal backend stand-in carrying only what TorchProfiler reads."""

    def __init__(self, profile):
        self.profile = profile

    def device_type(self):
        return "fake"

    def is_npu(self):
        return False


def _make_profiler_via_real_init(monkeypatch, activities, torch_lib="cuda", device_time_attr=None):
    """Build a TorchProfiler through the real __init__ with profile() stubbed.

    monkeypatches torch.profiler.profile so no real profiler is constructed;
    the real ProfilerActivity is used (so getattr(ProfilerActivity, a) resolves).
    Returns (profiler, fake_backend).
    """
    import torch.profiler as tp

    captured = {}

    def _fake_profile(activities=None, record_shapes=False, **kw):
        captured["activities"] = activities
        captured["record_shapes"] = record_shapes
        return object()  # opaque stub; not exercised in __init__

    monkeypatch.setattr(tp, "profile", _fake_profile)
    profile = {
        "torch_lib": torch_lib,
        "profiler": {"activities": activities},
    }
    if device_time_attr is not None:
        profile["profiler"]["device_time_attr"] = device_time_attr
    backend = _FakeBackend(profile)
    prof = TorchProfiler(backend)
    return prof, backend, captured


def test_device_time_fallback_device_time_attr():
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = "cuda_time_total"
    assert prof._device_time(_Evt(cuda_time_total=5.0), "cuda") == 5.0


def test_device_time_fallback_self_device_time_total():
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = None
    assert prof._device_time(_Evt(self_device_time_total=7.0), "musa") == 7.0


def test_device_time_fallback_self_device_total_zero_not_treated_as_missing():
    """v=0.0（合法空闲 kernel）不能被误判为缺失而 fallback。

    self_device_time_total=0.0 must hit candidate 2 (is not None), NOT fall
    through to the legacy self_{device}_time_total candidate.
    """
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = None
    evt = _Evt(self_device_time_total=0.0, self_cuda_time_total=9.0)
    assert prof._device_time(evt, "cuda") == 0.0


def test_device_time_fallback_legacy_self_device_total():
    """Candidate 3: self_{device}_time_total (legacy, e.g. self_cuda_time_total).

    No self_device_time_total present (torch <2.7 cuda, or torch_mlu mlu) —
    legacy field by device name must be picked up.
    """
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = None
    evt = _Evt(self_cuda_time_total=11.0)  # 无 self_device_time_total
    assert prof._device_time(evt, "cuda") == 11.0


def test_device_time_final_no_attrs_returns_zero():
    """No device_time_* attrs at all -> default 0.0 (getattr fallback)."""
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = None
    evt = _Evt()  # 无任何 device_time_* 属性
    assert prof._device_time(evt, "cuda") == 0.0


# --- I1: __init__ data-driven activities + _device contract ---


def test_init_device_acts_from_activities_with_cuda(monkeypatch):
    """activities=[CPU, CUDA] -> _device_acts=["CUDA"], _device=torch_lib."""
    prof, backend, captured = _make_profiler_via_real_init(
        monkeypatch,
        activities=["CPU", "CUDA"],
        torch_lib="cuda",
    )
    assert prof._device_acts == ["CUDA"]
    assert prof._device == "cuda"
    # profile() received resolved ProfilerActivity members (2 of them).
    assert len(captured["activities"]) == 2


def test_init_device_acts_empty_for_cpu_only(monkeypatch):
    """activities=[CPU] -> _device_acts=[] (CPU-only profile)."""
    prof, backend, captured = _make_profiler_via_real_init(
        monkeypatch,
        activities=["CPU"],
        torch_lib="cpu",
    )
    assert prof._device_acts == []
    assert prof._device == "cpu"


def test_init_device_is_torch_lib_not_activity_name(monkeypatch):
    """I4: _device = profile['torch_lib'], NOT _device_acts[0].lower().

    xpu uses activity 'CUDA' but torch_lib 'cuda' — device must be the lib.
    (Here simulated with torch_lib 'musa' + activity 'MUSA'.)
    """
    prof, backend, captured = _make_profiler_via_real_init(
        monkeypatch,
        activities=["CPU", "CUDA"],
        torch_lib="musa",
    )
    assert prof._device == "musa"  # torch_lib, not "cuda" (activity name)


def test_init_unknown_activity_raises_valueerror(monkeypatch):
    """I6: unknown ProfilerActivity name -> ValueError listing valid names."""
    with pytest.raises(ValueError, match="unknown ProfilerActivity 'BOGUS'"):
        _make_profiler_via_real_init(monkeypatch, activities=["CPU", "BOGUS"])


def test_init_device_time_attr_passed_through(monkeypatch):
    """device_time_attr is read from profile['profiler'] when present."""
    prof, backend, captured = _make_profiler_via_real_init(
        monkeypatch,
        activities=["CPU", "CUDA"],
        torch_lib="cuda",
        device_time_attr="cuda_time_total",
    )
    assert prof._device_time_attr == "cuda_time_total"


# --- I1: result() cpu + device branches ---


class _FakeEvent:
    """Stand-in for a torch.profiler Event for result() tests."""

    def __init__(self, key, count, cpu_time_total=0.0, **device_attrs):
        self.key = key
        self.count = count
        self.cpu_time_total = cpu_time_total
        self.__dict__.update(device_attrs)


class _FakeKeyAverages:
    """Fake _prof whose key_averages() returns a fixed event list."""

    def __init__(self, events):
        self._events = events

    def key_averages(self):
        return self._events


def _prof_for_result(events, device_acts, device="cuda", device_time_attr=None):
    """Build a TorchProfiler (bypassing __init__) wired for result()."""
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._prof = _FakeKeyAverages(events)
    prof._device_acts = device_acts
    prof._device = device
    prof._device_time_attr = device_time_attr
    return prof


def test_result_cpu_branch_no_device():
    """result() with empty _device_acts: elapsed_us=0.0 + cpu_time_total summed."""
    events = [
        _FakeEvent("op_a", 2, cpu_time_total=100.0),
        _FakeEvent("op_b", 1, cpu_time_total=50.0),
    ]
    prof = _prof_for_result(events, device_acts=[], device="cpu")
    res = prof.result(_FakeBackend({}), repeat_count=1)
    assert res.elapsed_us == 0.0
    assert res.kernel_details.kernels == []
    assert res.kernel_details.total_device_us == 0.0
    assert res.kernel_details.total_cpu_us == 150.0


def test_result_device_branch_collects_kernels():
    """result() with _device_acts: device kernels via _device_time, elapsed = total/repeat."""
    events = [
        _FakeEvent("k1", 2, cpu_time_total=10.0, self_cuda_time_total=200.0),
        _FakeEvent("k2", 1, cpu_time_total=5.0, self_cuda_time_total=80.0),
        _FakeEvent("idle", 1, cpu_time_total=1.0, self_cuda_time_total=0.0),  # skipped
    ]
    prof = _prof_for_result(events, device_acts=["CUDA"], device="cuda")
    res = prof.result(_FakeBackend({}), repeat_count=2)
    # total_device_us = 200+80 = 280; elapsed = 280/2 = 140
    assert res.elapsed_us == 140.0
    assert res.kernel_details.total_device_us == 280.0
    names = [k.name for k in res.kernel_details.kernels]
    assert names == ["k1", "k2"]  # idle (0.0) skipped
    k1 = res.kernel_details.kernels[0]
    assert k1.device_us == 200.0
    assert k1.calls == 2
    assert k1.avg_us == 100.0


def test_result_device_branch_uses_explicit_attr():
    """device_time_attr override takes precedence over self_device_time_total."""
    events = [_FakeEvent("k1", 1, cuda_time_total=999.0, self_cuda_time_total=200.0)]
    prof = _prof_for_result(events, device_acts=["CUDA"], device="cuda", device_time_attr="cuda_time_total")
    res = prof.result(_FakeBackend({}), repeat_count=1)
    assert res.kernel_details.kernels[0].device_us == 999.0


# --- I5: get_profiler RuntimeError includes backend alias ---


def test_get_profiler_npu_api_on_non_npu_includes_backend_alias():
    """I5: torch_npu.* on non-NPU backend -> RuntimeError names current alias."""
    backend = _FakeBackend({"torch_lib": "cuda"})
    with pytest.raises(RuntimeError, match=r"current is 'fake'"):
        get_profiler("torch_npu.something", backend)


def test_get_profiler_torch_with_npu_builtin_returns_npu_profiler():
    """NPU backend + profiler='builtin' -> NpuProfiler (production route).

    Covers default.yaml actual routing: torch.add + NPU + builtin → NpuProfiler.
    """
    from ttk.core_modules.framework_api.profiler import NpuProfiler

    class _NpuBackend:
        """Mock NPU backend with builtin profiler config."""

        def __init__(self):
            self.profile = {"profiler": "builtin"}

        def device_type(self):
            return "npu"

        def is_npu(self):
            return True

    backend = _NpuBackend()
    profiler = get_profiler("torch.add", backend)
    assert isinstance(profiler, NpuProfiler), f"Expected NpuProfiler, got {type(profiler).__name__}"
