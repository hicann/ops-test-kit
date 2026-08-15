from __future__ import annotations

"""Regression test: tf.* api_name must NOT select TfProfiler.

TfProfiler is removed (not ready). Any tf.* api_name should fall through
to the generic WallClockProfiler regardless of backend.

Task 7: get_profiler is hardware-neutral (is_npu + profile['profiler']); the
fake backend carries the new-contract attributes even though tf.* never reads
them, so the fixture stays honest about what get_profiler may touch.
"""
from ttk.core_modules.framework_api.profiler import get_profiler, WallClockProfiler


class _FakeBackend:
    # Task 7 contract surface: get_profiler may read is_npu()/profile (for
    # torch.*/torch_npu.*); tf.* falls through without touching them.
    torch_lib = "cuda"
    profile = {"profiler": {"activities": ["CPU", "CUDA"]}}

    def device_name(self, dev_id=0):
        return "AscendGPU-Model-Name"

    def device_type(self):
        return "xpu"

    def is_npu(self):
        return False


def test_tf_apiname_falls_through_to_wallclock():
    prof = get_profiler("tf.raw.ops.Add", _FakeBackend())
    assert isinstance(prof, WallClockProfiler)
