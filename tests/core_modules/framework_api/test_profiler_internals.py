# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TorchProfiler 内部数据驱动 activities + result _device_time 回退测试。

Task 8: TorchProfiler.__init__ 从 backend.profile['profiler'] 读取 activities
（数据驱动，无 =='gpu'/torch_lib=='cuda' 字符串比较）；result() 使用
_device_acts（非 CPU activities）+ _device_time 三候选纯自身回退，
覆盖 torch 2.7+（self_device_time_total）和 legacy
（self_{device}_time_total，如 self_cuda_time_total / self_mlu_time_total）。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


class _FakeEvent:
    """Stand-in for a torch.profiler.Event for result() tests."""

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


def test_device_time_fallback_explicit_attr():
    """_device_time 显式 attr 直接读取该属性。"""
    prof = TorchProfiler.__new__(TorchProfiler)
    prof._device_time_attr = "cuda_time_total"
    evt = _Evt(cuda_time_total=5.0)
    assert prof._device_time(evt, "cuda") == 5.0


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


# --- I5: get_profiler routing ---


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


def test_npu_find_csv_picks_latest_mtime(tmp_path, caplog):
    """_find_csv returns the newest-mtime CSV when multiple trace subdirs exist.

    torch_npu.profiler's tensorboard_trace_handler writes each run into a new
    timestamped subdir under _outdir, so _outdir accumulates multiple copies
    of kernel_details.csv / operator_details.csv. os.walk yields subdirs in
    filesystem (os.listdir) order, not by time; _find_csv must pick by mtime to
    avoid reading a stale trace from a previous run.
    """
    import os

    from ttk.core_modules.framework_api.profiler import NpuProfiler

    prof = NpuProfiler(None, testcase_name="fresh_capture", root_path=str(tmp_path))
    prof._outdir = str(tmp_path)

    older_dir = tmp_path / "trace_older"
    newer_dir = tmp_path / "trace_newer"
    older_dir.mkdir()
    newer_dir.mkdir()
    older_csv = older_dir / "kernel_details.csv"
    newer_csv = newer_dir / "kernel_details.csv"
    older_csv.write_text("Name,Duration(us)\nold,10\n")
    newer_csv.write_text("Name,Duration(us)\nnew,20\n")

    older_str = str(older_csv)
    newer_str = str(newer_csv)
    os.utime(older_str, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer_str, ns=(2_000_000_000, 2_000_000_000))

    found = prof._find_csv("kernel_details.csv")
    assert found == newer_str
    prof._start_time_ns = 1_500_000_000
    assert prof._find_csv("kernel_details.csv") == newer_str
    prof._start_time_ns = 3_000_000_000
    assert prof._find_csv("kernel_details.csv") is None
    assert prof.result(None, 3).elapsed_us == 0
    assert "historical traces are not reused" in caplog.text


def test_npu_find_csv_returns_none_when_missing(tmp_path, caplog):
    """_find_csv returns None when no matching CSV exists in the tree."""
    from ttk.core_modules.framework_api.profiler import NpuProfiler

    prof = NpuProfiler(None, testcase_name="missing_capture", root_path=str(tmp_path))
    prof._outdir = str(tmp_path)
    (tmp_path / "trace").mkdir()
    (tmp_path / "trace" / "other.csv").write_text("x\n1\n")

    assert prof._find_csv("kernel_details.csv") is None
    assert "has no new kernel_details.csv" in caplog.text
    assert "historical traces are not reused" in caplog.text


@pytest.mark.parametrize("api_name", ["torch.add", "torch_npu.npu_rms_norm", "custom_op"])
@pytest.mark.parametrize("aclgraph", [False, True])
def test_aclgraph_kernel_accounting(tmp_path, api_name, aclgraph):
    from ttk.core_modules.framework_api.profiler import ProfilerConfig
    from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure

    backend = SimpleNamespace(is_npu=lambda: True, profile={"profiler": "builtin"})
    config = ProfilerConfig(testcase_name="capture", root_path=str(tmp_path), aclgraph=aclgraph)
    profiler = get_profiler(api_name, backend, config)

    (Path(profiler._outdir) / "kernel_details.csv").write_text(
        "Step Id,Name,Model ID,Duration(us)\n"
        "1,ForeachCopy,4294967295,40\n"
        "1,MainKernel,48,20\n"
        "1,MainKernel,48,22\n"
        "1,SmallKernel,49,1\n"
        "1,ForeachCopy,0,2\n"
        ",MainKernel,48,999\n"
        "1,Invalid,-1,3\n"
        "1,Unknown,N/A,4\n"
        "1,Malformed,broken,5\n"
        "1,BadDuration,48,not-a-number\n"
    )
    result = profiler.result(backend, 2)
    assert result.elapsed_us == (22.5 if aclgraph else 48.5)
    details = {kernel.name: kernel for kernel in result.kernel_details.kernels}
    assert details["MainKernel"].calls == 2
    assert details["ForeachCopy"].device_us == (2 if aclgraph else 42)
    if aclgraph:
        assert set(details) == {"MainKernel", "SmallKernel", "ForeachCopy"}
    report = FrameworkApiReturnStructure()
    report.construct("PASS", "PASS", result, mode="aclgraph" if aclgraph else None)
    prefix = "graph_aclgraph_" if aclgraph else "eager_"
    assert getattr(report, prefix + "device_perf_us") == ("22.500" if aclgraph else "48.500")
    assert sum(k["total"] for k in json.loads(getattr(report, prefix + "kernel_details"))) == result.elapsed_us * 2


@pytest.mark.parametrize(
    "contents",
    [
        "Step Id,Name,Duration(us)\n1,Copy,40\n",
        "Step Id,Name,Model ID,Duration(us)\n1,Copy,4294967295,40\n",
        "Step Id,Name,Model ID,Duration(us)\n1,Copy,,40\n",
    ],
)
def test_aclgraph_unknown_capture_does_not_fall_back_to_total(tmp_path, caplog, contents):
    from ttk.core_modules.framework_api.profiler import NpuProfiler

    path = tmp_path / "kernel_details.csv"
    path.write_text(contents)
    assert NpuProfiler._parse_kernel_details(path, aclgraph=True) == ([], 0.0)
    assert "ACLGraph performance unavailable" in caplog.text


@pytest.mark.parametrize("kind", ["inplace", "indexes", "kwargs"])
@pytest.mark.parametrize("aclgraph", [False, True])
def test_compiled_inplace_restores_values_and_aclgraph_addresses(tmp_path, kind, aclgraph, monkeypatch):
    import torch

    from ttk.core_modules.framework_api import graph_execution
    from ttk.core_modules.framework_api.backends.cpu_torch_backend import CpuTorchBackend
    from ttk.core_modules.framework_api.profiler import DisabledProfiler

    monkeypatch.setattr(graph_execution, "get_profiler", lambda *args: DisabledProfiler())
    backend = CpuTorchBackend()
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    backup = value.clone()
    pointers = []

    def compiled(*args, **kwargs):
        target = kwargs["value"] if kind == "kwargs" else args[0]
        assert torch.equal(target, backup)
        pointers.append(target.data_ptr())
        return target.add_(1)

    switches = SimpleNamespace(TASK_PROFILING=False, run_time=3, root_path=str(tmp_path), warmup=True)
    result, _, status = graph_execution._run_compiled(
        compiled,
        [] if kind == "kwargs" else [value],
        {"value": value} if kind == "kwargs" else {},
        backend,
        0,
        switches,
        kind == "inplace",
        backup if kind == "inplace" else None,
        "torch.add",
        inplace_backups={0: backup} if kind != "inplace" else None,
        inplace_kwargs_keys={0: "value"} if kind == "kwargs" else None,
        deterministic_level=1,
        is_aclgraph=aclgraph,
    )
    assert status == "PASS"
    assert torch.equal(torch.from_numpy(result[0]), backup + 1)
    assert len(set(pointers)) == (1 if aclgraph else 3)
    assert pointers[-1] == value.data_ptr()
