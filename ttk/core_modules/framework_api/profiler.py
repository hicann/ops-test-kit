#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""
Performance profiling using framework-specific profilers.
Context manager pattern: profiler only collects data within `with` block.
Warmup and repeat logic is controlled by the caller.
"""

import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KernelInfo:
    name: str
    device_us: float
    calls: int
    avg_us: float
    max_us: float = 0.0
    min_us: float = 0.0


@dataclass
class KernelDetails:
    kernels: List[KernelInfo] = field(default_factory=list)
    total_device_us: float = 0.0
    total_cpu_us: float = 0.0


@dataclass
class ProfileResult:
    elapsed_us: float = 0.0
    kernel_details: Optional[KernelDetails] = None


@dataclass
class ProfilerConfig:
    """Runtime options shared by the framework profiler implementations."""

    testcase_name: str = ""
    root_path: str = "."
    dev_id: int = 0
    enabled: bool = True
    warmup_count: int = 0


class FrameworkProfiler(ABC):
    """Framework-level profiler abstraction (context manager)."""

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, *exc):
        pass

    @abstractmethod
    def result(self, backend, repeat_count) -> ProfileResult:
        pass

    def step(self):  # noqa: B027
        pass


class DisabledProfiler(FrameworkProfiler):
    """No-op profiler used when task profiling is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def result(self, backend, repeat_count) -> ProfileResult:
        return ProfileResult()


class NpuProfiler(FrameworkProfiler):
    """Profiler for NPU using torch_npu.profiler.

    Uses torch_npu.profiler.profile (start/step/stop) and parses the exported
    kernel_details.csv / operator_details.csv for device-side timing.
    """

    def __init__(self, backend, testcase_name="", root_path=".", warmup_count=0):
        self._testcase_name = testcase_name or "unknown"
        self._outdir = os.path.join(root_path, "msprof", "e2e", self._testcase_name)
        os.makedirs(self._outdir, exist_ok=True)
        self._warmup_count = warmup_count
        self._prof = None

    def __enter__(self):
        from torch_npu.profiler import (
            AiCMetrics,
            ExportType,
            ProfilerActivity,
            ProfilerLevel,
            _ExperimentalConfig,
            profile,
            schedule,
            tensorboard_trace_handler,
        )

        experimental_config = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization,
            export_type=ExportType.Text,
        )
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            experimental_config=experimental_config,
            schedule=schedule(wait=0, warmup=self._warmup_count, active=1, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self._outdir),
        )
        self._prof.start()
        return self

    def step(self):
        if self._prof:
            self._prof.step()

    def __exit__(self, *exc):
        if self._prof:
            self._prof.step()
            self._prof.stop()

    def result(self, backend, repeat_count) -> ProfileResult:
        kernel_csv = self._find_csv("kernel_details.csv")
        operator_csv = self._find_csv("operator_details.csv")

        kernels = []
        total_device_us = 0.0
        total_cpu_us = 0.0

        if kernel_csv:
            kernels, total_device_us = self._parse_kernel_details(kernel_csv)
        if operator_csv:
            total_cpu_us = self._parse_operator_cpu_time(operator_csv)

        return ProfileResult(
            elapsed_us=total_device_us / max(repeat_count, 1),
            kernel_details=KernelDetails(
                kernels=kernels,
                total_device_us=total_device_us,
                total_cpu_us=total_cpu_us,
            ),
        )

    def _find_csv(self, filename):
        """Find the latest-mtime CSV file in the profiler output directory tree.

        torch_npu.profiler's tensorboard_trace_handler writes each trace into
        a new timestamped subdir under _outdir, so multiple runs accumulate
        multiple copies of kernel_details.csv / operator_details.csv. os.walk
        yields subdirs in filesystem (os.listdir) order, not by time, so the
        first match is not necessarily the newest — pick by mtime instead.
        """
        matches = []
        for root, _, files in os.walk(self._outdir):
            if filename in files:
                matches.append(os.path.join(root, filename))
        if not matches:
            return None
        return max(matches, key=os.path.getmtime)

    @staticmethod
    def _parse_kernel_details(csv_path):
        """Parse kernel_details.csv for per-kernel device timing.

        Rows with empty Step Id belong to the profiler warmup phase and are
        skipped so they don't skew the aggregated timing.
        """
        kernels_map = {}  # name -> {total_us, calls, max_us, min_us}
        total_device_us = 0.0

        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    step_id = row.get("Step Id", "").strip()
                    if not step_id:
                        continue
                    name = row.get("Name", "").strip()
                    try:
                        duration = float(row.get("Duration(us)", 0))
                    except (ValueError, TypeError):
                        continue
                    if name and duration > 0:
                        total_device_us += duration
                        if name in kernels_map:
                            kernels_map[name]["total_us"] += duration
                            kernels_map[name]["calls"] += 1
                            kernels_map[name]["max_us"] = max(kernels_map[name]["max_us"], duration)
                            kernels_map[name]["min_us"] = min(kernels_map[name]["min_us"], duration)
                        else:
                            kernels_map[name] = {
                                "total_us": duration,
                                "calls": 1,
                                "max_us": duration,
                                "min_us": duration,
                            }
        except Exception as e:
            logging.warning(f"Failed to parse {csv_path}: {e}")

        kernels = [
            KernelInfo(
                name=name,
                device_us=info["total_us"],
                calls=info["calls"],
                avg_us=info["total_us"] / info["calls"],
                max_us=info["max_us"],
                min_us=info["min_us"],
            )
            for name, info in kernels_map.items()
        ]
        return kernels, total_device_us

    @staticmethod
    def _parse_operator_cpu_time(csv_path):
        """Parse operator_details.csv for total CPU time."""
        total_cpu_us = 0.0
        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cpu_us = float(row.get("Host Self Duration(us)", 0))
                    except (ValueError, TypeError):
                        continue
                    total_cpu_us += cpu_us
        except Exception as e:
            logging.warning(f"Failed to parse {csv_path}: {e}")
        return total_cpu_us


class TorchProfiler(FrameworkProfiler):
    """Profiler for torch using torch.profiler. Works on both CPU and accelerators.

    Hardware-neutral: activities come from
    ``backend.profile["profiler"]["activities"]`` (a list of ProfilerActivity
    attribute names like "CPU"/"CUDA"/"MLU"/"MUSA") rather than a string compare
    on torch_lib. ``_device_acts`` is the non-CPU subset; ``result`` reports
    device kernels via ``_device_time`` (multi-candidate fallback across torch
    versions / device bindings).
    """

    def __init__(self, backend):
        from torch.profiler import ProfilerActivity, profile

        cfg = backend.profile["profiler"]
        activities = []
        for a in cfg["activities"]:
            try:
                activities.append(getattr(ProfilerActivity, a))
            except AttributeError:
                valid = [n for n in dir(ProfilerActivity) if not n.startswith("_")]
                raise ValueError(f"unknown ProfilerActivity '{a}'; valid: {valid}") from None
        self._prof = profile(activities=activities, record_shapes=True)
        # Non-CPU activity names (e.g. ["MLU"]); empty for CPU-only profiles.
        self._device_acts = [a for a in cfg["activities"] if a != "CPU"]
        self._device_time_attr = cfg.get("device_time_attr")
        # device key for _device_time legacy fallback (self_{device}_time_total).
        # device = profile["torch_lib"] (mlu/musa/...), NOT the activity name
        # (activity name is a ProfilerActivity enum; torch_lib is the torch module attr).
        self._device = backend.profile["torch_lib"]

    def _device_time(self, evt, device=None):
        """3-candidate device-time extraction for one event (pure self_).

        Order: explicit device_time_attr (if configured) → self_device_time_total
        (torch 2.7+ cuda/musa unified) → self_{device}_time_total (legacy,
        e.g. self_cuda_time_total / self_mlu_time_total).

        Uses ``is not None`` (not truthiness) so a legitimate v=0.0 idle kernel
        is not mis-treated as missing and falls through to a wrong fallback.
        ``device`` defaults to self._device (profile["torch_lib"]).

        Only ``self_*`` fields are used (self time, excluding nested child ops);
        ``device_time_total`` (total time, includes nested) was dropped to avoid
        mixing self_/total semantics — it would inflate the sum when nested ops
        double-count, and it was redundant (self_device_time_total already
        covers the musa scenario it was meant for).
        """
        if device is None:
            device = self._device
        if self._device_time_attr is not None:
            return getattr(evt, self._device_time_attr, 0.0)
        v = getattr(evt, "self_device_time_total", None)
        if v is not None:
            return v
        return getattr(evt, f"self_{device}_time_total", 0.0)

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *exc):
        return self._prof.__exit__(*exc)

    def result(self, backend, repeat_count) -> ProfileResult:
        events = self._prof.key_averages()

        total_cpu_us = sum(getattr(e, "cpu_time_total", 0.0) for e in events)

        if self._device_acts:
            # Device kernels via _device_time fallback (covers mlu/musa/...).
            # device key comes from self._device (profile["torch_lib"]).
            device_kernels = []
            total_device_us = 0.0
            for evt in events:
                device_us = self._device_time(evt)
                if device_us > 0:
                    device_kernels.append(
                        KernelInfo(
                            name=evt.key,
                            device_us=device_us,
                            calls=evt.count,
                            avg_us=device_us / max(evt.count, 1),
                        )
                    )
                    total_device_us += device_us
            return ProfileResult(
                elapsed_us=total_device_us / max(repeat_count, 1),
                kernel_details=KernelDetails(
                    kernels=device_kernels,
                    total_device_us=total_device_us,
                    total_cpu_us=total_cpu_us,
                ),
            )

        # CPU backend: no device timing, only CPU wall-clock
        return ProfileResult(
            elapsed_us=0.0,
            kernel_details=KernelDetails(
                kernels=[],
                total_device_us=0.0,
                total_cpu_us=total_cpu_us / max(repeat_count, 1),
            ),
        )


class WallClockProfiler(FrameworkProfiler):
    """Fallback wall-clock profiler for unknown frameworks."""

    def __init__(self, backend):
        self._start = None
        self._elapsed = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._elapsed = time.perf_counter() - self._start

    def result(self, backend, repeat_count) -> ProfileResult:
        return ProfileResult(
            elapsed_us=(self._elapsed or 0.0) * 1e6 / max(repeat_count, 1),
            kernel_details=None,
        )


_KERNEL_TASK_TYPES = frozenset(
    {
        "AI_CORE",
        "AIV_SQE",
        "AI_VECTOR_CORE",
        "MIX_AIC",
        "MIX_AIV",
        "KERNEL_MIX_AIC",
        "KERNEL_MIX_AIV",
        "KERNEL_AIVEC",
        "KERNEL_AICORE",
    }
)


class TfNpuProfiler(FrameworkProfiler):
    """Profiler for TF NPU using CANN msprof directly.

    TF ops do not go through torch's dispatcher, so torch_npu.profiler's
    kernel_details.csv is not produced.  We use MsProfiler (libmsprofiler.so)
    to collect CANN task-level records, then parse task_time.csv for
    per-kernel device timing.  Eager mode falls back to wall-clock timing.
    """

    def __init__(self, backend, testcase_name="", root_path=".", dev_id=0):
        self._testcase_name = testcase_name or "unknown"
        self._outdir = os.path.join(root_path, "msprof", "e2e", self._testcase_name)
        os.makedirs(self._outdir, exist_ok=True)
        self._dev_id = dev_id
        self._prof = None
        self._wall = WallClockProfiler(backend)

    def __enter__(self):
        from ..msprof import MsProfiler, TtkMsProfType

        self._prof = MsProfiler(
            self._dev_id,
            result_path=self._outdir,
            ttk_prof_type=TtkMsProfType.API,
            start_step=0,
        )
        self._prof.__enter__()
        self._prof.step()
        self._wall.__enter__()
        return self

    def __exit__(self, *exc):
        self._wall.__exit__(*exc)
        if self._prof:
            self._prof.__exit__(*exc)

    def result(self, backend, repeat_count) -> ProfileResult:
        kernels, total_us = self._sum_op_avg_us()
        if total_us is not None and total_us > 0:
            return ProfileResult(
                elapsed_us=total_us,
                kernel_details=KernelDetails(
                    kernels=kernels,
                    total_device_us=total_us * max(repeat_count, 1),
                    total_cpu_us=0.0,
                ),
            )
        return self._wall.result(backend, repeat_count)

    @staticmethod
    def _parse_op_row(row):
        """Parse one op_statistic row into KernelInfo, or None if invalid."""
        try:
            avg_us = float(row.get("Avg Time(us)", 0))
            total_us = float(row.get("Total Time(us)", 0))
            count = int(row.get("Count", 0))
            min_us = float(row.get("Min Time(us)", 0))
            max_us = float(row.get("Max Time(us)", 0))
        except (ValueError, TypeError):
            return None
        if avg_us <= 0:
            return None
        name = row.get("OP Type", "").strip()
        return KernelInfo(
            name=name,
            device_us=total_us,
            calls=count,
            avg_us=avg_us,
            max_us=max_us,
            min_us=min_us,
        )

    @staticmethod
    def _collect_op_rows(csv_path):
        """Read op_statistic CSV and return (kernels, total_avg_us)."""
        kernels = []
        total = 0.0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                info = TfNpuProfiler._parse_op_row(row)
                if info is not None:
                    kernels.append(info)
                    total += info.avg_us
        return kernels, total

    def _sum_op_avg_us(self):
        """Sum of all NPU op kernel avg time (us) per call, from op_statistic_*.csv.

        Matches aclnn通路: reads Avg Time(us) column, which is already
        averaged by msprof over the actual采集次数.
        Returns (kernels, total_us) — one KernelInfo per op row.
        """
        csv_path = self._find_csv_prefix("op_statistic_")
        if not csv_path:
            return [], None
        try:
            kernels, total = self._collect_op_rows(csv_path)
        except Exception as e:
            logging.warning(f"Failed to parse {csv_path}: {e}")
            return [], None
        total = round(total, 3) if total > 0 else None
        return kernels, total

    def _find_csv_prefix(self, prefix):
        for root, _, files in os.walk(self._outdir):
            for f in files:
                if f.startswith(prefix) and f.endswith(".csv"):
                    return os.path.join(root, f)
        return None


def get_profiler(
    api_name: str,
    backend,
    config: Optional[ProfilerConfig] = None,
) -> FrameworkProfiler:
    """Select profiler based on api_name prefix and backend.

    Hardware-neutral: routes on is_npu() + profile['profiler'] rather
    than device_name() string compares.
    """
    if config is None:
        config = ProfilerConfig()
    elif not isinstance(config, ProfilerConfig):
        raise TypeError(f"config must be ProfilerConfig, got {type(config).__name__}")
    if not config.enabled:
        return DisabledProfiler()
    if api_name.startswith(("tf.", "tensorflow.")):
        if backend.is_npu():
            return TfNpuProfiler(backend, config.testcase_name, config.root_path, dev_id=config.dev_id)
        return WallClockProfiler(backend)
    if api_name.startswith("torch_npu."):
        if not backend.is_npu():
            raise RuntimeError(f"API '{api_name}' requires NPU backend, but current is '{backend.device_type()}'")
        return NpuProfiler(backend, config.testcase_name, config.root_path, warmup_count=config.warmup_count)
    if api_name.startswith("torch."):
        # NPU with builtin profiler -> NpuProfiler; otherwise TorchProfiler.
        if backend.is_npu() and backend.profile.get("profiler") == "builtin":
            return NpuProfiler(backend, config.testcase_name, config.root_path, warmup_count=config.warmup_count)
        return TorchProfiler(backend)

    if backend.is_npu():
        return NpuProfiler(backend, config.testcase_name, config.root_path, warmup_count=config.warmup_count)
    return WallClockProfiler(backend)
