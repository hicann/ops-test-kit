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
import shutil
import tempfile
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


class DisabledProfiler(FrameworkProfiler):
    """No-op profiler used when ``--task-prof false`` is selected."""

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

    def __init__(self, backend, result_path=None):
        self._preserve_result = result_path is not None
        if result_path is None:
            self._tmpdir = tempfile.mkdtemp(prefix="ttk_npu_prof_")
        else:
            self._tmpdir = os.path.abspath(result_path)
            if os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
            os.makedirs(self._tmpdir, exist_ok=True)
        self._prof = None

    def _cleanup_tmpdir(self):
        if self._preserve_result:
            return
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        from torch_npu.profiler import (
            ProfilerActivity,
            _ExperimentalConfig,
            ProfilerLevel,
            AiCMetrics,
            ExportType,
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
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            on_trace_ready=tensorboard_trace_handler(self._tmpdir),
        )
        self._prof.start()
        # Advance past warmup phase so the active phase starts on __exit__'s step()
        self._prof.step()
        return self

    def __exit__(self, *exc):
        if self._prof:
            self._prof.step()
            self._prof.stop()

    def __del__(self):
        self._cleanup_tmpdir()

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

        self._cleanup_tmpdir()

        return ProfileResult(
            elapsed_us=total_device_us / max(repeat_count, 1),
            kernel_details=KernelDetails(
                kernels=kernels,
                total_device_us=total_device_us,
                total_cpu_us=total_cpu_us,
            ),
        )

    def _find_csv(self, filename):
        """Find a CSV file in the profiler output directory tree."""
        for root, _, files in os.walk(self._tmpdir):
            if filename in files:
                return os.path.join(root, filename)
        return None

    @staticmethod
    def _parse_kernel_details(csv_path):
        """Parse kernel_details.csv for per-kernel device timing."""
        kernels_map = {}  # name -> {total_us, calls, max_us, min_us}
        total_device_us = 0.0

        try:
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
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
                            kernels_map[name] = {"total_us": duration, "calls": 1,
                                                  "max_us": duration, "min_us": duration}
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
        from torch.profiler import profile, ProfilerActivity

        cfg = backend.profile["profiler"]
        activities = []
        for a in cfg["activities"]:
            try:
                activities.append(getattr(ProfilerActivity, a))
            except AttributeError:
                valid = [n for n in dir(ProfilerActivity) if not n.startswith("_")]
                raise ValueError(
                    f"unknown ProfilerActivity '{a}'; valid: {valid}"
                ) from None
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
                    device_kernels.append(KernelInfo(
                        name=evt.key,
                        device_us=device_us,
                        calls=evt.count,
                        avg_us=device_us / max(evt.count, 1),
                    ))
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


def get_profiler(
    api_name: str, backend, *, enabled: bool = True, result_path=None
) -> FrameworkProfiler:
    """Select profiler based on api_name prefix and backend.

    Hardware-neutral: routes on is_npu() + profile['profiler'] rather
    than device_name() string compares.
    """
    if not enabled:
        return DisabledProfiler()
    if api_name.startswith("torch_npu."):
        if not backend.is_npu():
            raise RuntimeError(
                f"API '{api_name}' requires NPU backend, "
                f"but current is '{backend.alias()}'"
            )
        return NpuProfiler(backend, result_path=result_path)
    if api_name.startswith("torch."):
        # NPU with builtin profiler -> NpuProfiler; otherwise TorchProfiler.
        if backend.is_npu() and backend.profile.get("profiler") == "builtin":
            return NpuProfiler(backend, result_path=result_path)
        return TorchProfiler(backend)
    return WallClockProfiler(backend)
