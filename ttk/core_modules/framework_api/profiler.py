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


class NpuProfiler(FrameworkProfiler):
    """Profiler for NPU using torch_npu.profiler.

    Uses torch_npu.profiler.profile (start/step/stop) and parses the exported
    kernel_details.csv / operator_details.csv for device-side timing.
    """

    def __init__(self, backend):
        self._tmpdir = tempfile.mkdtemp(prefix="ttk_npu_prof_")
        self._prof = None

    def _cleanup_tmpdir(self):
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
    """Profiler for torch using torch.profiler. Works on both GPU and CPU."""

    def __init__(self, backend):
        from torch.profiler import profile, ProfilerActivity

        activities = [ProfilerActivity.CPU]
        if backend.device_name() == "gpu":
            activities.append(ProfilerActivity.CUDA)
        self._prof = profile(activities=activities, record_shapes=True)
        self._has_cuda = backend.device_name() == "gpu"

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *exc):
        return self._prof.__exit__(*exc)

    def result(self, backend, repeat_count) -> ProfileResult:
        events = self._prof.key_averages()

        total_cpu_us = sum(getattr(e, 'cpu_time_total', 0.0) for e in events)

        if self._has_cuda:
            device_kernels = []
            total_device_us = 0.0
            for evt in events:
                device_us = getattr(evt, 'cuda_time_total', 0.0)
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


class TfProfiler(FrameworkProfiler):
    """Profiler for tf.* using tf.profiler.experimental + wall-clock fallback."""

    def __init__(self, backend):
        self._tmpdir = tempfile.mkdtemp(prefix="ttk_tf_prof_")
        self._start = None
        self._elapsed = None
        self._profiler_ok = False

    def __enter__(self):
        try:
            import tensorflow as tf
            tf.profiler.experimental.start(self._tmpdir)
            self._profiler_ok = True
        except Exception as e:
            logging.warning(f"tf.profiler start failed, falling back to wall-clock: {e}")
            self._profiler_ok = False
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._profiler_ok:
            try:
                import tensorflow as tf
                tf.profiler.experimental.stop()
            except Exception:
                pass
        self._elapsed = time.perf_counter() - self._start
        self._cleanup_tmpdir()

    def _cleanup_tmpdir(self):
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def result(self, backend, repeat_count) -> ProfileResult:
        total_wall_us = (self._elapsed or 0.0) * 1e6
        kernels = []
        total_device_us = 0.0

        if self._profiler_ok:
            kernels, total_device_us = self._parse_trace()

        if not kernels:
            return ProfileResult(
                elapsed_us=total_wall_us / max(repeat_count, 1),
                kernel_details=None,
            )

        return ProfileResult(
            elapsed_us=total_device_us / max(repeat_count, 1),
            kernel_details=KernelDetails(
                kernels=kernels,
                total_device_us=total_device_us,
                total_cpu_us=total_wall_us,
            ),
        )

    def _parse_trace(self):
        """Parse tf profiler trace JSON for per-op device timing."""
        import json

        kernels_map = {}
        total_device_us = 0.0

        trace_files = []
        if self._tmpdir:
            for root, _, files in os.walk(self._tmpdir):
                for f in files:
                    if f.endswith(".json"):
                        trace_files.append(os.path.join(root, f))

        for trace_file in trace_files:
            try:
                with open(trace_file, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for evt in events:
                if evt.get("ph") != "X":
                    continue
                cat = evt.get("cat", "")
                if "op" not in cat.lower() and "kernel" not in cat.lower():
                    continue
                name = evt.get("name", "")
                dur_us = evt.get("dur", 0) / 1000.0
                if name and dur_us > 0:
                    total_device_us += dur_us
                    if name in kernels_map:
                        kernels_map[name]["total_us"] += dur_us
                        kernels_map[name]["calls"] += 1
                    else:
                        kernels_map[name] = {"total_us": dur_us, "calls": 1}

        kernels = [
            KernelInfo(
                name=name,
                device_us=info["total_us"],
                calls=info["calls"],
                avg_us=info["total_us"] / info["calls"],
            )
            for name, info in kernels_map.items()
        ]
        return kernels, total_device_us


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


def get_profiler(api_name: str, backend) -> FrameworkProfiler:
    """Select profiler based on api_name prefix and backend."""
    device = backend.device_name()
    if api_name.startswith("torch_npu."):
        if device != "npu":
            raise RuntimeError(
                f"API '{api_name}' requires NPU backend but current backend is '{device}'"
            )
        return NpuProfiler(backend)
    if api_name.startswith("torch."):
        if device == "npu":
            return NpuProfiler(backend)
        return TorchProfiler(backend)
    elif api_name.startswith("tf."):
        if device == "gpu":
            return TfProfiler(backend)
        return WallClockProfiler(backend)
    return WallClockProfiler(backend)