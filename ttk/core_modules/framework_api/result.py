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
Result data structures for framework_api profiling.
Follows TTK convention: __slots__-based class with get_titles() and pick_data().
"""
import json


class FrameworkApiReturnStructure:
    """Return structure from profile_process, written to result CSV."""

    __slots__ = (
        "precision_status",
        "eager_precision",
        "eager_device_perf_us",
        "eager_cpu_perf_us",
        "eager_kernel_count",
        "eager_kernel_details",
        "graph_cst_precision",
        "graph_cst_device_perf_us",
        "graph_cst_cpu_perf_us",
        "graph_cst_kernel_count",
        "graph_cst_kernel_details",
        "graph_dyn_precision",
        "graph_dyn_device_perf_us",
        "graph_dyn_cpu_perf_us",
        "graph_dyn_kernel_count",
        "graph_dyn_kernel_details",
        "graph_aclgraph_precision",
        "graph_aclgraph_device_perf_us",
        "graph_aclgraph_cpu_perf_us",
        "graph_aclgraph_kernel_count",
        "graph_aclgraph_kernel_details",
        "precision_metrics",
    )

    def __init__(self):
        self.precision_status = None
        self.eager_precision = None
        self.eager_device_perf_us = None
        self.eager_cpu_perf_us = None
        self.eager_kernel_count = None
        self.eager_kernel_details = None
        self.graph_cst_precision = None
        self.graph_cst_device_perf_us = None
        self.graph_cst_cpu_perf_us = None
        self.graph_cst_kernel_count = None
        self.graph_cst_kernel_details = None
        self.graph_dyn_precision = None
        self.graph_dyn_device_perf_us = None
        self.graph_dyn_cpu_perf_us = None
        self.graph_dyn_kernel_count = None
        self.graph_dyn_kernel_details = None
        self.graph_aclgraph_precision = None
        self.graph_aclgraph_device_perf_us = None
        self.graph_aclgraph_cpu_perf_us = None
        self.graph_aclgraph_kernel_count = None
        self.graph_aclgraph_kernel_details = None
        self.precision_metrics = {}

    @staticmethod
    def get_titles():
        return FrameworkApiReturnStructure.__slots__

    def pick_data(self, titles):
        return tuple(getattr(self, t, None) for t in titles)

    _MODE_PREFIX = {"static": "cst", "dynamic": "dyn", "aclgraph": "aclgraph"}

    def construct(self, precision_str, precision_passed, profile_result, mode=None, metrics=None):
        """
        Build from comparison and profiling results.

        Args:
            precision_str: precision value string from comparison
            precision_passed: "PASS" or "FAIL"
            profile_result: ProfileResult from profiler (may be None)
            mode: None for eager, "static" or "dynamic" for graph
            metrics: per-mode precision metrics dict (accumulated by mode key)
        """
        if mode is None:
            prefix = "eager_"
        else:
            prefix = f"graph_{self._MODE_PREFIX[mode]}_"

        setattr(self, f"{prefix}precision", precision_str)
        if precision_passed == "FAIL" or self.precision_status is None:
            self.precision_status = precision_passed

        if profile_result:
            if profile_result.elapsed_us > 0:
                setattr(self, f"{prefix}device_perf_us", f"{profile_result.elapsed_us:.3f}")
            else:
                setattr(self, f"{prefix}device_perf_us", "----")
            if profile_result.kernel_details:
                kd = profile_result.kernel_details
                setattr(self, f"{prefix}cpu_perf_us", f"{kd.total_cpu_us:.3f}")
                setattr(self, f"{prefix}kernel_count", str(len(kd.kernels)))
                setattr(self, f"{prefix}kernel_details", json.dumps(
                    [{"name": k.name, "avg": round(k.avg_us, 3),
                      "max": round(k.max_us, 3), "min": round(k.min_us, 3),
                      "calls": k.calls}
                      for k in kd.kernels],
                    ensure_ascii=False
                ))

        if metrics:
            mode_key = "eager" if mode is None else f"graph_{self._MODE_PREFIX[mode]}"
            self.precision_metrics[mode_key] = metrics
