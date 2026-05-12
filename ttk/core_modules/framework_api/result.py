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
        "precision",
        "precision_status",
        "device_perf_us",
        "cpu_perf_us",
        "kernel_count",
        "kernel_details",
    )

    def __init__(self):
        self.precision = None
        self.precision_status = None
        self.device_perf_us = None
        self.cpu_perf_us = None
        self.kernel_count = None
        self.kernel_details = None

    @staticmethod
    def get_titles():
        return FrameworkApiReturnStructure.__slots__

    def pick_data(self, titles):
        return tuple(getattr(self, t, None) for t in titles)

    def construct(self, precision_str, precision_passed, profile_result):
        """
        Build from comparison and profiling results.

        Args:
            precision_str: precision value string from comparison
            precision_passed: "PASS" or "FAIL"
            profile_result: ProfileResult from profiler
        """
        self.precision = precision_str
        self.precision_status = precision_passed

        if profile_result:
            if profile_result.elapsed_us > 0:
                self.device_perf_us = f"{profile_result.elapsed_us:.3f}"
            else:
                self.device_perf_us = "----"
            if profile_result.kernel_details:
                kd = profile_result.kernel_details
                self.cpu_perf_us = f"{kd.total_cpu_us:.3f}"
                self.kernel_count = str(len(kd.kernels))
                self.kernel_details = json.dumps(
                    [{"name": k.name, "avg": round(k.avg_us, 3),
                      "max": round(k.max_us, 3), "min": round(k.min_us, 3),
                      "calls": k.calls}
                     for k in kd.kernels],
                    ensure_ascii=False
                )