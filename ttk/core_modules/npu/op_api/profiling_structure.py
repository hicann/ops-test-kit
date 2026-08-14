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
Op Api NPU Profiling Structure
"""

__all__ = ["ApiProfilingReturnStructure", "ApiComparisonResult", "ApiProfilingResult"]


# Standard Packages
from typing import Optional, Union, Tuple, List

# Third-party Packages
from ...testcase_manager import TestcaseAclnn
from ....utilities import get_global_storage


def _pick_api_avg_us(api_prof, api_name):
    """Single aclnn call host-side avg time (us), excludes runtime APIs."""
    if not isinstance(api_prof, list):
        return None
    for item in api_prof:
        if item.get("API Name") == api_name:
            try:
                return round(float(item.get("Avg(us)", 0)), 3)
            except (TypeError, ValueError):
                return None
    return None


def _sum_op_avg_us(op_prof):
    """Sum of all NPU op kernel avg time (us) per call."""
    if not isinstance(op_prof, list):
        return None
    total = 0.0
    for item in op_prof:
        try:
            total += float(item.get("Avg Time(us)", 0))
        except (TypeError, ValueError):
            return None
    return round(total, 3)


class ApiProfilingResult:
    """
    RTS Profiling output
    """

    def __init__(
        self,
        success: bool,
        api_prof=None,
        op_prof=None,
        output_bytes=(None,),
        output_view_shapes=(None,),
        oob: str = "UNKNOWN",
        deterministic_status: Optional[str] = None,
    ):
        self.api_prof: Union[str, List[dict]] = api_prof
        self.op_prof: Union[str, List[dict]] = op_prof
        self.output_bytes: Optional[Union[tuple, list]] = output_bytes
        self.output_view_shapes: Optional[Union[tuple, list]] = output_view_shapes
        self.oob: Optional[str] = oob
        self.success = success
        self.deterministic_status: Optional[str] = deterministic_status

    @classmethod
    def fail(cls, fail_result: str) -> "ApiProfilingResult":
        return cls(False, fail_result, fail_result, (fail_result,), ("NO_OUTPUT",), "UNKNOWN")

    @property
    def oob_status(self):
        if not self.oob:
            return "PASS"
        oob_lst = self.oob.split(",")
        return "FAIL" if "FAIL" in oob_lst else "PASS"

    def failed(self):
        return not self.success and self.api_prof != "SUPPRESSED"


class ApiComparisonResult:
    __slots__ = ("precision", "passed", "metrics")

    def __init__(self, default_value):
        self.precision = default_value
        self.passed = default_value
        self.metrics = {}

    def set(self, a, b, metrics=None):
        self.precision = a
        self.passed = b
        self.metrics = metrics or {}
        return self

    def get(self) -> tuple:
        return tuple(getattr(self, name) for name in self.__slots__)


class ApiProfilingReturnStructure:
    """
    Structure for op api profiling return content.
    """

    __slots__ = (
        "precision",
        "precision_status",
        "precision_metrics",
        "batch_consistency_id",
        "deterministic_status",
        "soc",
        "api_perf_us",
        "op_perf_us",
        "perf_status",
        "xpu_metrics",
    )

    def __init__(self, default_value=None):
        self.precision = default_value
        # Precision
        self.precision_status = default_value
        self.precision_metrics = default_value
        # Special
        self.batch_consistency_id = default_value
        self.deterministic_status = default_value
        self.soc = get_global_storage().dev_plat
        # Performance
        self.api_perf_us = default_value
        self.op_perf_us = default_value
        self.perf_status = default_value
        self.xpu_metrics = {}

    # noinspection DuplicatedCode
    def construct(self, context: TestcaseAclnn, compare_result: ApiComparisonResult):
        """Construct the structure with context"""
        # Check prof_results and construct one if necessary
        self.precision = compare_result.precision
        self.precision_status = compare_result.passed
        self.precision_metrics = compare_result.metrics or {}
        self.batch_consistency_id = getattr(context, "batch_consistency_id", None)
        self.xpu_metrics = getattr(context, "xpu_metrics", {})
        # Performance data from msprof (already collected in ApiProfilingResult)
        prof_result = getattr(context, "prof_result", None)
        api_prof = getattr(prof_result, "api_prof", None) if prof_result else None
        op_prof = getattr(prof_result, "op_prof", None) if prof_result else None
        self.api_perf_us = _pick_api_avg_us(api_prof, getattr(context, "api_name", None))
        self.op_perf_us = _sum_op_avg_us(op_prof)
        self.perf_status = "PASS" if self.op_perf_us is not None else "UNKNOWN"

    @staticmethod
    def get_titles() -> tuple:
        return ApiProfilingReturnStructure.__slots__

    def pick_data(self, titles: Tuple[str]) -> tuple:
        """Pick result data via titles"""
        data = []
        for t in titles:
            if hasattr(self, t):
                data.append(getattr(self, t))
            else:
                data.append("")
        return tuple(data)
