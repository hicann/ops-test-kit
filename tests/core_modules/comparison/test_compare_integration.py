#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root directory of the software repository for the full text of the License.

"""
End-to-end integration of compare(): spec.tolerance → resolve_tolerance → compare(standards=) → 4-tuple → metrics.
"""

import numpy as np

from ttk.core_modules.comparison import compare
from ttk.core_modules.comparison.resolve import ResolvedStandard, resolve_tolerance


def test_compare_returns_4tuple_with_metrics():
    """compare() 返回 4-tuple，metrics 含 standard 字段。"""
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    standards = resolve_tolerance(None, None, None, ["float32"], None)  # -> stat_rel_err
    precision, log, is_pass, metrics = compare(outputs, goldens, ("float32",),
                                               standards=standards)
    assert is_pass is True
    assert 0 in metrics
    assert metrics[0]["standard"] == "stat_rel_err"


# —— 端到端:Spec.tolerance → resolve → compare → metrics → structure（CR5-I2）——

def test_threshold_override_flows_to_metrics():
    """Spec.tolerance threshold override 经 resolve → compare → stat_rel_err metrics。"""
    tolerance = {"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}
    standards = resolve_tolerance(tolerance, None, None, ["float32"], None)
    assert standards[0].params["threshold"] == 1e-3

    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = compare(outputs, goldens, ("float32",), standards=standards)
    assert metrics[0]["threshold"] == 1e-3
    assert metrics[0]["standard"] == "stat_rel_err"
    assert "mere" in metrics[0] and "mare" in metrics[0]


def test_metrics_flow_to_comparison_result():
    """compare() 4-tuple metrics → ComparisonResult.set(metrics) → .metrics 槽。"""
    from ttk.core_modules.npu.op.profiling_structure import ComparisonResult

    standards = resolve_tolerance(None, None, None, ["float32"], None)
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = compare(outputs, goldens, ("float32",), standards=standards)

    nested = {"dyn": metrics, "cst": metrics, "bin": metrics}
    cr = ComparisonResult(None).set("PASS", "PASS", "PASS", "PASS", nested)
    assert cr.metrics == nested
    assert cr.metrics["dyn"][0]["standard"] == "stat_rel_err"


def test_metrics_flow_to_api_structure():
    """op_api: compare() metrics → ApiComparisonResult.set(metrics) → ApiProfilingReturnStructure.precision_metrics。"""
    from ttk.core_modules.npu.op_api.profiling_structure import ApiComparisonResult, ApiProfilingReturnStructure

    standards = resolve_tolerance(None, None, None, ["float32"], None)
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    _p, _l, _ip, metrics = compare(outputs, goldens, ("float32",), standards=standards)

    acr = ApiComparisonResult(None).set("PASS", "PASS", metrics)
    assert acr.metrics == metrics

    prs = ApiProfilingReturnStructure()
    prs.construct(None, acr)  # context=None ok for this structure
    assert prs.precision_metrics == metrics
    assert prs.precision_metrics[0]["standard"] == "stat_rel_err"


def test_output_none_fails():
    """output=None + golden 非 None → NO_OUTPUT / FAIL。"""
    precision, _log, is_pass, _m = compare(
        [None], [np.array([1.0])], ("float32",),
        standards=[ResolvedStandard("stat_rel_err")])
    assert precision == "NO_OUTPUT"
    assert is_pass is False
