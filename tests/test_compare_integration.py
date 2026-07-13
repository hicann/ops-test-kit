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
import pytest

from ttk.core_modules.comparison import compare
from ttk.core_modules.comparison.resolve import resolve_tolerance, ResolvedStandard


def test_compare_returns_4tuple_with_metrics():
    outputs = [np.array([1.0, 2.0], np.float32)]
    goldens = [np.array([1.0, 2.0], np.float32)]
    standards = resolve_tolerance(None, None, None, ["float32"], None)  # -> stat_rel_err
    precision, log, is_pass, metrics = compare(outputs, goldens, ("float32",),
                                               standards=standards)
    assert is_pass is True
    assert 0 in metrics
    assert metrics[0]["standard"] == "stat_rel_err"


def test_compare_empty_outputs():
    precision, log, is_pass, metrics = compare([], [], (), standards=[])
    assert precision == "UNKNOWN" and is_pass is False
    assert metrics == {}


def test_compare_unknown_standard_raises():
    with pytest.raises(ValueError):
        compare([np.array([1.0])], [np.array([1.0])], ("float32",),
                standards=[ResolvedStandard("nonexistent_token")])


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
    from ttk.core_modules.npu.op_api.profiling_structure import (
        ApiComparisonResult, ApiProfilingReturnStructure
    )

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


# —— CR3 补齐：P2 NotImplementedError / None-dtype guard / cosine+requant metrics ——

def test_p2_tokens_raise_not_implemented():
    """cross_check/quant 被 validator 接受但 runtime 报 NotImplementedError。"""
    for token in ("quant",):    # cross_check 已注册成真比对类（本 step），不再 raise
        with pytest.raises(NotImplementedError):
            compare([np.array([1.0])], [np.array([1.0])], ("float32",),
                    standards=[ResolvedStandard(token)])


def test_none_dtype_routes_to_placeholder():
    """resolve_tolerance 对 None dtype emit 占位 token。"""
    rs = resolve_tolerance(None, None, None, [None], None)
    assert rs[0].token == "binary_equal"


def test_output_none_fails():
    """output=None + golden 非 None → NO_OUTPUT / FAIL。"""
    precision, _log, is_pass, _m = compare(
        [None], [np.array([1.0])], ("float32",),
        standards=[ResolvedStandard("stat_rel_err")])
    assert precision == "NO_OUTPUT"
    assert is_pass is False


def test_cosine_metrics_shape():
    """cosine compare_impl 填 standard+metrics。"""
    standards = resolve_tolerance({"float32": {"standard": "cosine"}}, None, None, ["float32"], None)
    _p, _l, _ip, metrics = compare(
        [np.array([1.0, 2.0, 3.0])], [np.array([1.0, 2.0, 3.0])],
        ("float32",), standards=standards)
    assert metrics[0]["standard"] == "cosine"
    assert "precision" in metrics[0] and "pass" in metrics[0]


def test_requant_metrics_shape():
    """requant compare_impl 填 standard+metrics。"""
    import ttk.core_modules.comparison.re_quantize  # noqa: F401
    from ttk.core_modules.comparison.registry import ComparisonRegister
    cls = ComparisonRegister.registry["requant"]
    c = cls(np.array([1, 2, 3], np.int8), np.array([1, 2, 3], np.int8), 0, "int8", {})
    r = c.compare_impl()
    assert r.standard == "requant"
    assert r.metrics["standard"] == "requant"
    assert "precision" in r.metrics and "pass" in r.metrics
