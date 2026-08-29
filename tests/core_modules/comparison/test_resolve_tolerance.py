# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""resolve_tolerance 单元测试：dtype→standard 路由 + threshold 解析（表默认/spec override/override 优先级）。"""

from ttk.core_modules.comparison.resolve import resolve_tolerance


def _tokens(rs):
    """提取 ResolvedStandard 列表的 token 字符串。"""
    return [r.token for r in rs]


def test_int_bool_always_binary_equal():
    """整型/布尔 dtype 始终路由到 binary_equal。"""
    for dtype in ["int32", "bool"]:
        assert _tokens(resolve_tolerance(None, None, None, [dtype], "close")) == ["binary_equal"]


def test_complex_defaults_isclose():
    """complex64 默认 isclose。"""
    assert _tokens(resolve_tolerance(None, None, None, ["complex64"], None)) == ["isclose"]


def test_fp8_defaults_requant():
    """float8_e5m2 默认 requant。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e5m2"], None)) == ["requant"]


def test_normal_float_defaults_stat_rel_err():
    """float32 默认 stat_rel_err。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float32"], None)) == ["stat_rel_err"]


def test_normal_float_cli_wins():
    """CLI 指定优先于 Spec.tolerance。"""
    assert _tokens(resolve_tolerance({"float32": {"standard": "binary_equal"}}, None, None, ["float32"], "close")) == [
        "close"
    ]


def test_normal_float_spec_binary_equal():
    """Spec.tolerance 可将 float16 改为 binary_equal。"""
    assert _tokens(resolve_tolerance({"float16": {"standard": "binary_equal"}}, None, None, ["float16"], None)) == [
        "binary_equal"
    ]


def test_multi_output_mixed():
    """多输出混合 dtype → 各自路由到对应 standard。"""
    out = _tokens(resolve_tolerance(None, None, None, ["int32", "float32", "complex64", "float8_e5m2"], None))
    assert out == ["binary_equal", "stat_rel_err", "isclose", "requant"]


# —— threshold 解析（resolve_tolerance 唯一解析点；比对类不查表）——
def _params(rs, idx=0):
    """取第 idx 个 ResolvedStandard 的 params dict。"""
    return rs[idx].params


def test_threshold_resolution():
    """Spec.tolerance threshold override 优先于表默认。"""
    spec = {"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}
    rs = resolve_tolerance(spec, None, None, ["float32"], None)
    assert _params(rs)["threshold"] == 1e-3
