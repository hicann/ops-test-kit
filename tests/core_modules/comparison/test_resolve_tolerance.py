# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
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


def test_fp8_defaults_mix_tolerance():
    """float8_e5m2 默认 mix_tolerance（生态算子开源精度标准）。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e5m2"], None)) == ["mix_tolerance"]


def test_normal_float_defaults_mix_tolerance():
    """float32 默认 mix_tolerance（生态算子开源精度标准）。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float32"], None)) == ["mix_tolerance"]


def test_normal_float_cli_wins():
    """CLI 指定优先于 Spec.tolerance。"""
    assert _tokens(resolve_tolerance({"float32": {"standard": "binary_equal"}}, None, None, ["float32"], "close")) == [
        "close"
    ]


def test_cli_mixed_alias_matches_mix_tolerance():
    """CLI 简写 --compare mixed 与 mix_tolerance 同判据同阈值表。"""
    rs = resolve_tolerance(None, None, None, ["float32"], "mixed")
    assert _tokens(rs) == ["mixed"]
    p = _params(rs)
    assert p["rtol"] == 2**-10  # mix_tolerance float32 表值
    assert p["required_matched_ratio"] == 0.99


def test_normal_float_spec_binary_equal():
    """Spec.tolerance 可将 float16 改为 binary_equal。"""
    assert _tokens(resolve_tolerance({"float16": {"standard": "binary_equal"}}, None, None, ["float16"], None)) == [
        "binary_equal"
    ]


def test_multi_output_mixed():
    """多输出混合 dtype → 各自路由到对应 standard。"""
    out = _tokens(resolve_tolerance(None, None, None, ["int32", "float32", "complex64", "float8_e5m2"], None))
    assert out == ["binary_equal", "mix_tolerance", "isclose", "mix_tolerance"]


# —— threshold 解析（resolve_tolerance 唯一解析点；比对类不查表）——
def _params(rs, idx=0):
    """取第 idx 个 ResolvedStandard 的 params dict。"""
    return rs[idx].params


def test_threshold_resolution():
    """Spec.tolerance threshold override 优先于表默认。"""
    spec = {"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}
    rs = resolve_tolerance(spec, None, None, ["float32"], None)
    assert _params(rs)["threshold"] == 1e-3


# —— mix_tolerance 阈值表解析 ——
def test_mix_tolerance_table_defaults():
    """dtype 命中混合容差表：rtol/atol/required_matched_ratio/max_abs_error_limit 表值。"""
    rs = resolve_tolerance(None, None, None, ["float16"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-9
    assert p["atol"] == 2**-9
    assert p["required_matched_ratio"] == 0.99
    assert p["max_abs_error_limit"] == max(1e-1, 32 * 2**-10)


def test_mix_tolerance_unknown_dtype_falls_back_float32():
    """表外浮点 dtype（float64）回落 float32 档。"""
    rs = resolve_tolerance(None, None, None, ["float64"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-10
    assert p["atol"] == 2**-16


def test_mix_tolerance_spec_override():
    """Spec.tolerance 可覆盖 rtol/atol/required_matched_ratio/max_abs_error_limit。"""
    spec = {"float32": {"standard": "mix_tolerance", "rtol": 0.5, "required_matched_ratio": 0.9}}
    p = _params(resolve_tolerance(spec, None, None, ["float32"], None))
    assert p["rtol"] == 0.5
    assert p["required_matched_ratio"] == 0.9
    assert p["atol"] == 2**-16  # 未覆盖的仍取表值


# —— threshold 与 mix_tolerance 的冲突护栏（不静默丢弃）——
def test_threshold_only_spec_without_standard_raises():
    """只配 threshold 未配 standard：默认路由 mix_tolerance 会丢弃阈值，必须报错。"""
    import pytest

    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance({"float16": {"threshold": 0.01}}, None, None, ["float16"], None)


def test_mix_tolerance_with_threshold_raises():
    """显式 standard='mix_tolerance' 却带 threshold：配置错误，报错。"""
    import pytest

    spec = {"float32": {"standard": "mix_tolerance", "threshold": 0.01}}
    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance(spec, None, None, ["float32"], None)


def test_threshold_only_spec_cli_stat_rel_err_still_works():
    """threshold-only Spec + CLI --compare stat_rel_err：判据命中，threshold 正常生效。"""
    rs = resolve_tolerance({"float16": {"threshold": 0.01}}, None, None, ["float16"], "stat_rel_err")
    assert _params(rs)["threshold"] == 0.01


def test_cli_mix_tolerance_overrides_stat_rel_err_threshold_raises():
    """Spec 声明 stat_rel_err+threshold 但 CLI 强制 mix_tolerance：阈值会失效，报错。"""
    import pytest

    spec = {"float32": {"standard": "stat_rel_err", "threshold": 0.002}}
    with pytest.raises(ValueError, match="threshold"):
        resolve_tolerance(spec, None, None, ["float32"], "mix_tolerance")


def test_mix_tolerance_fp8_table_defaults():
    """float8 命中混合容差阈值表（float8_e4m3fn / float8_e5m2 均在表内）。"""
    rs = resolve_tolerance(None, None, None, ["float8_e4m3fn"], None)
    p = _params(rs)
    assert p["rtol"] == 2**-2
    assert p["atol"] == 2**-4
    assert p["max_abs_error_limit"] == max(1e-0, 32 * 2**-3)


def test_mix_tolerance_hifloat32_has_ulp_part():
    """hifloat32 的 max_abs_error_limit 含 32*ULP 部分（标准：1e-1 or 32*ULP）。"""
    rs = resolve_tolerance(None, None, None, ["hifloat32"], None)
    p = _params(rs)
    assert p["max_abs_error_limit"] == max(1e-1, 32 * 2**-10)


def test_fp8_cli_requant_still_works():
    """float8 仍可通过 CLI --compare requant 显式选择 requant 判据。"""
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e5m2"], "requant")) == ["requant"]


def test_hifloat8_defaults_requant():
    """hifloat8 不在混合容差阈值表内（表外量化 dtype），保持 requant 默认。"""
    assert _tokens(resolve_tolerance(None, None, None, ["hifloat8"], None)) == ["requant"]
