# tests/test_resolve_tolerance.py
import pytest
from ttk.core_modules.comparison.resolve import resolve_tolerance, ResolvedStandard


def _tokens(rs):
    return [r.token for r in rs]


@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8", "uint1", "int4", "bool"])
def test_int_bool_always_binary_equal(dtype):
    assert _tokens(resolve_tolerance(None, None, None, [dtype], "close")) == ["binary_equal"]


@pytest.mark.parametrize("cli", [None, "stat_rel_err", "close"])
def test_complex_defaults_isclose(cli):
    assert _tokens(resolve_tolerance(None, None, None, ["complex64"], cli)) == ["isclose"]


def test_complex_bin_binary_overrides_to_binary_equal():
    assert _tokens(resolve_tolerance(None, None, None, ["complex64"], "binary")) == ["binary_equal"]
    assert _tokens(resolve_tolerance({"complex64": {"standard": "bin"}}, None, None, ["complex64"], None)) == ["binary_equal"]


def test_fp8_defaults_requant():
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e5m2"], None)) == ["requant"]


def test_fp8_respects_bin():
    assert _tokens(resolve_tolerance(None, None, None, ["float8_e4m3fn"], "bin")) == ["bin"]


def test_fp4_always_bin():
    assert _tokens(resolve_tolerance(None, None, None, ["float4_e2m1"], "close")) == ["bin"]


def test_normal_float_defaults_stat_rel_err():
    assert _tokens(resolve_tolerance(None, None, None, ["float32"], None)) == ["stat_rel_err"]


def test_normal_float_cli_wins():
    # CLI 指定优先于 Spec.tolerance
    assert _tokens(resolve_tolerance({"float32": {"standard": "binary_equal"}}, None, None, ["float32"], "close")) == ["close"]


def test_normal_float_spec_binary_equal():
    assert _tokens(resolve_tolerance({"float16": {"standard": "binary_equal"}}, None, None, ["float16"], None)) == ["binary_equal"]


def test_multi_output_mixed():
    out = _tokens(resolve_tolerance(None, None, None, ["int32", "float32", "complex64", "float8_e5m2"], None))
    assert out == ["binary_equal", "stat_rel_err", "isclose", "requant"]


def test_length_matches_outputs():
    assert len(resolve_tolerance(None, None, None, ["int32", "float32"], None)) == 2


# —— threshold 解析（resolve_tolerance 唯一解析点；比对类不查表）——
def _params(rs, idx=0):
    return rs[idx].params


def test_threshold_default_from_table():
    assert _params(resolve_tolerance(None, None, None, ["float32"], None))["threshold"] == 2**-13
    assert _params(resolve_tolerance(None, None, None, ["float16"], None))["threshold"] == 2**-10
    assert _params(resolve_tolerance(None, None, None, ["bfloat16"], None))["threshold"] == 2**-7


def test_threshold_unknown_dtype_uses_default():
    assert _params(resolve_tolerance(None, None, None, ["float64"], None))["threshold"] == 2**-13


def test_threshold_spec_override():
    rs = resolve_tolerance({"float32": {"standard": "stat_rel_err", "threshold": 1e-3}}, None, None,
                           ["float32"], None)
    assert _params(rs)["threshold"] == 1e-3


def test_threshold_override_wins_over_table():
    rs = resolve_tolerance({"float16": {"standard": "stat_rel_err", "threshold": 1e-2}}, None, None,
                           ["float16"], None)
    assert _params(rs)["threshold"] == 1e-2      # 不是表的 2^-10
