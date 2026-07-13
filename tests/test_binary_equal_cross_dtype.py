import numpy as np
import pytest

# 触发各比对类的注册（装饰器在 import 时执行）
import ttk.core_modules.comparison.binary_equal  # noqa: F401
import ttk.core_modules.comparison.is_close  # noqa: F401
import ttk.core_modules.comparison.cosine_similarity  # noqa: F401
import ttk.core_modules.comparison.re_quantize  # noqa: F401

from ttk.core_modules.comparison.registry import ComparisonRegister


def _cls(token):
    return ComparisonRegister.registry[token]


def test_binary_equal_alias_registered():
    assert _cls("binary_equal") is _cls("bin") is _cls("binary")


def test_isclose_alias_registered():
    assert _cls("isclose") is _cls("close")


def test_int32_int64_equal_passes():
    c = _cls("binary_equal")(np.array([1, 2, 3], np.int32), np.array([1, 2, 3], np.int64), 0, "int32", {})
    p, _l, is_pass, metrics = c.compare()
    assert is_pass is True and p == "100%"
    assert metrics["standard"] == "binary_equal" and metrics["pass"] is True


def test_int32_int64_different_fails():
    c = _cls("binary_equal")(np.array([1, 2], np.int32), np.array([1, 3], np.int64), 0, "int32", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


def test_bool_vs_int_passes():
    c = _cls("binary_equal")(np.array([False, True]), np.array([0, 1], np.int32), 0, "bool", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is True


def test_int4_integer_binary_equality():
    # int4 按整数二进制一致判断（binary_equal tobytes 哈希）
    ml_dtypes = pytest.importorskip("ml_dtypes")
    int4 = ml_dtypes.int4
    # 一致 → PASS
    a = np.array([1, 2, 3], dtype=int4)
    c = _cls("binary_equal")(a, np.array([1, 2, 3], dtype=int4), 0, "int4", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is True
    # 不一致 → FAIL
    c2 = _cls("binary_equal")(a, np.array([1, 9, 3], dtype=int4), 0, "int4", {})
    _p, _l, is_pass, _m = c2.compare()
    assert is_pass is False


def test_float_cross_dtype_rejected():
    c = _cls("binary_equal")(np.array([1.0], np.float32), np.array([1.0], np.float64), 0, "float32", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


def test_int_vs_float_rejected():
    c = _cls("binary_equal")(np.array([1], np.int32), np.array([1.0], np.float32), 0, "int32", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


def test_uint64_int32_precision_loss_rejected():
    # promote_types(int32, uint64) -> float64 -> 拒
    c = _cls("binary_equal")(np.array([1], np.uint64), np.array([1], np.int32), 0, "uint64", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


def test_empty_both_pass():
    c = _cls("binary_equal")(np.array([]), np.array([]), 0, "int32", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is True


def test_empty_one_fails():
    c = _cls("binary_equal")(np.array([1], np.int32), np.array([], np.int32), 0, "int32", {})
    _p, _l, is_pass, _m = c.compare()
    assert is_pass is False


def test_isclose_populates_metrics():
    c = _cls("isclose")(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 0, "float32",
                        {"rtol": [1e-3], "atol": [1e-8], "ptol": [1e-3]})
    _p, _l, is_pass, metrics = c.compare()
    assert metrics["standard"] == "isclose"
    assert metrics["pass"] is True
    assert "precision" in metrics
