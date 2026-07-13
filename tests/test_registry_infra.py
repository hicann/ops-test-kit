import numpy as np
import pytest

from ttk.core_modules.comparison.registry import (
    EachCompareResult, ComparisonBase, _to_numpy, register_comparison
)


def test_each_compare_result_defaults():
    r = EachCompareResult(1)
    assert r.precision == 1
    assert r.diff_index is None
    assert r.is_pass is False
    assert r.log == ""
    assert r.standard == ""
    assert r.metrics == {}
    assert r.error_info is None
    # distinct default dicts (not shared)
    assert EachCompareResult(1).metrics is not EachCompareResult(1).metrics


def test_to_numpy_passthrough():
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert _to_numpy(a) is a


def test_to_numpy_torch_float():
    torch = pytest.importorskip("torch")
    t = torch.tensor([1.0, 2.0], dtype=torch.float32)
    out = _to_numpy(t)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [1.0, 2.0])


def test_to_numpy_torch_bfloat16():
    torch = pytest.importorskip("torch")
    ml_dtypes = pytest.importorskip("ml_dtypes")
    t = torch.tensor([256.0, 512.0], dtype=torch.bfloat16)
    out = _to_numpy(t)
    assert out.dtype == ml_dtypes.bfloat16
    assert list(out.astype(np.float32)) == [256.0, 512.0]


# 一个最小 ComparisonBase 子类用于测 compare() 4-tuple + _check_empty
@register_comparison("__test_dummy")
class _Dummy(ComparisonBase):
    STANDARD_NAME = "dummy"
    def compare_impl(self):
        return EachCompareResult(0.5, is_pass=True, standard="dummy",
                                 metrics={"k": 1})


def test_compare_returns_4tuple_and_unifies_numpy():
    torch = pytest.importorskip("torch")
    c = _Dummy(torch.tensor([1.0]), torch.tensor([1.0]), 0, "float32", {})
    precision, log, is_pass, metrics = c.compare()
    assert precision == "50.0%"
    assert is_pass is True
    assert metrics == {"k": 1}
    assert isinstance(c.output, np.ndarray)  # 入口已转 numpy（torch.tensor → np.ndarray）


def test_check_empty_both_empty():
    c = _Dummy(np.array([]), np.array([]), 0, "float32", {})
    r = c._check_empty()
    assert r.is_pass is True and r.precision == 1


def test_check_empty_one_empty():
    c = _Dummy(np.array([1.0]), np.array([]), 0, "float32", {})
    r = c._check_empty()
    assert r.is_pass is False and r.precision == 0


def test_check_empty_both_nonempty():
    c = _Dummy(np.array([1.0]), np.array([1.0]), 0, "float32", {})
    assert c._check_empty() is None
