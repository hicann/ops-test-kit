# tests/test_stat_rel_err.py
import numpy as np
import pytest

from ttk.core_modules.comparison.stat_rel_err import StatRelErrComparison


def _impl(actual, golden, dtype, threshold):
    """返回 EachCompareResult。threshold 必须显式给（resolve_tolerance 解析好的最终值）。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare_impl()


def _run(actual, golden, dtype, threshold):
    """返回 compare() 的 4-tuple。"""
    c = StatRelErrComparison(np.asarray(actual), np.asarray(golden), 0, dtype, {"threshold": threshold})
    return c.compare()


# —— 防线 1：mismatch 真值表全覆盖（13 cell）——
@pytest.mark.parametrize("a,g,expect_pass,mere_none", [
    # match：全非有限且一致 → PASS，mere=None
    (np.nan,  np.nan,   True,  True),
    (np.inf,  np.inf,   True,  True),
    (-np.inf, -np.inf,  True,  True),
    # finite/finite → mere 路径（mere 算出来，非 None）
    (1.0,  1.0, True,  False),    # mere=0 < th → PASS
    (1.0,  2.0, False, False),    # mere≈0.5 >> th → FAIL
    # mismatch → FAIL，mere=None
    (np.nan, 1.0,     False, True),
    (1.0,    np.nan,  False, True),
    (np.inf, 1.0,     False, True),
    (1.0,    np.inf,  False, True),
    (np.nan, np.inf,  False, True),    # 类型不同
    (np.inf, np.nan,  False, True),
    (np.inf, -np.inf, False, True),    # 符号不同
    (-np.inf, np.inf, False, True),
])
def test_mismatch_truth_table(a, g, expect_pass, mere_none):
    r = _impl([a], [g], "float32", 2**-13)
    assert r.is_pass == expect_pass
    assert (r.metrics["mere"] is None) == mere_none


# —— 防线 2：混合数组（mismatch 与数值 FAIL 的 diff_idx 分流）——
def test_mismatch_present_diff_idx_only_mismatch():
    r = _impl([np.nan, 1.0], [1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False and r.metrics["mere"] is None
    assert list(r.diff_index) == [0]

def test_mismatch_present_bad_finite_still_only_mismatch():
    r = _impl([np.nan, 99.0], [1.0, 1.0], "float32", 2**-13)
    assert r.metrics["mere"] is None
    assert list(r.diff_index) == [0]

def test_numeric_fail_diff_idx_worst_first_full():
    r = _impl([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is False
    assert list(r.diff_index) == [2, 1, 0]

def test_diff_index_full_log_caps_display():
    a = np.arange(1.0, 201.0); g = np.zeros(200)
    r = _impl(a, g, "float32", 2**-13)
    assert len(r.diff_index) == 200
    _p, log, _ip, _m = _run(a, g, "float32", 2**-13)
    # _log_diff_output prints "Index: ... RealIndex: ..." per row, capping at idx==100 (101 rows).
    # Count "RealIndex:" (unique per row) — "Index:" is a substring of "RealIndex:" so would double-count.
    assert log.count("RealIndex:") <= 101

# —— 防线 3：mere/mare/阈值公式数值校验 ——
def test_mere_mare_values():
    r = _impl([1.0, 2.0], [1.0, 4.0], "float32", 2**-13)
    assert r.metrics["mere"] == pytest.approx(0.25, abs=1e-4)
    assert r.metrics["mare"] == pytest.approx(0.5, abs=1e-4)
    assert r.metrics["threshold"] == 2**-13
    assert r.is_pass is False

def test_golden_zero_floor():
    r = _impl([1e-9], [0.0], "float32", 2**-13)
    assert r.metrics["mere"] == pytest.approx(0.01, abs=1e-4)
    assert r.is_pass is False

# —— 结构 / 边界 ——
def test_pass_clean_no_diff():
    r = _impl([1.0, 1.0], [1.0, 1.0], "float32", 2**-13)
    assert r.is_pass is True and r.diff_index is None

def test_all_nan_agree_pass():
    r = _impl([np.nan, np.nan], [np.nan, np.nan], "float32", 2**-13)
    assert r.is_pass is True
    assert r.metrics["mere"] is None and r.metrics["mare"] is None

# threshold 解析测试（resolve_tolerance 职责）在 T1 test_resolve_tolerance.py

def test_empty_arrays_both_pass():
    precision, _l, is_pass, _m = _run([], [], "float32", 2**-13)
    assert precision == "100%" and is_pass is True

def test_no_runtime_warning(recwarn):
    _impl([np.inf, np.nan, 1.0], [np.inf, np.nan, 1.0], "float32", 2**-13)
    bad = [w for w in recwarn.list
           if "invalid value" in str(w.message) or "divide" in str(w.message)]
    assert bad == []
