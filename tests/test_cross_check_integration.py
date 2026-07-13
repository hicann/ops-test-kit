# tests/test_cross_check_integration.py
"""端到端：spec.tolerance → resolve → compare(third_parties=) → metrics。"""
import numpy as np
from ttk.core_modules.comparison import compare
from ttk.core_modules.comparison.resolve import resolve_tolerance


def test_cross_check_e2e_pass():
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}
    standards = resolve_tolerance(tolerance, None, 1e-8, ["float32"], None)
    g = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    outputs = [g.copy()]
    goldens = [g.copy()]
    third_parties = [g.copy()]
    precision, log, is_pass, metrics = compare(
        outputs, goldens, ("float32",), standards=standards, third_parties=third_parties)
    assert is_pass
    assert metrics[0]["standard"] == "cross_check"
    assert "config" in metrics[0] and "result" in metrics[0]


def test_cross_check_e2e_count_mismatch():
    """third_parties 少于 outputs → COMPARE_FAILURE（整体 return）。

    设计(comparison.py:48): third_parties < outputs（不够逐个比对）才报错; 多于则忽略多余。"""
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}
    standards = resolve_tolerance(tolerance, None, 1e-8, ["float32", "float32"], None)
    precision, log, is_pass, metrics = compare(
        [np.array([1.0]), np.array([2.0])], [np.array([1.0]), np.array([2.0])],
        ("float32", "float32"),
        standards=standards,
        third_parties=[np.array([1.0])])  # 1 < 2
    assert precision == "COMPARE_FAILURE"
    assert not is_pass
    assert metrics["reason"] == "third_party count != outputs"


def test_cross_check_e2e_no_third_party():
    """cross_check + third_parties=None → GOLDEN_FAILURE。"""
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}
    standards = resolve_tolerance(tolerance, None, 1e-8, ["float32"], None)
    precision, log, is_pass, metrics = compare(
        [np.array([1.0])], [np.array([1.0])], ("float32",), standards=standards)
    assert precision == "GOLDEN_FAILURE"
    assert not is_pass
