# tests/test_cross_check.py
import numpy as np
import pytest
from ttk.core_modules.comparison.cross_check import CrossCheckComparison


def _make(output, golden, third_party, params):
    return CrossCheckComparison(output, golden, 0, "float32", params, third_party=third_party)


def test_pass_when_all_good():
    """三方误差一致 → PASS + metrics 结构。"""
    params = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5,
              "small_value": 2**-14, "small_value_atol": 2**-30,
              "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    g = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    c = _make(g.copy(), g.copy(), g.copy(), params)
    precision, log, is_pass, metrics = c.compare()
    assert is_pass is True
    assert metrics["standard"] == "cross_check"
    assert "config" in metrics and "result" in metrics   # 单元测试定位 metrics 形状（不只靠 Task 9 E2E）


def test_third_party_none_golden_failure():
    """third_party=None → GOLDEN_FAILURE。"""
    params = {"level": "L1", "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    c = _make(np.array([1.0]), np.array([1.0]), None, params)
    precision, log, is_pass, metrics = c.compare()
    assert precision == "GOLDEN_FAILURE"
    assert not is_pass
    assert metrics["reason"] == "third_party unavailable"


def test_ratio_exceeded():
    """NPU 误差远大于 third_party → ratio_exceeded FAIL。"""
    params = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5,
              "small_value": 2**-14, "small_value_atol": 2**-30,
              "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    g = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = g + 1.0  # NPU 误差大
    third = g + 0.001  # third_party 误差小
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert not is_pass
    assert "ratio exceeded" in metrics["reason"]


def test_safe_div_branches():
    """safe_div: max(den,1e-7) floor avoids div-by-0; nan/inf den -> inf."""
    assert CrossCheckComparison._safe_div(0, 0) == 1.0            # 0/0 -> 1 (both perfect, consistent)
    assert CrossCheckComparison._safe_div(1.0, 0) == 1e7          # num>0/den=0 -> num/1e-7 (no inf)
    assert CrossCheckComparison._safe_div(6, 3) == 2.0            # normal
    assert CrossCheckComparison._safe_div(0.1, float("nan")) == float("inf")  # nan den -> inf
    assert isinstance(CrossCheckComparison._safe_div(6, 3), float)   # 守护 float() 强转


def test_small_value_partition_pass():
    """防回归（spec §9）：golden 全小（<small_value）+ third_party 误差大 + NPU 精确 → small_ratio 小 → PASS。"""
    params = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5,
              "small_value": 2**-14, "small_value_atol": 2**-30,
              "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    g = np.full(100, 1e-15, dtype=np.float32)   # 全 < small_value（2**-14≈6e-5）→ 全 small 分区，large 空
    out = g.copy()                                # NPU 精确 → err_target=0
    third = g + 1e-8                              # third_party 误差大（>> small_value_atol=2**-30≈9.3e-10）→ err_third=100
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert is_pass
    assert metrics["result"]["small_err_cnt_target"] == 0
    assert metrics["result"]["small_err_cnt_third"] > 0
    assert metrics["result"]["mare"] is None   # 全小值域 large-empty → mare N/A（非 0.0 误导）


def test_small_value_exceeded_fail():
    """小值域 NPU 误差大 + third_party 精确 → small_ratio>2.0 → FAIL reason=small_value_exceeded（spec §9）。"""
    params = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5,
              "small_value": 2**-14, "small_value_atol": 2**-30,
              "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    g = np.full(100, 1e-15, dtype=np.float32)   # 全 small（< 2**-14）
    out = g + 1e-8                              # NPU 误差大（>> atol=2**-30）→ err_target=100
    third = g.copy()                            # third_party 精确 → err_third=0
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert not is_pass
    assert "small value ErrorCount ratio exceeded" in metrics["reason"]
    assert metrics["result"]["small_err_cnt_target"] == 100
    assert metrics["result"]["small_err_cnt_third"] == 0


def test_nan_inf_mismatch():
    """NaN/Inf 特殊位 mismatch → reason=NaN/Inf mismatch（special_ok 短路优先于 ratio）。"""
    params = {"level": "L1", "mare_ratio": 5.0, "mere_ratio": 1.5, "rmse_ratio": 1.5,
              "small_value": 2**-14, "small_value_atol": 2**-30,
              "legacy": {"rtol": None, "ptol": None, "atol": 1e-8}}
    g = np.array([1.0, np.nan], dtype=np.float32)
    out = np.array([1.0, 1.0], dtype=np.float32)   # golden[1]=nan 但 NPU 非 nan → special 位 mismatch
    third = np.array([1.0, np.nan], dtype=np.float32)
    c = _make(out, g, third, params)
    precision, log, is_pass, metrics = c.compare()
    assert not is_pass
    assert metrics["reason"] == "NaN/Inf mismatch"
