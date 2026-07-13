# tests/test_resolve_convergence.py
import numpy as np
import pytest
from ttk.core_modules.comparison.resolve import resolve_tolerance, ResolvedStandard


def test_legacy_injection():
    """legacy（precision_tolerances/absolute_precision）注入 params["legacy"]。"""
    standards = resolve_tolerance(None, [(0.1, 0.01)], 1e-8, ["float32"], None)
    assert standards[0].params["legacy"]["rtol"] == 0.1
    assert standards[0].params["legacy"]["ptol"] == 0.01
    assert standards[0].params["legacy"]["atol"] == 1e-8


def test_cross_check_level_preset():
    """cross_check level → ratio 预设。"""
    tol = {"float32": {"standard": "cross_check", "level": "L1"}}
    standards = resolve_tolerance(tol, None, 1e-8, ["float32"], None)
    s = standards[0]
    assert s.token == "cross_check"
    assert s.params["level"] == "L1"
    assert s.params["mare_ratio"] == 5.0
    assert s.params["mere_ratio"] == 1.5
    assert s.params["rmse_ratio"] == 1.5
    assert "small_value" in s.params
    assert "small_value_atol" in s.params


def test_cross_check_unknown_level_raises():
    """未知 level → ValueError（TOLERANCE_INVALID）。"""
    tol = {"float32": {"standard": "cross_check", "level": "L3"}}
    with pytest.raises(ValueError, match="unknown level"):
        resolve_tolerance(tol, None, 1e-8, ["float32"], None)


def test_cross_check_defaults_to_L1_when_no_level_no_ratios():
    """无 level + 无 ratio → 默认 L1（不再 raise）。"""
    tol = {"float32": {"standard": "cross_check"}}
    result = resolve_tolerance(tol, None, 1e-8, ["float32"], None)
    params = result[0].params
    assert params["level"] == "L1"
    assert "mare_ratio" in params
    assert "mere_ratio" in params
    assert "rmse_ratio" in params


def test_cross_check_unsupported_dtype_raises():
    """unsupported dtype → ValueError（不 KeyError）。"""
    tol = {"float64": {"standard": "cross_check", "level": "L1"}}
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_tolerance(tol, None, 1e-8, ["float64"], None)


def test_cross_check_small_value_override():
    """small_value override 优先于 dtype 表默认。"""
    tol = {"float32": {"standard": "cross_check", "level": "L1", "small_value": 1e-4}}
    standards = resolve_tolerance(tol, None, 1e-8, ["float32"], None)
    assert standards[0].params["small_value"] == 1e-4


def test_cross_check_level_L0_L2():
    """L0/L2 预设 → 对应 ratio（spec §9 level 矩阵）。"""
    for level, (mare, mere, rmse) in [("L0", (10.0, 2.0, 2.0)), ("L2", (2.0, 1.2, 1.2))]:
        tol = {"float32": {"standard": "cross_check", "level": level}}
        s = resolve_tolerance(tol, None, 1e-8, ["float32"], None)[0]
        assert s.params["mare_ratio"] == mare
        assert s.params["mere_ratio"] == mere
        assert s.params["rmse_ratio"] == rmse


def test_cross_check_explicit_ratio_override():
    """显式 ratio 覆盖 level 预设。"""
    tol = {"float32": {"standard": "cross_check", "level": "L1", "mare_ratio": 3.0}}
    s = resolve_tolerance(tol, None, 1e-8, ["float32"], None)[0]
    assert s.params["mare_ratio"] == 3.0      # override
    assert s.params["mere_ratio"] == 1.5      # level 预设保留
    assert s.params["rmse_ratio"] == 1.5


def test_cross_check_no_level_all_ratios():
    """无 level + 全 ratio → L1 标签 + 自定 ratio。"""
    tol = {"float32": {"standard": "cross_check",
                       "mare_ratio": 3.0, "mere_ratio": 1.0, "rmse_ratio": 1.0}}
    s = resolve_tolerance(tol, None, 1e-8, ["float32"], None)[0]
    assert s.params["level"] == "L1"          # 无 level 时 L1 标签
    assert s.params["mare_ratio"] == 3.0
    assert s.params["mere_ratio"] == 1.0
    assert s.params["rmse_ratio"] == 1.0


def test_isclose_reads_legacy_rtol():
    """C1: isclose 从 legacy 子 dict 读 rtol（非顶层）。"""
    import ttk.core_modules.comparison.is_close  # noqa: F401 — 触发 @register_comparison
    from ttk.core_modules.comparison.registry import ComparisonRegister
    from ttk.core_modules.comparison.resolve import resolve_tolerance
    standards = resolve_tolerance(None, [(0.001, 0.001)], 1e-9, ["float32"], None)
    cls = ComparisonRegister.registry["isclose"]
    out = np.array([1.0, 2.0]); gold = np.array([1.0, 2.0])
    c = cls(out, gold, 0, "float32", standards[0].params)
    # rtol/ptol/atol 应来自 legacy 子 dict（被 _get_rtol 经 get(idx) 取出）
    assert c.rtol == [0.001]
    assert c.atol == [1e-9]


def test_cosine_reads_legacy_rtol():
    """C1: cosine 从 legacy 子 dict 读 rtol（cosine 只读 rtol，无 ptol/atol）。"""
    import ttk.core_modules.comparison.cosine_similarity  # noqa: F401 — 触发 @register_comparison
    from ttk.core_modules.comparison.registry import ComparisonRegister
    from ttk.core_modules.comparison.resolve import resolve_tolerance
    standards = resolve_tolerance(None, [(0.01, 0.001)], 1e-9, ["float32"], "cosine")
    cls = ComparisonRegister.registry["cosine"]
    out = np.array([1.0, 2.0]); gold = np.array([1.0, 2.0])
    c = cls(out, gold, 0, "float32", standards[0].params)
    assert c.rtol == [0.01]


def test_legacy_injection_absolute_precision_list():
    """absolute_precision 传 list（production e2e 路径形式）→ atol 按 idx 取（覆盖 list 分支）。"""
    standards = resolve_tolerance(None, [(0.1, 0.01), (0.2, 0.02)], [1e-8, 1e-9],
                                 ["float32", "float32"], None)
    assert standards[0].params["legacy"]["atol"] == 1e-8
    assert standards[1].params["legacy"]["atol"] == 1e-9


def test_cosine_legacy_not_reading_top_level():
    """C1: cosine 改读 legacy 后，顶层 rtol 不再被读 → 回落默认 0.01（验顶层 rtol 被忽略，真 failing test）。"""
    import ttk.core_modules.comparison.cosine_similarity  # noqa: F401 — 触发 @register_comparison
    from ttk.core_modules.comparison.registry import ComparisonRegister
    cls = ComparisonRegister.registry["cosine"]
    out = np.array([1.0, 2.0]); gold = np.array([1.0, 2.0])
    c = cls(out, gold, 0, "float32", {"rtol": 0.05})   # 顶层 rtol=0.05（无 legacy 子 dict）
    assert c._get_rtol(np.float32) == 0.01   # 改读 legacy 后顶层 0.05 被忽略 → None → 默认 0.01
