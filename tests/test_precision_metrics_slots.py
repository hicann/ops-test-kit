from ttk.core_modules.npu.op.profiling_structure import ComparisonResult, ProfilingReturnStructure
from ttk.core_modules.npu.op_api.profiling_structure import (
    ApiComparisonResult, ApiProfilingReturnStructure)
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure


def test_comparison_result_metrics_slot():
    r = ComparisonResult(None).set("d", "c", "b", "PASS", {"dyn": {"standard": "stat_rel_err"}})
    assert r.metrics == {"dyn": {"standard": "stat_rel_err"}}
    assert "metrics" in ComparisonResult.__slots__


def test_comparison_result_metrics_default():
    r = ComparisonResult(None).set("d", "c", "b", "PASS")  # 不传 metrics
    assert r.metrics == {}


def test_profiling_structure_precision_metrics_slot():
    s = ProfilingReturnStructure()
    assert hasattr(s, "precision_metrics")
    assert "precision_metrics" in ProfilingReturnStructure.__slots__


def test_api_comparison_result_metrics():
    r = ApiComparisonResult(None).set("95%", "PASS", {0: {"standard": "isclose"}})
    assert r.metrics == {0: {"standard": "isclose"}}
    ApiComparisonResult(None).set("95%", "PASS")  # default ok
    assert "metrics" in ApiComparisonResult.__slots__


def test_api_profiling_structure_precision_metrics_slot():
    s = ApiProfilingReturnStructure()
    assert hasattr(s, "precision_metrics")
    assert "precision_metrics" in ApiProfilingReturnStructure.__slots__


def test_framework_api_structure_precision_metrics_slot():
    s = FrameworkApiReturnStructure()
    assert hasattr(s, "precision_metrics")
    assert "precision_metrics" in FrameworkApiReturnStructure.__slots__
