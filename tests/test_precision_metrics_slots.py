from ttk.core_modules.npu.op.profiling_structure import ComparisonResult, ProfilingReturnStructure
from ttk.core_modules.npu.op_api.profiling_structure import (
    ApiComparisonResult, ApiProfilingReturnStructure)
from ttk.core_modules.framework_api.result import FrameworkApiReturnStructure
from ttk.core_modules.geir.geir_struct import GeirReturnStructure


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


def test_geir_structure_precision_metrics_slot():
    s = GeirReturnStructure()
    assert hasattr(s, "precision_metrics")
    assert s.precision_metrics == {}
    titles = GeirReturnStructure.get_titles()
    assert "precision_metrics" in titles
    assert titles.index("precision_metrics") == titles.index("xpu_metrics") + 1


def test_geir_structure_pick_data_includes_precision_metrics():
    s = GeirReturnStructure()
    s.precision_metrics = {"cst": {0: {"standard": "stat_rel_err"}}}
    titles = GeirReturnStructure.get_titles()
    row = s.pick_data(titles)
    assert row[titles.index("precision_metrics")] == {"cst": {0: {"standard": "stat_rel_err"}}}
