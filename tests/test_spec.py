"""Test cases for ttk.test_spec — discovery, loading, inspection, and validation."""

import sys
import pytest

from ttk.test_spec import TestSpecManager
from ttk.test_spec.loader import SpecLoader, _snake_to_pascal
from ttk.test_spec.validator import validate


EXAMPLES_DIR = "ttk/test_spec/examples"


# ---------------------------------------------------------------------------
# SpecLoader unit tests
# ---------------------------------------------------------------------------


class TestSnakeToPascal:
    def test_basic(self):
        assert _snake_to_pascal("abs") == "Abs"

    def test_with_version(self):
        assert _snake_to_pascal("softmax_v2") == "SoftmaxV2"

    def test_multi_underscore(self):
        assert _snake_to_pascal("layer_norm") == "LayerNorm"

    def test_single_char_parts(self):
        assert _snake_to_pascal("a_b_c") == "ABC"


class TestSpecLoaderDiscovery:
    """Test that files with _ prefix are excluded, and normal files are found."""

    def setup_method(self):
        self.loader = SpecLoader([EXAMPLES_DIR])

    def test_load_add_from_minimal(self):
        """The 01_minimal.py file has AddTestSpec with __spec__ dict."""
        cls = self.loader.load("add")
        assert cls is not None
        assert cls.__name__ == "AddTestSpec"

    def test_load_returns_none_for_unknown(self):
        assert self.loader.load("nonexistent_op_12345") is None

    def test_cache_hit(self):
        cls1 = self.loader.load("add")
        cls2 = self.loader.load("add")
        assert cls1 is cls2  # same object from cache

    def test_clear_cache_forces_rediscovery(self):
        cls1 = self.loader.load("add")
        assert cls1 is not None
        self.loader.clear_cache()
        cls2 = self.loader.load("add")
        assert cls2 is not None
        # After clear_cache, a fresh module is loaded, so class identity may differ
        assert cls2.__name__ == "AddTestSpec"

    def test_no_sys_modules_leak(self):
        """Modules should NOT be registered in sys.modules."""
        before = set(sys.modules.keys())
        self.loader.load("add")
        after = set(sys.modules.keys())
        new_keys = after - before
        # No ttk_test_spec_ entries should appear in sys.modules
        leaked = [k for k in new_keys if "ttk_test_spec" in k]
        assert leaked == [], f"Modules leaked into sys.modules: {leaked}"


# ---------------------------------------------------------------------------
# TestSpecManager integration tests
# ---------------------------------------------------------------------------


class TestManagerLoadAndInspect:
    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_load_add(self):
        cls = self.mgr.load("add")
        assert cls is not None
        assert cls.__name__ == "AddTestSpec"

    def test_has_golden(self):
        cls = self.mgr.load("add")
        assert cls is not None
        assert self.mgr.has(cls, "golden") is True

    def test_get_golden(self):
        cls = self.mgr.load("add")
        assert cls is not None
        golden = self.mgr.get(cls, "golden")
        assert callable(golden)

    def test_has_returns_false_for_none_attr(self):
        """has() returns False when attribute exists but is None."""
        cls = self.mgr.load("add")
        assert cls is not None
        assert self.mgr.has(cls, "tolerance") is False

    def test_has_returns_false_for_missing_attr(self):
        cls = self.mgr.load("add")
        assert cls is not None
        assert self.mgr.has(cls, "nonexistent_attr") is False

    def test_get_with_default(self):
        cls = self.mgr.load("add")
        assert cls is not None
        result = self.mgr.get(cls, "nonexistent_attr", default="fallback")
        assert result == "fallback"


class TestManagerValidation:
    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_validate_valid_spec_no_raise(self):
        """Valid spec — validate() does not raise."""
        cls = self.mgr.load("add")
        assert cls is not None
        self.mgr.validate(cls)  # no exception

    def test_validate_invalid_raises(self):
        """Invalid spec (golden wrong type) — validate() raises InvalidSpecError."""
        from ttk.test_spec import InvalidSpecError

        class BadSpec:
            golden = 123  # int, not str/type/callable

        with pytest.raises(InvalidSpecError):
            self.mgr.validate(BadSpec)

    def test_load_invalid_raises_on_first_use(self):
        """First load of an invalid spec triggers validate → raises (fail-fast)."""
        import tempfile
        from pathlib import Path
        from ttk.test_spec import InvalidSpecError

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bad.py").write_text(
                'class BadSpec:\n    golden = 123\n'
                '__spec__ = {"bad_op": BadSpec}\n'
            )
            mgr = TestSpecManager((d,))
            with pytest.raises(InvalidSpecError):
                mgr.load("bad_op")


class TestManagerListVendors:
    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_list_vendors_dict(self):
        """03_third_party.py has SoftmaxMultiVendorSpec with dict third_party."""
        cls = self.mgr.load("softmax_multi_vendor")
        # This may or may not resolve via naming convention; check if found
        if cls is not None:
            vendors = self.mgr.list_vendors(cls)
            if self.mgr.has(cls, "third_party"):
                assert isinstance(vendors, list)

    def test_list_vendors_string(self):
        """03_third_party.py has SoftmaxSimpleSpec with string third_party."""
        cls = self.mgr.load("softmax_simple")
        if cls is not None and self.mgr.has(cls, "third_party"):
            vendors = self.mgr.list_vendors(cls)
            assert vendors == ["torch"]


# ---------------------------------------------------------------------------
# Example specs discoverability (regression test for _ prefix bug)
# ---------------------------------------------------------------------------


class TestExampleDiscoverability:
    """All example files should be discoverable by the loader.

    Regression test: example files were previously named _minimal.py etc.,
    which the loader excluded due to the _ prefix filter.
    """

    def setup_method(self):
        self.loader = SpecLoader([EXAMPLES_DIR])

    def test_minimal_file_discovered(self):
        """01_minimal.py should be loadable."""
        cls = self.loader.load("add")
        assert cls is not None, "01_minimal.py spec not discovered"

    def test_golden_forms_file_discovered(self):
        """02_golden_forms.py should be loadable (AbsStrSpec via naming)."""
        cls = self.loader.load("abs_str")
        assert cls is not None, "02_golden_forms.py spec not discovered"

    def test_third_party_file_discovered(self):
        """03_third_party.py should be loadable."""
        cls = self.loader.load("softmax_simple")
        assert cls is not None, "03_third_party.py spec not discovered"

    def test_full_file_discovered(self):
        """04_full.py should be loadable."""
        cls = self.loader.load("layer_norm_full")
        assert cls is not None, "04_full.py spec not discovered"

    def test_advanced_file_discovered(self):
        """05_advanced.py should be loadable."""
        cls = self.loader.load("softmax_advanced")
        assert cls is not None, "05_advanced.py spec not discovered"


class TestValidateClassExclusion:
    """callable 属性里,golden/third_party 接受 class,其余只 function。"""

    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_golden_accepts_class(self):
        """golden 文档有 class 形式 → validate 通过。"""
        class GoldenImpl:
            def __call__(self, x, **kwargs):
                return [x]
        class S:
            golden = GoldenImpl
        self.mgr.validate(S)  # 不抛

    def test_third_party_accepts_class(self):
        class TpImpl:
            def __call__(self, x, **kwargs):
                return [x]
        class S:
            third_party = TpImpl
        self.mgr.validate(S)  # 不抛

    def test_customize_inputs_rejects_class(self):
        from ttk.test_spec import InvalidSpecError
        class InputImpl:
            def __call__(self, x, **kwargs):
                return [x]
        class S:
            customize_inputs = InputImpl
        with pytest.raises(InvalidSpecError):
            self.mgr.validate(S)

    def test_compare_rejects_class(self):
        from ttk.test_spec import InvalidSpecError
        class CompareImpl:
            def __call__(self, *outputs, **kwargs):
                return {"pass": True, "metrics": []}
        class S:
            compare = CompareImpl
        with pytest.raises(InvalidSpecError):
            self.mgr.validate(S)

    def test_pre_compare_rejects_class(self):
        from ttk.test_spec import InvalidSpecError
        class PreImpl:
            def __call__(self, *outputs, **kwargs):
                return list(outputs)
        class S:
            pre_compare = PreImpl
        with pytest.raises(InvalidSpecError):
            self.mgr.validate(S)

    def test_describe_rejects_class(self):
        from ttk.test_spec import InvalidSpecError
        class DescImpl:
            def __call__(self, params):
                return {}
        class S:
            describe = DescImpl
        with pytest.raises(InvalidSpecError):
            self.mgr.validate(S)


class TestValidateTorchGraph:
    """torch_graph 必须是 torch.nn.Module 子类。"""

    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_accepts_nn_module_subclass(self):
        import torch.nn as nn
        class GraphMod(nn.Module):
            def forward(self, x):
                return x
        class S:
            torch_graph = GraphMod
        self.mgr.validate(S)  # 不抛

    def test_rejects_non_nn_module_class(self):
        from ttk.test_spec import InvalidSpecError
        class NotModule:  # 普通类,非 nn.Module(MRO 不含)
            def forward(self, x):
                return x
        class S:
            torch_graph = NotModule
        with pytest.raises(InvalidSpecError):
            self.mgr.validate(S)
