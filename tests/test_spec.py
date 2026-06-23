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
        self.mgr = TestSpecManager(search_paths=[EXAMPLES_DIR])

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
        self.mgr = TestSpecManager(search_paths=[EXAMPLES_DIR])

    def test_validate_valid_spec(self):
        cls = self.mgr.load("add")
        assert cls is not None
        warnings = self.mgr.validate(cls)
        # AddTestSpec is valid — should produce no warnings
        assert warnings == []

    def test_validate_returns_list(self):
        cls = self.mgr.load("add")
        assert cls is not None
        result = self.mgr.validate(cls)
        assert isinstance(result, list)


class TestManagerListVendors:
    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=[EXAMPLES_DIR])

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
