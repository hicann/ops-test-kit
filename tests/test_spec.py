"""Test cases for ttk.test_spec — discovery, loading, inspection, and validation."""

import ast
import sys
import pytest

from ttk.test_spec.manager import TestSpecManager
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
                '__spec__ = {"bad_op": "BadSpec"}\n'
            )
            mgr = TestSpecManager((d,))
            with pytest.raises(InvalidSpecError):
                mgr.load("bad_op")


class TestManagerListVendors:
    def setup_method(self):
        self.mgr = TestSpecManager(search_paths=(EXAMPLES_DIR,))

    def test_list_vendors_dict(self):
        """03_third_party.py SoftmaxMultiVendorSpec: dict third_party → [torch, tf]."""
        cls = self.mgr.load("softmax_multi_vendor")
        assert cls is not None, "softmax_multi_vendor 未加载"
        assert self.mgr.list_vendors(cls) == ["torch", "tf"]

    def test_list_vendors_string(self):
        """03_third_party.py has SoftmaxSimpleSpec with string third_party."""
        cls = self.mgr.load("softmax_simple")
        if cls is not None and self.mgr.has(cls, "third_party"):
            vendors = self.mgr.list_vendors(cls)
            assert vendors == ["torch"]


# ---------------------------------------------------------------------------
# Example specs discoverability (regression test for _ prefix bug)
# ---------------------------------------------------------------------------


class TestSingleFilePluginPath:
    """Regression test: --plugin-path pointing to a specific .py file (not a dir)
    must be loaded. Previously _iter_py_files skipped non-dir paths entirely."""

    def test_load_spec_from_single_file(self):
        """SpecLoader should load a spec when given a .py file path directly."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            spec_file = Path(d) / "my_op.py"
            spec_file.write_text(
                'import numpy\n'
                'class MyOpTestSpec:\n'
                '    def golden(x, **kwargs):\n'
                '        return [numpy.abs(x)]\n'
                '__spec__ = {"my_op": "MyOpTestSpec"}\n'
            )
            loader = SpecLoader([str(spec_file)])
            cls = loader.load("my_op")
            assert cls is not None, "spec not found when --plugin-path is a .py file"
            assert cls.__name__ == "MyOpTestSpec"

    def test_load_spec_from_single_file_naming_convention(self):
        """Naming convention (class name derived from op name) also works for single file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            spec_file = Path(d) / "relu.py"
            spec_file.write_text(
                'import numpy\n'
                'class ReluTestSpec:\n'
                '    def golden(x, **kwargs):\n'
                '        return [numpy.maximum(x, 0)]\n'
            )
            loader = SpecLoader([str(spec_file)])
            cls = loader.load("relu")
            assert cls is not None
            assert cls.__name__ == "ReluTestSpec"

    def test_single_file_and_dir_mixed(self):
        """A mix of file path and dir path should both be searched."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            d_path = Path(d)
            (d_path / "dir_spec.py").write_text(
                'import numpy\n'
                'class DirOpTestSpec:\n'
                '    def golden(x, **kwargs):\n'
                '        return [x]\n'
                '__spec__ = {"dir_op": "DirOpTestSpec"}\n'
            )
            file_spec = d_path / "file_spec.py"
            file_spec.write_text(
                'import numpy\n'
                'class FileOpTestSpec:\n'
                '    def golden(x, **kwargs):\n'
                '        return [x]\n'
                '__spec__ = {"file_op": "FileOpTestSpec"}\n'
            )
            loader = SpecLoader([str(d_path), str(file_spec)])
            assert loader.load("dir_op") is not None
            assert loader.load("file_op") is not None


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


# ---------------------------------------------------------------------------
# Lazy AST discovery (Task LA-T1): __spec__ value as string + 0-exec index
# ---------------------------------------------------------------------------


class TestLazyAstDiscovery:
    """Guards for the AST-based lazy loader (G1-G11)."""

    def _write(self, tmp_path, name, body):
        f = tmp_path / name
        f.write_text(body, encoding="utf-8")
        return f

    def test_g1_top_placement_spec(self, tmp_path):  # G1: __spec__ 写顶部
        self._write(tmp_path, "s.py",
            '__spec__ = {"op": "Cls"}\n'             # 顶部,类之前
            'class Cls:\n'
            '    def __call__(self): return [1]\n')
        cls = SpecLoader([tmp_path]).load("op")
        assert cls is not None and cls.__name__ == "Cls"

    def test_g2_class_object_value_rejected(self, tmp_path):  # G2
        self._write(tmp_path, "s.py",
            'class Cls: pass\n'
            '__spec__ = {"op": Cls}\n')              # 旧类对象写法
        with pytest.raises(ValueError, match="only string class names"):
            SpecLoader([tmp_path]).load("op")

    def test_g2b_non_str_constant_rejected(self, tmp_path):  # G2b: int/None
        self._write(tmp_path, "s.py",
            '__spec__ = {"op": 123}\nclass Cls: pass\n')
        with pytest.raises(ValueError, match="only string class names"):
            SpecLoader([tmp_path]).load("op")

    def test_g4_lazy_one_exec_not_n(self, tmp_path):  # G4 核心
        for i in range(3):
            self._write(tmp_path, f"f{i}.py",
                f'class C{i}:\n    def __call__(self): return [{i}]\n'
                f'__spec__ = {{"op{i}": "C{i}"}}\n')
        loader = SpecLoader([tmp_path])
        import unittest.mock as _m
        with _m.patch.object(loader, "_import_file", wraps=loader._import_file) as spy:
            loader.load("op0")
        assert spy.call_count == 1                  # 不是 3

    def test_g7_per_file_cache(self, tmp_path):  # G7（回归：per-file cache 锁定）
        # 注:当前 _file_cache 也 memoize dict,故本测在旧代码上也 ==1（非 bug-catcher）;
        # 锁定新 per-file module cache 行为,防重构退化。
        self._write(tmp_path, "s.py",
            '__spec__ = {"a": "C", "b": "C"}\nclass C: pass\n')
        loader = SpecLoader([tmp_path])
        import unittest.mock as _m
        with _m.patch.object(loader, "_import_file", wraps=loader._import_file) as spy:
            loader.load("a"); loader.load("b")
        assert spy.call_count == 1                  # 同文件,exec 1 次

    def test_g9_index_built_once(self, tmp_path):  # G9 memoization(RED:旧代码 0 次 ast.parse)
        for i in range(3):
            self._write(tmp_path, f"f{i}.py",
                f'__spec__ = {{"op{i}": "C{i}"}}\nclass C{i}: pass\n')
        loader = SpecLoader([tmp_path])
        import unittest.mock as _m
        with _m.patch("ttk.test_spec.loader.ast.parse", wraps=ast.parse) as spy:
            loader.load("op0"); loader.load("op1")
        assert spy.call_count == 3                  # 精确:建一次 + memoized

    def test_g10_classname_collision_first_wins(self, tmp_path):  # G10
        d1, d2 = tmp_path / "a", tmp_path / "b"
        d1.mkdir(); d2.mkdir()
        self._write(d1, "f.py", '__spec__={"op":"C"}\nclass C:\n    TAG=1\n')
        self._write(d2, "f.py", '__spec__={"op":"C"}\nclass C:\n    TAG=2\n')
        cls = SpecLoader([d1, d2]).load("op")       # first-wins (d1)
        assert cls.TAG == 1

    def test_g11_clear_cache_rebuilds(self, tmp_path):  # G11
        self._write(tmp_path, "s.py", '__spec__={"op":"C"}\nclass C: pass\n')
        loader = SpecLoader([tmp_path])
        assert loader.load("op") is not None
        loader.clear_cache()
        assert loader.load("op") is not None        # 重建后仍可 load

    # 回归守卫
    def test_g5_naming_convention_fallback(self, tmp_path):  # G5
        self._write(tmp_path, "s.py", 'class FooTestSpec:\n    def __call__(self): return [1]\n')
        assert SpecLoader([tmp_path]).load("foo") is not None

    def test_g8_mark_source_both_paths(self, tmp_path):  # G8
        # __spec__ 路
        self._write(tmp_path, "a.py", '__spec__={"op1":"C1"}\nclass C1: pass\n')
        # 命名约定路
        self._write(tmp_path, "b.py", 'class FooTestSpec: pass\n')
        loader = SpecLoader([tmp_path])
        c1 = loader.load("op1"); c2 = loader.load("foo")
        assert hasattr(c1, "__ttk_spec_file__") and hasattr(c2, "__ttk_spec_file__")
