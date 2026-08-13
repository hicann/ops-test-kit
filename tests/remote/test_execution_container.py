import pytest
import numpy as np

from ttk.remote.server.execution_container import (
    format_device,
    _FRAMEWORK_RESERVED,
    bind_params,
    UnknownParamError,
    to_device,
    resolve_callable,
)


class TestFrameworkReserved:
    def test_device_is_reserved(self):
        assert "device" in _FRAMEWORK_RESERVED


class TestDeviceFormatting:
    def test_torch_format(self):
        assert format_device("torch", {"torch_lib": "cuda"}, 0) == "cuda:0"

    def test_tf_format(self):
        assert format_device("tf", {"torch_lib": "cuda", "tf_device_type": "GPU"}, 0) == "/device:GPU:0"

    def test_cpu_format(self):
        assert format_device("torch", {}, "cpu") == "cpu"


class TestMatchParamsV1:
    def test_single_tensors(self):
        from ttk.remote.server.execution_container import match_params_v1
        import numpy as np
        schema = [
            {"name": "x", "index": 0},
            {"name": "y", "index": 1},
        ]
        flat = [np.array([1.0]), np.array([2.0])]
        result = match_params_v1(schema, flat)
        assert result["x"] is flat[0]
        assert result["y"] is flat[1]

    def test_tensor_list(self):
        from ttk.remote.server.execution_container import match_params_v1
        import numpy as np
        schema = [
            {"name": "x", "indices": [0, 1]},
            {"name": "y", "index": 2},
        ]
        flat = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
        result = match_params_v1(schema, flat)
        assert result["x"] == [flat[0], flat[1]]
        assert result["y"] is flat[2]

    def test_none_optional(self):
        from ttk.remote.server.execution_container import match_params_v1
        import numpy as np
        schema = [
            {"name": "x", "index": 0},
            {"name": "z", "index": None},
            {"name": "y", "index": 1},
        ]
        flat = [np.array([1.0]), np.array([2.0])]
        result = match_params_v1(schema, flat)
        assert result["x"] is flat[0]
        assert result["z"] is None
        assert result["y"] is flat[1]

    def test_empty_schema(self):
        from ttk.remote.server.execution_container import match_params_v1
        result = match_params_v1([], [])
        assert result == {}

    def test_single_element_tensor_list(self):
        """Critical boundary: single-element tensor list uses indices not index."""
        from ttk.remote.server.execution_container import match_params_v1
        import numpy as np
        schema = [
            {"name": "x", "indices": [0]},
        ]
        flat = [np.array([1.0, 2.0])]
        result = match_params_v1(schema, flat)
        assert result["x"] == [flat[0]]  # list, not bare tensor


class TestBindParams:
    """bind_params: name-driven binding, * = calling style, unknown = raise (spec §7.2-7.4)."""

    # --- positional vs keyword by * ---

    def test_no_star_inputs_are_positional(self):
        class Impl:
            def __call__(self, x1, x2):
                return (x1, x2)
        args, kwargs = bind_params(Impl.__call__, {"x1": 1, "x2": 2})
        assert args == [1, 2]
        assert kwargs == {}

    def test_after_star_are_keyword(self):
        class Impl:
            def __init__(self, *, axis=-1):
                self.axis = axis
        args, kwargs = bind_params(Impl.__init__, {"axis": 7})
        assert args == []
        assert kwargs == {"axis": 7}

    def test_mixed_positional_and_keyword(self):
        class Impl:
            def __init__(self, x1, x2, *, axis=-1):
                pass
        args, kwargs = bind_params(Impl.__init__, {"x1": 1, "x2": 2, "axis": 9})
        assert args == [1, 2]
        assert kwargs == {"axis": 9}

    # --- device reserved param ---

    def test_device_reserved_always_injected_to_kwargs(self):
        class Impl:
            def __init__(self, x1, *, device, axis=-1):
                pass
        args, kwargs = bind_params(Impl.__init__, {"x1": 1, "axis": 3}, device="cuda:0")
        assert args == [1]
        assert kwargs == {"axis": 3, "device": "cuda:0"}

    def test_device_none_not_injected(self):
        class Impl:
            def __call__(self, x1):
                return (x1,)
        args, kwargs = bind_params(Impl.__call__, {"x1": 5}, device=None)
        assert args == [5]
        assert "device" not in kwargs

    # --- unknown name raises ---

    def test_unknown_name_raises(self):
        class Bad:
            def __call__(self, x1, bias):
                pass
        with pytest.raises(UnknownParamError):
            bind_params(Bad.__call__, {"x1": 1})  # 'bias' is neither input nor attr

    def test_unknown_name_message_names_param(self):
        class Bad:
            def __call__(self, x1, bogus):
                pass
        try:
            bind_params(Bad.__call__, {"x1": 1})
        except UnknownParamError as e:
            assert "bogus" in str(e)
        else:
            pytest.fail("expected UnknownParamError")

    # --- **kwargs absorbs leftover; self skipped ---

    def test_var_keyword_absorbs_leftover(self):
        class Impl:
            def __call__(self, x1, **kw):
                return (x1, kw)
        args, kwargs = bind_params(Impl.__call__, {"x1": 1, "axis": 2, "extra": 3})
        assert args == [1]
        assert kwargs == {"axis": 2, "extra": 3}

    def test_self_is_skipped(self):
        class Impl:
            def __call__(self, x1):
                return (x1,)
        # self not in name_to_value -> must NOT raise (it is skipped)
        args, kwargs = bind_params(Impl.__call__, {"x1": 9})
        assert args == [9]

    # --- unconsumed leftover warns (no **kwargs) ---

    def test_unconsumed_leftover_warns_not_raises(self, caplog):
        import logging as _logging
        class Impl:
            def __call__(self, x1):
                return (x1,)
        with caplog.at_level(_logging.WARNING, logger="root"):
            args, kwargs = bind_params(Impl.__call__, {"x1": 1, "ignored": 2})
        assert args == [1]
        assert kwargs == {}
        assert any("not consumed" in rec.message for rec in caplog.records)

    # --- 新契约: 有默认值的参数用默认值; warn_leftover 抑制 ---

    def test_defaulted_param_uses_default_when_not_supplied(self):
        class Impl:
            def __call__(self, x, axis=-1):
                return (x, axis)
        args, kwargs = bind_params(Impl.__call__, {"x": 1})  # axis 未传, 有默认
        assert args == [1]
        assert kwargs == {}

    def test_warn_leftover_false_suppresses(self, caplog):
        import logging as _logging
        class Impl:
            def __call__(self, x1):
                return (x1,)
        with caplog.at_level(_logging.WARNING, logger="root"):
            args, kwargs = bind_params(Impl.__call__, {"x1": 1, "ignored": 2},
                                       warn_leftover=False)
        assert args == [1]
        assert not any("not consumed" in rec.message for rec in caplog.records)

    # --- *args collects unconsumed entries ---

    def test_var_positional_collects_leftover(self):
        class Impl:
            def __call__(self, *args, **kwargs):
                pass
        args, kwargs = bind_params(Impl.__call__, {"x": 1, "y": 2, "alpha": 0.5})
        assert args == [1, 2, 0.5]
        assert kwargs == {"x": 1, "y": 2, "alpha": 0.5}

    def test_var_positional_with_named_params(self):
        class Impl:
            def __call__(self, x, *args, **kwargs):
                pass
        args, kwargs = bind_params(Impl.__call__, {"x": 1, "y": 2, "alpha": 0.5})
        assert args == [1, 2, 0.5]
        assert kwargs == {"y": 2, "alpha": 0.5}

    def test_var_positional_no_warn(self, caplog):
        import logging as _logging
        class Impl:
            def __call__(self, *args):
                pass
        with caplog.at_level(_logging.WARNING, logger="root"):
            args, kwargs = bind_params(Impl.__call__, {"x": 1, "y": 2})
        assert args == [1, 2]
        assert not any("not consumed" in rec.message for rec in caplog.records)

    # --- self pool entry handling ---

    def test_self_pool_entry_to_var_args(self):
        """ACLNN param named 'self' reachable via *args, not via **kwargs."""
        class Impl:
            def __call__(self, *args, **kwargs):
                pass
        args, kwargs = bind_params(Impl.__call__, {"self": 10, "other": 20})
        assert args == [10, 20]
        assert "self" not in kwargs

    def test_self_excluded_from_var_keyword(self):
        """Pool entry 'self' never injected into **kwargs."""
        class Impl:
            def __call__(self, x, **kwargs):
                pass
        args, kwargs = bind_params(Impl.__call__, {"self": 10, "x": 1, "alpha": 0.5})
        assert args == [1]
        assert "self" not in kwargs
        assert kwargs == {"alpha": 0.5}


class _FakeTensor:
    def __init__(self, v):
        self.v = v
        self.moved_to = None
    def to(self, device):
        self.moved_to = device
        return self


class TestToDevice:
    """to_device: framework-side H2D for Mode B (spec §7.5)."""

    def test_none_and_cpu_passthrough(self):
        assert to_device(None, "cuda:0", "torch") is None
        assert to_device(_FakeTensor(1), "cpu", "torch").v == 1  # cpu -> no move needed path

    def test_tensor_moved(self, monkeypatch):
        import sys, types
        fake = types.ModuleType("torch")
        fake.Tensor = _FakeTensor
        monkeypatch.setitem(sys.modules, "torch", fake)
        t = _FakeTensor(5)
        out = to_device(t, "cuda:0", "torch")
        assert out is t
        assert t.moved_to == "cuda:0"

    def test_list_recursed(self, monkeypatch):
        import sys, types
        fake = types.ModuleType("torch")
        fake.Tensor = _FakeTensor
        monkeypatch.setitem(sys.modules, "torch", fake)
        items = [_FakeTensor(1), _FakeTensor(2)]
        out = to_device(items, "cuda:1", "torch")
        assert [o.moved_to for o in out] == ["cuda:1", "cuda:1"]

    def test_non_tensor_passthrough(self, monkeypatch):
        import sys, types
        fake = types.ModuleType("torch")
        fake.Tensor = _FakeTensor
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert to_device(42, "cuda:0", "torch") == 42
        assert to_device("s", "cuda:0", "torch") == "s"

    def test_no_torch_is_noop(self, monkeypatch):
        import sys
        # 模拟 torch 不可导入
        monkeypatch.setitem(sys.modules, "torch", None)
        val = object()
        assert to_device(val, "cuda:0", "torch") is val


class TestResolveCallable:
    """resolve_callable: dotted api string -> callable; reject classes (spec §7.6)."""

    def test_resolves_function(self):
        import numpy
        fn = resolve_callable("numpy.abs")
        assert fn is numpy.abs

    def test_resolves_nested_attr(self):
        fn = resolve_callable("numpy.linalg.norm")
        assert callable(fn)

    def test_rejects_class(self):
        with pytest.raises(ValueError):
            resolve_callable("numpy.ndarray")  # class, not a function

    def test_rejects_too_short(self):
        with pytest.raises(ValueError):
            resolve_callable("noparts")
