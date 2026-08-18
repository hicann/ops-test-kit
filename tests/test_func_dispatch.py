import inspect
import numpy
import pytest
from ttk.utilities.func_dispatch import framework_of, bind_by_name, resolve_callable_str, UnknownParamError


def test_framework_of():
    import torch
    assert framework_of(numpy.add) == "numpy"          # ufunc
    assert framework_of(numpy.isposinf) == "numpy"      # 非 ufunc
    assert framework_of(torch.add) == "torch"           # 顶层 builtin
    assert framework_of(torch.ops.aten.amax) == "torch"  # OpOverloadPacket
    assert framework_of(lambda x: x) is None            # 自定义
    def g(x, **kw): return x
    assert framework_of(g) is None
    class C:
        def __call__(self, x): return x
    assert framework_of(C) is None                      # type(class)
    assert framework_of(C()) is None                    # instance


def test_bind_by_name_partial_split():
    class G:
        def __init__(self, x, *, axis):
            self.x, self.axis = x, axis
        def __call__(self, y):
            return [self.x + y + self.axis]
    pool = {"x": 1, "y": 2, "axis": 9}
    ia, ik = bind_by_name(G.__init__, pool)
    inst = G(*ia, **ik)
    ca, ck = bind_by_name(inst.__call__, pool)
    assert inst(*ca, **ck) == [12]
    assert inst.x == 1 and inst.axis == 9


def test_bind_by_name_kwargs_absorb():
    class G:
        def __call__(self, x, **kw):
            return [x, kw.get("axis")]
    pool = {"x": 1, "axis": 9}
    ca, ck = bind_by_name(G().__call__, pool)
    assert G()(*ca, **ck) == [1, 9]


def test_bind_by_name_object_init():
    # object.__init__(self, /, *args, **kwargs) —— bind_by_name 不应崩
    # *args 收集未消费 pool 条目, **kwargs 也吸收 (不含 self)
    args, kwargs = bind_by_name(object.__init__, {"x": 1, "axis": 9})
    assert args == [1, 9] and kwargs == {"x": 1, "axis": 9}


def test_bind_by_name_var_args_collects_leftover():
    class G:
        def __call__(self, *args, **kwargs):
            return [list(args), kwargs]
    pool = {"x": 1, "y": 2, "alpha": 0.5}
    ca, ck = bind_by_name(G().__call__, pool)
    assert ca == [1, 2, 0.5]
    assert ck == {"x": 1, "y": 2, "alpha": 0.5}


def test_bind_by_name_var_args_with_named_params():
    class G:
        def __call__(self, x, *args, **kwargs):
            return [x, list(args), kwargs]
    pool = {"x": 1, "y": 2, "alpha": 0.5}
    ca, ck = bind_by_name(G().__call__, pool)
    assert ca == [1, 2, 0.5]
    assert ck == {"y": 2, "alpha": 0.5}


def test_bind_by_name_self_pool_entry_to_var_args():
    class G:
        def __call__(self, *args, **kwargs):
            return [list(args), kwargs]
    pool = {"self": 10, "other": 20}
    ca, ck = bind_by_name(G().__call__, pool)
    assert ca == [10, 20]
    assert "self" not in ck
    assert ck == {"other": 20}


def test_bind_by_name_self_excluded_from_kwargs():
    class G:
        def __call__(self, x, **kwargs):
            return [x, kwargs]
    pool = {"self": 10, "x": 1, "alpha": 0.5}
    ca, ck = bind_by_name(G().__call__, pool)
    assert ca == [1]
    assert "self" not in ck
    assert ck == {"alpha": 0.5}


def test_bind_by_name_unknown_raises():
    def f(x, missing): return x
    with pytest.raises(UnknownParamError):
        bind_by_name(f, {"x": 1})


def test_bind_by_name_required_missing_raises_with_leftover():
    """必填参数名拼错时不得贪婪绑定未消费的 pool 项(否则静默算错)。"""
    def f(x, scale):
        return (x, scale)
    # 'scale' 既非 input 也非 attr; 'alpha' 是未消费项, 不应被贪婪绑定给 'scale'。
    with pytest.raises(UnknownParamError):
        bind_by_name(f, {"x": 1, "alpha": 0.5})


def test_resolve_callable_str():
    import numpy
    assert resolve_callable_str("numpy.add") is numpy.add
    assert resolve_callable_str("np.negative") is numpy.negative
    import torch
    assert resolve_callable_str("torch.mm") is torch.mm
    assert callable(resolve_callable_str("torch.ops.aten.amax"))  # the exact form that NameErrored before
