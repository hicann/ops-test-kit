"""_invoke class 分支契约: __init__/__call__ 参数(除kwargs)并集 ⊆ inputs∪attrs;
input/attr 喂给声明它的方法(都声明则都喂)。device 保留注入, 有默认值用默认。

Mode A/B 二分已删, 以下场景都应通:
  - 无自定义 __init__ + 非空 attrs (object.__init__ 拒绝任意 kwarg → cls() 实例化)
  - __init__/__call__ 同名 input (两边都喂)
  - 无输入 tensor 的算子 (range/eye 风格, device 框架注入)
  - 有默认值的参数未传 → 用默认
  - 非法参数名 (无默认值且不在 inputs∪attrs) → UnknownParamError
"""
import pytest

# torch is needed by the provider="torch" path's _to_vendor_tensor
# (no-op on cpu but the import is referenced).
torch = pytest.importorskip("torch")

from ttk.remote.server import executor
from ttk.remote.server.execution_container import UnknownParamError


def test_invoke_no_init_class_with_attrs():
    """无自定义 __init__ 的类 + 非空 attrs 不应 TypeError(object.__init__ 不收任意 kwarg)。"""

    class NoInit:
        def __call__(self, x):
            return [x]

    out = executor._invoke(
        NoInit,
        named={"x": 1},
        attrs={"axis": 9, "eps": 1e-5},
        provider="torch",
        device_id="cpu",
        use_device=False,
    )
    assert out == [1]


def test_invoke_init_and_call_same_input():
    """__init__(x1) + __call__(x1) 同名 input: 两边都喂(原 bug 复现)。"""
    class Foo:
        def __init__(self, x1):
            self.x1 = x1
        def __call__(self, x1):
            return [self.x1 + x1]

    x1 = torch.tensor([1.0, 2.0])
    out = executor._invoke(
        Foo, named={"x1": x1}, attrs={},
        provider="torch", device_id="cpu", use_device=False,
    )
    assert torch.equal(out[0], x1 + x1)


def test_invoke_no_input_tensor_device_injected():
    """无输入 tensor 算子(range/eye 风格): device 由框架注入到声明它的方法。"""
    seen = {}

    class EyeOp:
        def __init__(self, n, device):
            seen["device"] = device
            self.n = n
        def __call__(self):
            return [self.n]

    out = executor._invoke(
        EyeOp, named={}, attrs={"n": 3},
        provider="torch", device_id="cpu", use_device=False,
    )
    assert out == [3]
    assert seen["device"] == "cpu"


def test_invoke_defaulted_param_uses_default():
    """有默认值的参数未传时用默认值。"""
    class WithAxis:
        def __call__(self, x, axis=-1):
            return [(x, axis)]

    out = executor._invoke(
        WithAxis, named={"x": 1}, attrs={},
        provider="torch", device_id="cpu", use_device=False,
    )
    assert out[0] == (1, -1)


def test_invoke_unknown_param_raises():
    """参数名既非 input/attr、又无默认值 → UnknownParamError(契约校验)。"""
    class Bad:
        def __call__(self, x, bogus):  # bogus 无默认值, 不在 inputs∪attrs
            return [x]

    with pytest.raises(UnknownParamError):
        executor._invoke(
            Bad, named={"x": 1}, attrs={},
            provider="torch", device_id="cpu", use_device=False,
        )


def test_invoke_attrs_not_duplicated_positionally():
    """attrs 不得既按位置又按关键字重复传入(issue #98 检视 #1)。"""
    def f(inp, normalized_shape, weight=None, eps=1e-5):
        return [inp, normalized_shape, weight, eps]

    out = getattr(executor, "_invoke")(
        f, named={"x": 5}, attrs={"normalized_shape": 64, "eps": 1e-3},
        provider="torch", device_id="cpu", use_device=False,
    )
    # inp=5(位置绑定 x), normalized_shape/eps 走关键字, weight 用默认值。
    assert out == [5, 64, None, 1e-3]
