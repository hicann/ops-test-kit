# ttk/test_spec/__init__.py
"""
TestSpec — 统一的算子测试规范

不依赖任何测试框架或工具。每个 API/算子 一个 TestSpec 类，所有测试规范要素作为可选属性定义。

约定速查：
    class AbsTestSpec:
        '''Abs 算子测试规范'''

        # golden — CPU 真值（可选）：字符串 / 函数 / 类
        golden = "numpy.abs"

        # third_party — 三方标杆（可选）：字符串 / dict / 类
        third_party = {"torch": "torch.abs", "tf": "tf.raw_ops.Abs"}

        # tolerance — 精度标准（可选；abs 精确运算，二进制一致）
        tolerance = {
            "float32": {"standard": "binary_equal"},
        }

        # compare / pre_compare / customize_inputs 未定义 → 框架用默认

    # __spec__ dict 注册（可选，命名约定兜底）；值为类名字符串（非类对象）——
    # loader AST 扫描不 exec、spec 隔离 + 惰性 load；可写文件顶部
    __spec__ = {"abs": "AbsTestSpec"}

类名约定（__spec__ 不存在时回退）：
    op_name: softmax_v2  →  SoftmaxV2TestSpec
    op_name: abs         →  AbsTestSpec

发现机制：loader 首次 load 时 AST 静态建索引（扫描不 exec），惰性 load 仅 exec
命中的那一个文件；__spec__ 因 AST 扫描而可写文件顶部。完整约定见 README.md。
"""

__all__ = ["SpecNotFoundError", "InvalidSpecError", "validate", "get_spec_attr", "get_spec_class_meta"]


class SpecNotFoundError(Exception):
    """找不到规范类"""
    pass


class InvalidSpecError(Exception):
    """属性类型不符合约定，validate 失败时抛出（fail-fast）"""
    pass


# 延迟 import，避免循环依赖
def __getattr__(name):
    if name == "get_spec_attr":
        from .manager import get_spec_attr
        return get_spec_attr
    if name == "get_spec_class_meta":
        from .manager import get_spec_class_meta
        return get_spec_class_meta
    if name == "validate":
        from .validator import validate
        return validate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
