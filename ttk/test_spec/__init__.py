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
            "float32": {"standard": "BinaryCompareStandard"},
        }

        # compare / pre_compare / customize_inputs 未定义 → 框架用默认

    __spec__ = {"abs": AbsTestSpec}  # 优先注册；可选，命名约定兜底

类名约定（__spec__ 不存在时回退）：
    op_name: softmax_v2  →  SoftmaxV2TestSpec
    op_name: abs         →  AbsTestSpec

完整约定见 README.md 和 op_assets_desc.md。
"""

__all__ = ["TestSpecManager", "SpecNotFoundError", "InvalidSpecError", "get_spec_attr"]


class SpecNotFoundError(Exception):
    """找不到规范类"""
    pass


class InvalidSpecError(Exception):
    """属性类型不符合约定，validate 失败时抛出（fail-fast）"""
    pass


# 延迟 import，避免循环依赖
def __getattr__(name):
    if name == "TestSpecManager":
        from .manager import TestSpecManager
        return TestSpecManager
    if name == "get_spec_attr":
        from .manager import get_spec_attr
        return get_spec_attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
