# TestSpec — 算子测试规范

> **不依赖任何测试框架或工具，Ascend 生态统一的算子测试规范。**

## 快速上手

```python
import numpy

class AbsTestSpec:
    """Abs 算子测试规范"""
    def golden(x, **kwargs):
        return [numpy.abs(x)]

    third_party = {"torch": "torch.abs", "tf": "tf.raw_ops.Abs"}
    tolerance = {"float32": {"standard": "BinaryCompareStandard"}}
```

## 约定速查

| 属性 | 用途 | 合法类型 |
|------|------|---------|
| `golden` | CPU 真值 | `str` / 函数 / 类 |
| `third_party` | 三方标杆 | `str` / `dict` / 类 |
| `compare` | 自定义比对 | 函数 |
| `pre_compare` | 比对前处理 | 函数 |
| `customize_inputs` | 自定义输入 | 函数 |
| `tolerance` | 精度标准 | `dict(dtype→{standard, ...})` |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类（仅 torch.API） |

所有属性可选。`golden` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `golden = "numpy.abs"` | numpy 有直接对应 API |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度，`*` 前=输入，`*` 后=属性 |
| 类 | `class Golden: def __call__(self, x):` | 需要状态管理，`__init__` + `__call__` + `__del__` |

`third_party` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `third_party = "torch.nn.functional.softmax"` | 单 API 对标 |
| dict | `third_party = {"torch": "...", "tf": "..."}` | 多 vendor / 多框架 |
| 类 | `third_party = MyImplClass` | 小算子拼接，需要 `__call__` |

## 类命名

`op_name` → 类名 = `PascalCase + TestSpec`：

| op_name | 类名 |
|---------|------|
| `abs` | `AbsTestSpec` |
| `softmax_v2` | `SoftmaxV2TestSpec` |
| `layer_norm` | `LayerNormTestSpec` |

也可用 `__spec__` dict 显式注册：

```python
__spec__ = {"abs": AbsTestSpec}
```

## 消费方式

```python
from ttk.test_spec import TestSpecManager

mgr = TestSpecManager(search_paths=("path/to/specs",))
cls = mgr.load("softmax_v2")

if cls and mgr.has(cls, "golden"):
    golden_fn = mgr.get(cls, "golden")

if cls and mgr.has(cls, "third_party"):
    tp = mgr.get(cls, "third_party")

mgr.validate(cls)  # 类型检查，不合规抛 InvalidSpecError（fail-fast）
```

## 更多示例

见 `examples/` 目录，从 `01_minimal.py` 开始渐进阅读。

## 命名与发现规则

- **文件命名**: spec 文件不能以 `_` 开头（loader 跳过 `_` 前缀文件）
- **`_snake_to_pascal` 限制**: 仅处理纯 snake_case（如 `softmax_v2` → `SoftmaxV2`）。对于已是 PascalCase 或含数字拼接的算子名（如 `BatchMatMul`、`matmul_3d`），请使用 `__spec__` dict 显式注册。
