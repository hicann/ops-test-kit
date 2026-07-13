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
    tolerance = {"float32": {"standard": "binary_equal"}}
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

## 类形式参数绑定（Impl class）

`golden` / `third_party` 用**类**形式时（`__init__` + `__call__`），输入 tensor 在调用前已完成 H2D，出现在 `__init__` 还是 `__call__` 都行。绑定规则：

- **参数名必须是算子输入/属性名**：`__init__`/`__call__` 所有参数（除 `self`/`**kwargs`）的并集 ⊆ `inputs ∪ attrs`，否则 `UnknownParamError`。
- **喂给声明它的方法**：每个 input/attr 绑到声明它的方法；两边都声明同名参数则两边都喂。
- **有默认值的参数可省略**：未传时用默认值（`def __call__(self, x, axis=-1)` 不传 axis 即用 -1）。
- **`device` 是框架保留参数**：有输入 tensor 时算子内部用 `tensor.device`（不必声明）；无输入 tensor 的算子（`range`/`eye` 等）声明 `device` 参数，框架注入目标设备。

```python
class AddImpl:
    def __init__(self, x):       # x 可进 __init__
        self.x = x
    def __call__(self, x, y):    # 同名 x 两边都喂；y 只喂 __call__
        return [self.x + x + y]

class EyeImpl:
    def __init__(self, n, device):    # 无输入 tensor: device 框架注入
        self.n, self.device = n, device
    def __call__(self):
        import torch
        return [torch.eye(self.n, device=self.device)]
```

## 类命名

`op_name` → 类名 = `PascalCase + TestSpec`：

| op_name | 类名 |
|---------|------|
| `abs` | `AbsTestSpec` |
| `softmax_v2` | `SoftmaxV2TestSpec` |
| `layer_norm` | `LayerNormTestSpec` |

也可用 `__spec__` dict 显式注册：

```python
# 值为类名字符串（非类对象）—— loader AST 扫描不 exec、spec 隔离 + 惰性 load；可写文件顶部
__spec__ = {"abs": "AbsTestSpec"}
```

## 消费方式

```python
from ttk.test_spec import get_spec_attr, get_spec_class_meta

golden_fn = get_spec_attr("softmax_v2", "golden", ("path/to/specs",))
tp = get_spec_attr("softmax_v2", "third_party", ("path/to/specs",))
meta = get_spec_class_meta("softmax_v2", ("path/to/specs",))   # {spec_file, class_name}
```

## 更多示例

见 `examples/` 目录，从 `01_minimal.py` 开始渐进阅读。

## 命名与发现规则

- **文件命名**: spec 文件不能以 `_` 开头（loader 跳过 `_` 前缀文件）
- **`_snake_to_pascal` 限制**: 仅处理纯 snake_case（如 `softmax_v2` → `SoftmaxV2`）。对于已是 PascalCase 或含数字拼接的算子名（如 `BatchMatMul`、`matmul_3d`），请使用 `__spec__` dict 显式注册。
- **AST 静态建索引**: loader 首次 `load` 时遍历所有 `.py` 文件，用 `ast.parse` 扫描模块级 `__spec__` 赋值（**不 exec 文件**），构建 `op → (file, classname)` 索引。扫描零副作用，spec 文件不会被提前执行。
- **惰性 load**: 仅当某个 op 被 `load(op)` 命中、且索引解析出它的来源文件时，loader 才 `exec` 那一个文件（按文件缓存模块对象）。其余文件在本次会话里可能永不 exec。
- **`__spec__` 可写顶部**: 因为扫描只读 AST、不 exec，`__spec__` 写在文件顶部（class 定义之前）和写在底部等价 —— 但写顶部更清晰，让阅读者一眼看到 op→class 映射。`01_minimal.py` 即采用顶部写法示范。
