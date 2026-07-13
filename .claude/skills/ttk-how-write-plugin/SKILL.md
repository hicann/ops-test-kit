---
name: ttk-how-write-plugin
description: 编写/定义算子测试规范（TestSpec，尚未跑或要新增 golden/输入/精度标准时）。只要用户提到算子测试规范、golden、customize_inputs、自定义输入、third_party、精度标准、tolerance、__spec__、TestSpec、插件、算子没有 golden、怎么写 golden、kernel/aclnn/e2e 的 golden 用 numpy 还是 torch，就必须使用此 skill。即使用户只是说"这个算子没 golden 怎么办"、"golden 用 numpy 还是 torch"、"怎么写 tolerance"，也应触发。已跑完结果不对要排障用 ttk-how-diagnose。
---

# 编写算子测试规范（TestSpec）

算子测试规范（Golden / 自定义输入 / 三方标杆 / 精度标准 / 自定义比对）通过 **TestSpec**（`__spec__` 标记）定义，归属于算子仓库（如 ops-math、ops-nn 的 `tests/assets/`），与测试框架完全解耦。任何工具都可消费，不依赖 TTK。

## 何时需要

| 场景 | 资产 | 说明 |
|------|------|------|
| 算子无内置 Golden | `golden` | 自定义 CPU 参考输出 |
| 需要特殊输入数据 | `customize_inputs` | 自定义输入生成逻辑 |
| 需要三方标杆对照 | `third_party` | torch/tf 等参考实现 |
| 默认比对不适用 | `compare` / `pre_compare` | 自定义比对逻辑 |

## 快速上手

```python
import numpy

__spec__ = {"add": "AddTestSpec"}

class AddTestSpec:
    """Add 算子测试规范（kernel 流程，numpy）"""
    def golden(x, y, **kwargs):
        return [numpy.add(x, y)]

    third_party = {"torch": "torch.add"}
    tolerance = {"float32": {"standard": "stat_rel_err"}}
```

## 数据类型

- **`golden` / `customize_inputs`**：按消费流程区分（golden 在 CPU 计算真值，不下设备、不 H2D）

  | 流程 | 数据类型 | 参数来源 |
  |------|---------|---------|
  | kernel | numpy.ndarray | 算子 def.cpp 输入参数 |
  | aclnn | torch.Tensor | aclnn*.h 头文件参数 |
  | e2e | torch.Tensor | torch API 参数 |

- **`compare` / `pre_compare`**：入参**始终为 numpy ndarray**（任何场景；框架把输出转 numpy 再传入）

> E2E 默认由框架在 CPU 跑同 API 作 Golden；如需控制输入范围用 CSV 的 `input_data_ranges` 字段。

## 资产属性

所有属性可选，按需声明：

| 属性 | 用途 | 类型 |
|------|------|------|
| `golden` | CPU 真值 | `str` / 函数 / 类 |
| `customize_inputs` | 自定义输入生成 | 函数 |
| `third_party` | 三方标杆 | `str` / `dict` / 类 |
| `tolerance` | 精度标准 | `dict(dtype→{standard,...})` |
| `compare` | 自定义比对 | 函数 |
| `pre_compare` | 比对前处理 | 函数 |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类（仅 torch API） |

## golden 三形式

| 形式 | 示例 | 适用 |
|------|------|------|
| 字符串 | `golden = "numpy.abs"` | 有直接对应 API |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度 |
| 类 | `golden = GoldenCls` | 需要状态管理（`__init__` + `__call__`） |

函数形式两种写法 —— 类内直接定义方法（方法名必须是 `golden`），或独立函数 + 类属性赋值（函数名任意）：

```python
class AbsTestSpec:
    def golden(x, **kwargs):          # 写法1：类内方法
        return [numpy.abs(x)]

# 写法2：独立函数 + 类属性赋值
def abs_golden(x, **kwargs):
    return [numpy.abs(x)]

class AbsTestSpec:
    golden = abs_golden
```

> `customize_inputs` / `compare` / `pre_compare` 同理：独立函数需在类中赋值同名属性。

## 函数签名约定

Golden/Input 函数参数分为**输入张量、算子属性、框架元信息**三部分。框架按位置传输入张量、按关键字传属性；当**输入有可选（带默认值）且属性有必选（无默认值）**时，为避免位置参数与关键字属性歧义，用 `*` 分隔（`*` 前是输入，`*` 后是属性）：

```python
# 输入有可选 + 属性有必选 → 用 * 分隔
def op_golden(x, y, bias=None, *, axis, keepdims=False, **kwargs):
    ...
```

其他场景不加 `*`：

```python
# 输入全部必选，属性全部可选 → 不加 *
def op_golden(x, y, alpha=1.0, **kwargs):
    ...

# 无属性 → 不加 *
def op_golden(x, y, **kwargs):
    ...
```

类形式时参数名必须是算子输入/属性名（`__init__`/`__call__` 参数并集 ⊆ `inputs ∪ attrs`）；有默认值的参数可省略。

## customize_inputs（自定义输入）

参数名匹配算子定义，返回修改后的输入（结构与原输入一致）：

```python
# kernel 流程（numpy）
def customize_inputs(x, min_val, max_val, **kwargs):
    if min_val[0] == max_val[0]:
        min_val[0] = numpy.min(x)
        max_val[0] = numpy.max(x)
    return (x, min_val, max_val)
```

## 多输出算子

返回列表，每个元素对应一个输出：

```python
def sort_golden(x, descending=False, **kwargs):
    values = numpy.sort(x, axis=-1)
    indices = numpy.argsort(x, axis=-1)
    return [values, indices]
```

## tolerance（精度标准）

按 dtype 声明 **2.1 官方标准**：`stat_rel_err` / `binary_equal` / `cross_check`（`quant` 待支持）。

```python
tolerance = {"float32": {"standard": "stat_rel_err"}}
```

> CLI 框架别名（`close`/`cosine`/`binary`/`requant`）走 `--compare`，不进 `Spec.tolerance`。

## 注册与发现

- **类命名**：`op_name` → `PascalCase + TestSpec`（如 `abs` → `AbsTestSpec`、`softmax_v2` → `SoftmaxV2TestSpec`）
- **`__spec__` dict**：算子名非纯 snake_case 时（如 `BatchMatMul`）用 `__spec__ = {"op": "ClassName"}` 显式注册（值为类名字符串；loader AST 扫描不 exec 文件）
- **文件命名**：spec 文件不能以 `_` 开头（loader 跳过）
- **惰性加载**：loader 首次按 `__spec__` AST 建索引，仅命中时 exec 该文件

## 使用

`--plugin` 指定 spec 文件或目录，建议使用绝对路径。指向目录时框架逐层扫描：

```shell
# 指向目录（算子仓通常放 tests/assets/）
python3 -m ttk kernel -i cases.csv --plugin /path/to/ops-math/math/abs/tests/assets/

# 多个路径（逗号分隔）
python3 -m ttk aclnn -i cases.csv --plugin /path/to/assets_a/,/path/to/assets_b/
```

## 规则

1. **注册名必须匹配**：类名（`PascalCase+TestSpec`）或 `__spec__` 的 key 必须与 CSV 的 `op_name` / `api_name` 完全一致
2. **参数顺序**：函数参数名和顺序与算子定义文件（def.cpp / aclnn*.h）中的输入参数一致
3. **kwargs 始终接收**：通过 `**kwargs` 接收元信息（dtypes、shapes、formats、soc_version 等，完整字段见 `references/kernel-plugin.md` / `aclnn-plugin.md`）
4. **返回类型**：`golden` 返回列表（每个输出一个元素）；`customize_inputs` 返回与输入结构一致的 tuple

## 调试

```shell
# 检查 spec 文件语法
python3 -c "import ast; ast.parse(open('spec.py').read()); print('ok')"

# 单用例测试
python3 -m ttk kernel -i cases.csv -t case_name --plugin /path/to/assets/ --single-log
```

常见问题：
- 资产不生效：检查 `--plugin` 路径、类名（`PascalCase+TestSpec`）或 `__spec__` 注册名是否与 CSV `op_name`/`api_name` 一致
- 参数数量不匹配：核对函数签名与算子定义文件的输入参数是否一致

## CSV 数据范围替代

如果只需要调整随机数据范围，可以不写 `customize_inputs`，直接在 CSV 中设置：

```csv
input_data_ranges,"((-1, 1), (0.1, 10))"
```

或使用 `--input-dist normal` 切换为正态分布。

## 详细规范

完整规范见 `ttk/test_spec/README.md`，渐进示例见 `ttk/test_spec/examples/`（从 `01_minimal.py` 开始）。

按流程的示例与 kwargs 字段：
- kernel 流程（numpy）→ `references/kernel-plugin.md`
- aclnn 流程（torch）→ `references/aclnn-plugin.md`

## 相关 Skill

- 资产写好了怎么跑？→ `ttk-how-run-test`
- 跑不过怎么排查？→ `ttk-how-diagnose`
- 用例 CSV 怎么写？→ `ttk-how-write-case`
