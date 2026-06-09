---
name: ttk-how-write-plugin
description: 编写自定义 Golden 和 Input 插件。只要用户提到自定义 golden 函数、__golden__、__input__、插件开发、算子没有内置 golden、怎么写 golden，或问"用 numpy 还是 torch"，就必须使用此 skill。涵盖 Kernel（numpy）和 ACLNN（torch）两个级别。
---

# 编写自定义插件

插件是算子资产的一部分，归属于算子仓库（如 ops-math、ops-nn 的 `tests/assets/` 目录），与测试框架完全解耦。只要遵循 `__golden__` / `__input__` 接口定义规则，任何工具都可以调用，不依赖 TTK 框架本身。

> **E2E 模式不支持 Input 插件**，也不需要自定义 Golden（框架直接在 CPU 上跑同 API 作为 Golden）。如需控制数据范围，使用 CSV 的 `input_data_ranges` 字段。

## 何时需要插件

| 场景 | 类型 | 说明 |
|------|------|------|
| 算子无内置 Golden | Golden 插件 | 自定义参考输出计算 |
| 需要特殊输入数据 | Input 插件 | 自定义输入生成逻辑 |

## 注册方式

插件通过 `__golden__` / `__input__` 字典注册，不使用装饰器。字典的 key 是测试级别（`"kernel"` / `"aclnn"`），value 是 `{算子名: 函数名}` 映射。一个文件可以同时注册 kernel 和 aclnn 级别。

```python
__golden__ = {
    "kernel": {"my_op": "my_op_golden"},
    "aclnn": {"aclnnMyOp": "aclnn_my_op_golden"}
}

__input__ = {
    "kernel": {"my_op": "my_op_input"}
}
```

## 级别与数据类型

Kernel 模式在 CANN 底层运行，Golden/Input 用 numpy；ACLNN 模式在 torch API 层运行，用 torch。两者不能混用。

| 级别 | 数据类型 | 参数来源 |
|------|---------|---------|
| kernel | numpy.ndarray | 算子 def.cpp 中的输入参数 |
| aclnn | torch.Tensor | aclnn*.h 头文件中的参数 |

## 函数签名约定

Golden/Input 函数参数分为输入张量、算子属性、框架元信息三部分。当**输入有可选（带默认值）且属性有必选（无默认值）**时，用 `*` 分隔：

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

## 多输出算子

返回列表，每个元素对应一个输出：

```python
def sort_golden(x, descending=False, **kwargs):
    values = np.sort(x, axis=-1)
    indices = np.argsort(x, axis=-1)
    if descending:
        values = np.flip(values, axis=-1)
        indices = np.flip(indices, axis=-1)
    return [values, indices]
```

## 使用插件

通过 `--plugin` 指定插件文件或目录，建议使用绝对路径。指向目录时，框架会逐层扫描查找 `__golden__` / `__input__` 注册。算子仓通常放在 `tests/assets/` 目录下。

```shell
# 指向具体文件（建议绝对路径）
python3 -m ttk kernel -i cases.csv --plugin /path/to/ops-math/math/abs/tests/assets/golden.py

# 指向目录（框架自动扫描）
python3 -m ttk kernel -i cases.csv --plugin /path/to/ops-math/math/abs/tests/assets/

# 多个路径（逗号分隔）
python3 -m ttk kernel -i cases.csv --plugin /path/to/golden.py,/path/to/input.py

# ACLNN 模式
python3 -m ttk aclnn -i cases.csv --plugin /path/to/tests/assets/
```

## 优先级

**Golden**: Custom plugin > Builtin golden > 无 Golden（标记 UNSUPPORTED）

**Input**: CSV 手动二进制文件（`manual_input_binaries` / `manual_tensor_binaries`）> Custom plugin > input_data_ranges + --input-dist > 默认 uniform(-1,1)

## 规则

1. **注册名必须匹配**：`__golden__` 中的 key 必须与 CSV 的 `op_name` 或 `api_name` 完全一致
2. **参数顺序**：函数参数名和顺序与算子定义文件（def.cpp / aclnn*.h）中的输入参数一致
3. **kwargs 始终接收**：通过 `**kwargs` 接收元信息（dtypes, shapes, formats, soc_version 等）
4. **返回类型**：kernel 级别返回 numpy，aclnn 级别返回 torch
5. **优先级**：Custom > Builtin。同名注册会覆盖并打印警告

## 调试

```shell
# 检查插件语法
python3 -c "exec(open('golden.py').read()); print(__golden__)"

# 单用例测试
python3 -m ttk kernel -i cases.csv -t case_name --plugin golden.py --single-log
```

常见问题：
- Golden 不生效：检查 `--plugin` 路径是否正确、`__golden__` 中的 key 是否与 CSV 匹配
- 参数数量不匹配：核对函数签名与算子 def.cpp 中的输入参数是否一致

## 模式专属示例

| 级别 | 参考 | 说明 |
|------|------|------|
| Kernel | `references/kernel-plugin.md` | Golden + Input 示例，numpy，kwargs 字段 |
| ACLNN | `references/aclnn-plugin.md` | Golden + Input 示例，torch，kwargs 字段 |

实际项目示例（路径相对于对应算子仓库根目录）：
- `math/abs/tests/assets/golden.py` — 简单 Kernel Golden
- `math/sort_with_index/tests/assets/golden.py` — 多输出 + torch 辅助
- `math/reduce_max/tests/assets/golden.py` — 同时注册 kernel 和 aclnn
- `math/reduce_max/tests/assets/input.py` — Input 插件
- `norm/layer_norm/tests/assets/golden.py` — 复杂 Kernel Golden（含格式处理）

## 相关 Skill

- 插件写好了怎么跑？→ `ttk-how-run-test`
- 插件跑不过怎么排查？→ `ttk-how-diagnose`
- 用例 CSV 怎么写？→ `ttk-how-write-case`
