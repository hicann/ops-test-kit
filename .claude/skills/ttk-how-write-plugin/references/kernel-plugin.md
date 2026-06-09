# Kernel 插件示例（numpy）

Kernel 模式在 CANN 底层运行，Golden 和 Input 函数均使用 numpy.ndarray。

函数参数名和顺序与算子 def.cpp 中的输入参数一致（不含输出）。

## Golden 示例

### 简单逐元素算子

```python
import numpy as np

__golden__ = {"kernel": {"my_relu": "my_relu_golden"}}

def my_relu_golden(x, **kwargs):
    '''Golden for my_relu. Parameters follow @my_relu_def.cpp without outputs.'''
    return np.maximum(x, 0)
```

### 带属性的算子

```python
import numpy as np

__golden__ = {"kernel": {"my_reduce": "my_reduce_golden"}}

def my_reduce_golden(x, axis=0, keepdims=False, **kwargs):
    '''
    Golden for my_reduce. Parameters follow @my_reduce_def.cpp without outputs.

    **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
              full_soc_version, short_soc_version, testcase_name
    '''
    ori_dtype = kwargs.get("input_dtypes", ["float32"])[0]
    if "bfloat16" in str(ori_dtype).lower():
        x = x.astype("float32")
        return np.sum(x, axis=axis, keepdims=keepdims).astype(np.dtype("bfloat16"), copy=False)
    return np.sum(x, axis=axis, keepdims=keepdims)
```

### 多输出算子

```python
import numpy as np

__golden__ = {"kernel": {"my_sort": "my_sort_golden"}}

def my_sort_golden(x, descending=False, **kwargs):
    '''Golden for my_sort. Returns [values, indices].'''
    values = np.sort(x, axis=-1)
    indices = np.argsort(x, axis=-1)
    if descending:
        values = np.flip(values, axis=-1)
        indices = np.flip(indices, axis=-1)
    return [values, indices]
```

## Input 示例

### 简单避免零值

```python
import numpy as np

__input__ = {"kernel": {"my_div_op": "my_div_op_input"}}

def my_div_op_input(x, y, **kwargs):
    '''除法算子输入，避免零值导致除零。返回 [x, y]。'''
    x_data = np.random.uniform(0.1, 10.0, size=x.shape).astype(x.dtype)
    y_data = np.random.uniform(0.1, 10.0, size=y.shape).astype(y.dtype)
    return [x_data, y_data]
```

### 归约算子生成 axes Tensor

```python
import numpy as np

__input__ = {"kernel": {"reduce_max": "reduce_max_input"}}

def reduce_max_input(x, axes, **kwargs):
    '''
    为 reduce_max 生成 axes 输入。
    axes 为 None 时（动态 shape），生成全轴归约的 axes Tensor。
    返回 [x, axes_arr]。
    '''
    input_dtypes = kwargs.get('input_dtypes')
    axes_dtype = input_dtypes[0] if input_dtypes else np.int64
    if axes is None or (hasattr(axes, '__len__') and len(axes) == 0):
        axes_arr = np.array(tuple(range(len(x.shape))), dtype=axes_dtype)
    else:
        axes_arr = np.array(tuple(set(int(a) for a in axes)), dtype=axes_dtype)
    return [x, axes_arr]
```

### 索引算子生成关联输入

```python
import numpy as np

__input__ = {"kernel": {"gather_nd": "gather_nd_input"}}

def gather_nd_input(x, indices, **kwargs):
    '''
    为 gather_nd 生成 x 和 indices。
    x 生成有序数据以便验证索引正确性。
    返回 [params, res_indices]。
    '''
    if str(x.dtype) != "bool":
        params = np.arange(0, x.size, 1, dtype=x.dtype).reshape(x.shape)
    else:
        params = np.random.choice(a=[False, True], size=x.shape, p=[0.5, 0.5])

    ranks = indices.shape[-1]
    res_indices = []
    for rank in range(ranks):
        idx = np.random.uniform(0, params.shape[rank], (1,)).astype(indices.dtype)
        res_indices.append(idx.item())
    for index in indices.shape[0:-1]:
        res_indices = res_indices * index
    res_indices = np.reshape(res_indices, indices.shape).astype(indices.dtype, copy=False)
    return [params, res_indices]
```

## kwargs 字段

| 字段 | 说明 |
|------|------|
| `input_dtypes` | 输入 dtype 列表，如 `['float32', 'float32']` |
| `output_dtypes` | 输出 dtype 列表 |
| `input_ori_shapes` | 原始输入 shape |
| `output_ori_shapes` | 原始输出 shape |
| `input_formats` / `output_formats` | 张量格式，如 `['ND', 'NCHW']` |
| `input_ori_formats` / `output_ori_formats` | 原始张量格式 |
| `input_ranges` | 输入数据范围（Input 函数专用） |
| `full_soc_version` | SoC 版本全称 |
| `short_soc_version` | SoC 版本简称，如 `Ascend910B3` |
| `testcase_name` | 当前用例名称 |

## CSV 数据范围替代

如果只需要调整随机数据范围，可以不用写 Input 插件，直接在 CSV 中设置：

```csv
input_data_ranges,"((-1, 1), (0.1, 10))"
```

或使用 `--input-dist normal` 切换为正态分布。

## 实际项目示例

以下文件位于对应算子仓库（ops-math / ops-nn）中：
- `math/abs/tests/assets/golden.py` — 简单逐元素，含 complex32/bfloat16 处理
- `math/sort_with_index/tests/assets/golden.py` — 多输出，numpy+torch 混用
- `math/reduce_max/tests/assets/golden.py` — 同时注册 kernel 和 aclnn 两个级别
- `math/reduce_max/tests/assets/input.py` — 归约算子 axes Tensor 生成
- `math/histogram_v2/tests/assets/input.py` — 直方图算子输入
- `index/gather_nd/tests/assets/input.py` — 索引算子关联输入
- `index/scatter_nd/tests/assets/input.py` — scatter 算子输入
- `norm/layer_norm/tests/assets/golden.py` — 复杂 Golden，含格式转换（FRACTAL_NZ）
