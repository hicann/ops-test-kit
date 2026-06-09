# ACLNN 插件示例（torch）

ACLNN 模式在 torch API 层运行，Golden 和 Input 函数均使用 torch.Tensor。

参数名和顺序与 `aclnnXxxGetWorkspaceSize` 函数签名一致（不含 workspaceSize 和 executor）。

## Golden 示例

### 简单算子

```python
import torch

__golden__ = {"aclnn": {"aclnnAdd": "aclnn_add_golden"}}

def aclnn_add_golden(selfT, other, alpha=1.0, **kwargs):
    '''
    Aclnn golden for aclnnAdd.
    Parameters follow aclnnAddGetWorkspaceSize without workspaceSize & executor.

    **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
             use_torch, short_soc_version, testcase_name
    '''
    return torch.add(selfT, other, alpha=alpha)
```

### 带 out 参数的算子

```python
import torch

__golden__ = {"aclnn": {"aclnnMyOp": "aclnn_my_op_golden"}}

def aclnn_my_op_golden(selfT, other, alpha=1.0, out=None, **kwargs):
    '''
    Aclnn golden with out parameter.

    **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
             use_torch, short_soc_version, testcase_name
    '''
    result = torch.add(selfT, other, alpha=alpha)
    if out is not None:
        out.copy_(result)
        return out
    return result
```

### 多输出算子

```python
import torch

__golden__ = {"aclnn": {"aclnnSort": "aclnn_sort_golden"}}

def aclnn_sort_golden(selfT, descending=False, out0=None, out1=None, **kwargs):
    '''Returns [values, indices].'''
    values, indices = torch.sort(selfT, descending=descending)
    return [values, indices]
```

### 同时注册 Kernel 和 ACLNN

```python
import numpy as np
import torch

__golden__ = {
    "kernel": {"reduce_max": "reduce_max_golden"},
    "aclnn": {"aclnnMaxV2": "aclnn_max_v2_golden"}
}

def reduce_max_golden(x, axes, **kwargs):
    '''Kernel golden: numpy ndarray in, numpy ndarray out.'''
    return np.max(x, axis=tuple(axes) if axes is not None else None, keepdims=False)

def aclnn_max_v2_golden(selfT, dims, keepDims, noopWithEmptyDims, out, **kwargs):
    '''
    Aclnn golden: parameters follow aclnnMaxV2GetWorkspaceSize without workspaceSize & executor.
    All Tensors are torch.Tensor.

    **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
             use_torch, short_soc_version, testcase_name
    '''
    if dims is None or (isinstance(dims, (tuple, list)) and len(dims) == 0):
        if noopWithEmptyDims:
            result = selfT
        else:
            result = selfT.flatten()
            if keepDims:
                result = result.reshape([1] * selfT.dim())
    else:
        result = torch.amax(selfT, dim=dims, keepdim=keepDims)
    return result
```

## Input 示例

```python
import torch

__input__ = {"aclnn": {"aclnnMyDivOp": "aclnn_my_div_op_input"}}

def aclnn_my_div_op_input(selfT, other, **kwargs):
    '''
    除法算子输入，避免零值。返回 [selfT, other]。

    **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes, input_ranges,
             use_torch, short_soc_version, testcase_name
    '''
    selfT_data = torch.rand(selfT.shape, dtype=selfT.dtype).uniform_(0.1, 10.0)
    other_data = torch.rand(other.shape, dtype=other.dtype).uniform_(0.1, 10.0)
    return [selfT_data, other_data]
```

## kwargs 字段

| 字段 | 说明 |
|------|------|
| `tensor_dtypes` | 张量 dtype 嵌套结构 |
| `tensor_formats` | 张量格式嵌套结构 |
| `scalar_dtypes` | 标量 dtype 嵌套结构 |
| `input_ranges` | 输入数据范围（Input 函数专用） |
| `use_torch` | 是否支持 torch dtype |
| `short_soc_version` | SoC 版本简称 |
| `testcase_name` | 当前用例名称 |

## 实际项目示例

以下文件位于对应算子仓库（ops-math / ops-nn）中：
- `math/reduce_max/tests/assets/golden.py` — 同时注册 kernel 和 aclnn 两个级别
- `math/sort_with_index/tests/assets/golden.py` — 多输出
