# ACLNN 流程 TestSpec 示例（torch）

ACLNN 流程在 torch API 层运行，`golden` / `customize_inputs` 用 **torch.Tensor**。参数名和顺序与 `aclnnXxxGetWorkspaceSize` 函数签名一致（不含 workspaceSize 和 executor）。

> 当 ACLNN 接口入参名与 Python 保留名冲突（如 `self`）时，类形式有两种写法：(1) 用 `*args` 按位置接收所有未具名消费的输入/属性，`args[0]` 对应 C header 第一个参数，以此类推；(2) 展开写但改名（如 `selfT`），改名参数按位置兜底——**约束：改名参数的出现顺序须与所消费的池条目顺序一致**（如 C header `self,other,alpha` 可用 `(selfT, other, alpha)` 捕获）。函数形式无此限制（参数名可任意，按位置绑定）。

## golden 示例

### 简单算子

```python
import torch

__spec__ = {"aclnnAdd": "AclnnAddTestSpec"}

class AclnnAddTestSpec:
    def golden(selfT, other, alpha=1.0, **kwargs):
        '''
        Parameters follow aclnnAddGetWorkspaceSize without workspaceSize & executor.

        **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes,
                 use_torch, short_soc_version, testcase_name
        '''
        return [torch.add(selfT, other, alpha=alpha)]
```

### 带 out 参数的算子

```python
import torch

class AclnnMyOpTestSpec:
    def golden(selfT, other, alpha=1.0, out=None, **kwargs):
        result = torch.add(selfT, other, alpha=alpha)
        if out is not None:
            out.copy_(result)
            return [out]
        return [result]
```

### 多输出算子

```python
import torch

class AclnnSortTestSpec:
    def golden(selfT, descending=False, out0=None, out1=None, **kwargs):
        '''Returns [values, indices].'''
        values, indices = torch.sort(selfT, descending=descending)
        return [values, indices]
```

## customize_inputs 示例

```python
import torch

class AclnnMyDivOpTestSpec:
    def golden(selfT, other, **kwargs):
        return [torch.div(selfT, other)]

    def customize_inputs(selfT, other, **kwargs):
        '''除法算子输入，避免零值。返回 (selfT, other)。

        **kwargs: tensor_dtypes, tensor_formats, scalar_dtypes, input_ranges,
                 use_torch, short_soc_version, testcase_name
        '''
        selfT_data = torch.rand(selfT.shape, dtype=selfT.dtype).uniform_(0.1, 10.0)
        other_data = torch.rand(other.shape, dtype=other.dtype).uniform_(0.1, 10.0)
        return (selfT_data, other_data)
```

## tolerance

```python
class AclnnMyOpTestSpec:
    def golden(selfT, **kwargs):
        return [torch.abs(selfT)]
    tolerance = {"float32": {"standard": "stat_rel_err"}}
```

## kwargs 字段

| 字段 | 说明 |
|------|------|
| `tensor_dtypes` | 张量 dtype 嵌套结构 |
| `tensor_formats` | 张量格式嵌套结构 |
| `scalar_dtypes` | 标量 dtype 嵌套结构 |
| `input_ranges` | 输入数据范围（customize_inputs 专用） |
| `use_torch` | 是否支持 torch dtype |
| `short_soc_version` | SoC 版本简称 |
| `testcase_name` | 当前用例名称 |
