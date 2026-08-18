# E2E 用例编写

适用于 `python3 -m ttk e2e`，使用 `TestcaseE2e`，共 23 个字段（含[公共字段](../Test_Case_Generation.md#公共字段所有模式通用)）。

**参数来源**：E2E用例的参数（张量数量/顺序/dtype、关键字参数等）来自框架API的函数签名。查看PyTorch官方文档或torch_npu扩展文档获取参数定义。

**API类型**：

| api_name 示例 | 类型 | 说明 |
|---------------|------|------|
| `torch.add` | 模块函数 | 直接调用 |
| `torch.nn.functional.relu` | 子模块函数 | 调用子模块中的函数 |
| `torch.Tensor.relu_` | Tensor方法 | 通过Tensor实例调用（原地操作） |

## 身份标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `api_name` | STRING | **是** | *(无)* | 框架API路径。如 `torch.add`、`torch.nn.functional.relu`、`torch.Tensor.relu_` |

## 张量属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `tensor_view_shapes` | SHAPE_NESTED | **是** | *(无)* | 输入张量shape，支持TensorList嵌套。如 `"((2,3,4),(2,3,4))"` |
| `tensor_dtypes` | DTYPE_NESTED | **是** | *(无)* | 输入张量数据类型。如 `"('float32','float32')"`。支持广播：`"('float32',)"` 填充所有位置 |
| `tensor_formats` | DTYPE_NESTED | 否 | `()` | 张量格式。支持广播 |
| `tensor_storage_shapes` | SHAPE_NESTED | 否 | `()` | 非连续张量的存储shape。回退到视图shape |
| `tensor_view_offsets` | INT_NESTED | 否 | `()` | 张量视图在存储中的偏移量。 |
| `tensor_view_strides` | STRIDE | 否 | `()` | 张量视图步长。未指定时自动计算 |

## 输出与属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_tensor_indexes` | INT_TUPLE | 否 | *(自动)* | 输出张量索引。Inplace Tensor方法（`torch.Tensor.xxx_`）自动填充 `(0,)`；带 `out` 参数的API需手动指定输出位置索引 |
| `attributes` | DICT | 否 | `{}` | 框架API关键字参数。如 `{'alpha': 1.0}`。API签名中的必选非张量参数**必须**提供 |
| `golden_api` | STRING | 否 | `""` | 替代Golden计算的API。如 `torch.nn.functional.conv2d`。设为 `"disable"` 可禁用Golden生成 |

## 参考用例

`examples/case_store/e2e/` 目录下提供了各种场景的示例：

| 文件 | 涵盖场景 |
|------|----------|
| `torch_add.csv` | 基本用例、`torch.add` |
| `torch_npu_conv2d.csv` | `torch_npu.npu_conv2d`、`golden_api` |
| `tf_ops.csv` | TensorFlow API |

## 批一致性字段

配合 `--deterministic-level 3` 使用，用于跨用例输出切片比对。详见 [确定性计算与批一致性](../Deterministic_Compute.md)。

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `batch_seed` | FREE_EVAL | 否 | `None` | 分组标识，相同 seed 的用例归为一组。每个输出一个 seed，嵌套列表对应多输出。如 `(100,)` |
| `batch_axis` | FREE_EVAL | 否 | `None` | 切片所在轴。嵌套结构为 `输出 → 张量组 → 轴`。如 `(([0],),)` 表示第 0 个输出的第 0 个张量组在轴 0 上切片 |
| `batch_slice_info` | FREE_EVAL | 否 | `None` | 切片范围 `(start, stop, step)`，嵌套结构与 `batch_axis` 对齐。如 `(([[0,5,1]],),)` 表示切片 `[0:5:1]` |

> `batch_consistency_id` 由 `batch_seed` + `batch_axis` + `batch_slice_info` 自动生成，无需填写。相同 seed 且切片结构相同的用例归入同一比对组。

### 字段嵌套结构

三个字段按 `输出 → 张量组 → 切片` 三层嵌套，逐层 zip 对齐：

```
batch_seed        = ( (100,), )        # 第0个输出的第0个张量组的seed=100
batch_axis        = ( ([0],) )         # 第0个输出的第0个张量组，轴=0
batch_slice_info  = ( ([[0,5,1]],) )   # 第0个输出的第0个张量组，切片[0:5:1]
```

### CSV 示例

```csv
testcase_name,api_name,tensor_view_shapes,batch_seed,batch_axis,batch_slice_info,...
slice_0,torch.add,"((5,8),)",(100,),(([0],),),(([[0,5,1]],),),...
slice_1,torch.add,"((5,8),)",(100,),(([0],),),(([[5,10,1]],),),...
full,torch.add,"((10,8),)",(100,),,,...
```

上例中 `slice_0` 取输出轴 0 的 `[0:5]`，`slice_1` 取 `[5:10]`，`full` 不切片取全部。三者 seed 相同，`slice_0` + `slice_1` 的切片长度之和等于 `full`，归入同一比对组。
