# ACLNN 用例编写

适用于 `python3 -m ttk aclnn`，使用 `TestcaseAclnn`，共 27 个字段（含[公共字段](../Test_Case_Generation.md#公共字段所有模式通用)）。

## 用例标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `api_name` | STRING | **是** | *(无)* | ACLNN API名称。如 `aclnnAdd`、`aclnnCat`。 |

## 张量属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `tensor_view_shapes` | SHAPE_NESTED | **是** | `()` | 张量视图shape，支持TensorList嵌套。如 `"(((3,3),(3,2)),(3,5))"` 表示TensorList输入。 |
| `tensor_dtypes` | DTYPE_NESTED | **是** | `()` | 张量数据类型，支持嵌套。如 `"('float32','float32')"` 或 `"(('float32','float32'),'float32')"` |
| `tensor_formats` | DTYPE_NESTED | 否 | `('ND',)` | 张量格式，支持嵌套。如 `(('ND','ND'),'ND')` |
| `tensor_storage_shapes` | SHAPE_NESTED | 否 | `()` | 张量存储shape（用于非连续张量）。回退到视图shape。 |
| `tensor_view_offsets` | INT_NESTED | 否 | `()` | 张量视图在存储中的偏移量。 |
| `tensor_view_strides` | SHAPE_NESTED | 否 | `()` | 张量视图步长。未指定时自动从shape计算。 |

## 输出属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_tensor_indexes` | INT_TUPLE | 否 | *(自动)* | 指示哪些张量是输出的索引。未设置时按参数命名规则自动填充：以 `Ref`/`Out`/`Output`/`OutOptional`/`OutputOptional` 结尾或等于 `output` 的张量参数识别为输出；Backward/Grad API 会排除 `gradOutput`/`gradOut`/`grad_output`/`attentionOut` 以及非末位的 `output`；无匹配时回退到最后一个张量 |
| `output_inplace_indexes` | INT_TUPLE | 否 | `()` | 原地操作输出索引。无需手动填写：框架自动从算子信息库推断（当输出名称与输入名称相同时，表示输出覆写到该输入内存上） |

## 属性与标量

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `attributes` | DICT | 否 | `{}` | API属性参数。如 `{'dim': -1}`。也支持通过参数名指定小 shape 张量的值（适用于其数值不能由框架随机生成的场景，如索引参数、shape/size 参数等，其值与其他输入 shape 有关联）。如 `{'group_list': [1, 1, 1]}` |
| `scalar_dtypes` | DTYPE_NESTED | 否 | `()` | 标量参数数据类型。支持ScalarList嵌套。**不支持压缩/广播**（标量无 shape，数量只能通过 dtypes 元素数体现），必须逐个显式指定。如 `('int64',)` 或 `(('int32','int32'),)` |
| `scalar_data_ranges` | FLOAT_RANGE_NESTED | 否 | `((None, None),)` | 每个标量的数据范围 `(min, max)`。None 时 min=-2、max=2。标量值默认随机生成，也可在 `attributes` 中指定具体值（非必须） |

## Dump与手动数据

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dump_file_prefix` | STRING | 否 | `None` | 数据dump文件的自定义文件名前缀。 |
| `manual_tensor_binaries` | EVAL | 否 | `()` | 手动张量二进制文件路径。作为Python表达式求值。 |
| `manual_golden_binaries` | STRING_TUPLE | 否 | `()` | 手动Golden输出二进制文件路径。 |

## 非连续张量

通过 `tensor_view_strides`、`tensor_view_offsets`、`tensor_storage_shapes` 控制张量的非连续布局。视图 shape 和实际存储 shape 可以不同。未指定 strides/offsets/storage_shapes 时默认为连续张量。

| 字段组合 | 示例值 | 含义 |
|---------|--------|------|
| `tensor_view_shapes` | `"((2,3),)"` | 视图为 2x3 |
| `tensor_view_strides` | `"((4,1),)"` | 步长 (4,1)，行间 stride=4 |
| `tensor_view_offsets` | `"(5,)"` | 从存储第 5 个元素开始 |
| `tensor_storage_shapes` | `"((3,6),)"` | 存储 shape 为 3x6 |

## 通过 attributes 指定张量值

在 `attributes` 字典中按接口定义的参数名指定小 shape 张量值，适用于其数值不能由框架随机生成的场景。对于小 shape 张量，比 `manual_tensor_binaries` 或自定义输入插件更便捷。

| 场景 | 示例 | 说明 |
|------|------|------|
| 索引/shape/size 参数 | `{'group_list': [1, 1, 1]}` | 值与其他输入 shape 有关联，不能随机生成 |
| 指定具体值 | `{'value': 1.5}` | 为测试目的指定某个张量的具体值 |

## 批一致性字段

`--deterministic-level 3` 控制确定性执行。
只有下列三字段齐全才构成 relation，单独配置 level 3 仍可正常执行。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `batch_seed` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample seed` |
| `batch_axis` | FREE_EVAL | `None` | `输入槽 -> 逻辑轴` |
| `batch_slice_info` | FREE_EVAL | `None` | `输入槽 -> 轴组 -> sample 切片`，每个切片为 `(start, stop, step)` |

## 参考用例

`examples/case_store/aclnn/` 目录下的示例：

| 文件 | 验证特性 | 关键列 |
|------|---------|--------|
| `aclnn_add.csv` | 基础 aclnn API + 标量参数 | `scalar_dtypes` |
| `aclnn_cat.csv` | 拼接 `dim` 属性 × 6 dtype | `attributes`（dim）、`output_tensor_indexes` |
| `aclnn_convolution.csv` | 复杂属性（stride/padding/dilation/groups）解析 | `attributes`、`tensor_formats` |
| `aclnn_inplace_fill_tensor.csv` | 原地操作（inplace） | `input_data_ranges`、`precision_tolerances` |
| `aclnn_masked_select.csv` | bool mask 筛选 + 动态输出（elewise/broadcast） | `input_data_ranges`、`precision_tolerances` |
| `aclnn_nonzero_v2.csv` | `as_tuple` 属性分支 + 动态输出 | `attributes`（as_tuple） |
| `aclnn_split_tensor.csv` | 拆分为 TensorList 输出 + `splitSections`/`dim` | `attributes`、`output_tensor_indexes` |
| `aclnn_add.xlsx` | xlsx 多 sheet（T1/T2）输入验证 | — |
