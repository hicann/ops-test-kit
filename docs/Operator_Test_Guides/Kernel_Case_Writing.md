# Kernel 用例编写

适用于 `python3 -m ttk kernel` / `python3 -m ttk geir`，使用 `TestcaseOp`，共 26 个字段（含[公共字段](../Test_Case_Generation.md#公共字段所有模式通用)）。

具体shape通过 `input_shapes` / `output_shapes` 直接指定，动态shape由框架自动推导（将正数维度替换为 `-1`）。输入/输出的**数量和顺序**由算子定义文件决定，必须严格一一对应。TensorList 分组通过 shape 字段的嵌套结构表达，如 `(((3,3),(3,2)),(3,5))` 表示 TensorList(2) + 单个张量。**何时使用 TensorList 格式**：当算子信息库中 ParamType = DYNAMIC 时，即使只包含 1 个张量也必须用 TensorList 嵌套格式。

**算子定义与信息库来源**：
- **算子定义**：`xxx_def.cpp`（如 `add_def.cpp`），定义 INPUT/OUTPUT/ATTR 声明
- **算子信息库**：`aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`，包含每个输入/输出的 name、支持 dtype、支持 format、paramType（optional/dynamic）、ValueDepend 等
  - builtin 算子信息库路径：`{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{芯片系列号}/{算子仓}/aic-{芯片系列号}-ops-info-{算子仓[4:]}.json`
  - 自定义算子信息库路径：`{ASCEND_OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/config/{芯片系列号}/aic-{芯片系列号}-ops-info.json`
  - 例如 add 算子在 `ascend910b` 芯片上的 builtin 信息库：`$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ascend910b/ops_math/aic-ascend910b-ops-info-math.json`

## 身份标识

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `op_name` | STRING | **是** | *(无)* | 算子名称，优先使用算子信息库中 `opInterface.value` 的值（下划线格式）。若未配置 opInterface，则将驼峰算子名转为下划线格式即可。如 `add`、`mat_mul_v3`。 |

## 输入/输出Shape

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `input_shapes` | SHAPE_NESTED | **是** | `()` | 输入张量shape。支持TensorList嵌套。如 `"((128, 1024), (1, 1024))"` 表示两个张量，`"(((3,3),(3,2)),(3,5))"` 表示TensorList(2) + 单个张量。可选输入（paramType=optional）不传递时用 `None` 占位，对应的 dtype/format 也写 `None`。 |
| `input_dtypes` | DTYPE_NESTED | **是** | `()` | 输入数据类型。支持TensorList嵌套。支持广播：`('float32',)` 填充所有输入位置。如 `"('float32', 'float32')"` |
| `output_shapes` | SHAPE_INFER_NESTED | **是** | `None` | 输出张量shape。支持TensorList嵌套。如 `"((128, 1024),)"` |
| `output_dtypes` | DTYPE_NESTED | **是** | *(无)* | 输出数据类型。支持TensorList嵌套。支持广播：`('float32',)` 填充所有位置。如 `"('float32',)"` |

## 格式与原始Shape

| 字段 | 类型 | 是否必填 | 默认值 | 回退到 | 说明 |
|------|------|---------|--------|--------|------|
| `input_formats` | DTYPE_NESTED | 否 | `('ND',)` | *(无)* | 输入张量格式。如 `('ND', 'NCHW')`。支持广播：`('ND',)` 填充所有位置 |
| `input_ori_shapes` | SHAPE_NESTED | 否 | → `input_shapes` | `input_shapes` | 原始输入shape（格式转换前）。 |
| `input_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | `input_formats` | 原始输入格式（用于格式转换）。支持广播 |
| `output_formats` | DTYPE_NESTED | 否 | `('ND',)` | `output_ori_formats` | 输出张量格式。支持广播 |
| `output_ori_shapes` | SHAPE_INFER_NESTED | 否 | `None` | `output_shapes` | 原始输出shape。 |
| `output_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | `output_formats` | 原始输出格式。支持广播 |

## 属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `attributes` | DICT | 否 | `{}` | 算子属性（编译期和运行期）。如 `{'transpose_x1': True, 'transpose_x2': False}`。属性名匹配op_info输入名的会自动提取为特殊张量覆盖。当算子信息库中某个输入的 ValueDepend 为 OPTIONAL 或 REQUIRED 时，**必须**在 attributes 中指定该张量的值（如 ReduceMin 的 `axes`）。对于非 ValueDepend 但数值不能随机生成的小 shape 张量（如索引参数、shape 大小参数、size 参数等，其值与其他输入 shape 有关联），也可在 attributes 中指定，避免为小张量写输入定制函数（如 grouped_matmul 的 `group_list`）。 |

## 特殊属性

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `output_inplace_indexes` | INT_TUPLE | 否 | `()` | 原地操作输出索引。无需手动填写：框架自动从算子信息库推断（当输出名称与输入名称相同时，表示输出覆写到该输入内存上） |
| `output_shape_unknown_indexes` | INT_TUPLE | 否 | `()` | 编译期 shape 未知的**输出张量索引**。如 `"(0,)"` 表示第 0 个输出的 shape 在编译期无法确定 |

## 选项

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dump_file_prefix` | STRING | 否 | `None` | 数据dump文件的自定义文件名前缀。 |
| `manual_input_binaries` | EVAL | 否 | `()` | 手动输入二进制文件路径。作为Python表达式求值。支持TensorList嵌套。 |
| `manual_golden_binaries` | EVAL | 否 | `()` | 手动Golden输出二进制文件路径。支持TensorList嵌套。 |

## 参考用例

`examples/case_store/kernel/` 目录下提供了各种场景的示例：

| 文件 | 涵盖场景 |
|------|----------|
| `abs.csv` | 基本用例，多 dtype |
| `add.csv` | 多 dtype、广播、input_data_ranges |
| `concat_d.csv` | TensorList 输入（DYNAMIC） |
| `mat_mul_v3.csv` | 可选输入（None 占位）、属性 |
| `non_zero.csv` | 输出 shape 未知（output_shape_unknown_indexes） |
| `reduce_min.csv` | ValueDepend 张量输入（axes 通过 attributes 指定） |
| `split.csv` | TensorList 输出、动态参数 |
| `zeros_like.csv` | 基本用例 |
