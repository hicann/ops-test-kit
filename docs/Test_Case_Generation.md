# 用例生成


---

# 简介

TTK使用CSV文件批量定义测试用例。第一行为表头（列名），后续每行代表一个测试用例。不同测试模式使用不同的列定义。

**模式自动识别逻辑**（基于CSV表头 + 子命令）：
- 表头包含 `api_name`：第一个非空 `api_name` 值不以 `aclnn` 开头 → **E2E模式**；以 `aclnn` 开头或全部为空 → **ACLNN模式**
- 表头不包含 `api_name`：由子命令决定 — `ttk geir` → **GEIR模式**，`ttk kernel` → **Kernel模式**

各模式的执行命令见 [任务执行](./Task_Execution.md)。

---

# 公共字段（所有模式通用）

以下9个字段为Kernel、GEIR、ACLNN、E2E四种模式共有。

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `testcase_name` | STRING | **是** | 自动生成 | 用例唯一名称。缺失时自动生成为 `auto_testcase_name_N`。 |
| `network_name` | STRING | 否 | `None` | 网络/模型名称标签（如 `llama3_70b_train`）。 |
| `input_data_ranges` | FLOAT_RANGE_NESTED | 否 | `((None, None),)` | 每个输入张量的随机数据范围。每个元素为 `[min, max]`（闭区间，min 和 max 为边界值），None 时 min=-2、max=2。第 3 个及以上的值为必现值，会混入随机数据中保证出现。min==max 时生成固定值。支持TensorList嵌套。如 `"((-1, 1), (0, 10))"` 或 `"((-1, 1, 0), (0, 10))"` |
| `precision_tolerances` | FLOAT_RANGE_NESTED | 否 | `None` | 每个输出的精度容差对 `(rtol, atol)`。如 `"((0.001, 0.001),)"` |
| `absolute_precision` | FLOAT_OR_NESTED | 否 | `1e-8` | 默认绝对精度容差。可以是单个浮点数或嵌套容器实现逐输出控制。 |
| `is_enabled` | BOOL | 否 | `True` | 设为 `False` 跳过此用例。 |
| `remark` | STRING | 否 | `None` | 自由备注信息。 |
| `soc_series` | STRING_TUPLE | 否 | `None` | SoC过滤。前缀 `-` 表示排除。如 `('Ascend910A', '-Ascend310P')` |
| `priority` | INT | 否 | `0` | 优先级，用于选择性执行。 |

---

# 各模式特殊字段

三种模式在公共字段基础上各有专属字段。

| 模式 | 子命令 | 专属字段数 | 核心标识字段 | 详见 |
|------|--------|-----------|------------|------|
| Kernel | `ttk kernel` | 17 | `op_name`、`input_shapes`、`output_shapes`、`attributes` | [Kernel用例编写](./Operator_Test_Guides/Kernel_Case_Writing.md) |
| GEIR | `ttk geir` | 18 | `op_name`、`input_shapes`、`dyn_input_shapes` | [GEIR用例编写](./Operator_Test_Guides/GEIR_Case_Writing.md) |
| ACLNN | `ttk aclnn` | 18 | `api_name`、`tensor_view_shapes`、`output_tensor_indexes`、`scalar_dtypes` | [ACLNN用例编写](./Operator_Test_Guides/ACLNN_Case_Writing.md) |
| E2E | `ttk e2e` | 14 | `api_name`、`tensor_view_shapes`、`golden_api` | [E2E用例编写](./Operator_Test_Guides/E2E_Case_Writing.md) |

> GEIR 继承 Kernel 的字段，额外多 `dyn_input_shapes` 用于控制图编译时的动态 shape 描述。

---

# 数据类型字符串

| 数据类型 | 字符串表示 |
|---------|-----------|
| 16位浮点 | `float16`, `fp16` |
| 32位浮点 | `float32`, `fp32` |
| 64位浮点 | `float64`, `fp64` |
| BFloat16 | `bfloat16`, `bf16` |
| 8位有符号整数 | `int8` |
| 16位有符号整数 | `int16` |
| 32位有符号整数 | `int32` |
| 64位有符号整数 | `int64` |
| 8位无符号整数 | `uint8` |
| 16位无符号整数 | `uint16` |
| 32位无符号整数 | `uint32` |
| 64位无符号整数 | `uint64` |
| 布尔型 | `bool` |
| 32位复数 | `complex32`, `c32` |
| 64位复数 | `complex64`, `c64` |
| 128位复数 | `complex128`, `c128` |

# CSV字段值格式参考

| 类型 | 格式 | 示例 |
|------|------|------|
| shape元组 | 双引号包裹的嵌套元组 | `"((128, 1024), (1, 1024))"` |
| dtype元组 | 双引号包裹的字符串元组 | `"('float32', 'float32')"` |
| 嵌套shape（TensorList） | 双引号包裹的三层嵌套元组 | `"(((3,3),(3,2)),(3,5))"` |
| 字典 | 直接写Python字典 | `{'dim': -1}` |
| 数值 | 直接写数值 | `1e-8`, `0`, `True` |
| 空值 | 留空 | 两个连续逗号 `,,` |
| None输入 | `None` 关键字 | `None` |

> **注意**：包含特殊字符（括号、逗号、引号）的字段必须用双引号包裹。

# 回退链

部分字段在未显式设置时自动回退到其他字段：

| 字段 | 回退到 |
|------|--------|
| `input_ori_shapes` | `input_shapes` |
| `output_ori_shapes` | `output_shapes` |
| `input_ori_formats` | `input_formats` |
| `output_ori_formats` | `output_formats` |
| `input_formats` | *(无回退，默认 `('ND',)`)* |
| `output_formats` | `output_ori_formats` |
