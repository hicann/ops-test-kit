---
name: ttk-how-write-case
description: 编写测试用例。只要用户提到写用例、生成 CSV、构造测试数据，或问"shape 怎么填"、"dtype 怎么写"、"CSV 格式报错"、"用例怎么构造"、"TensorList 怎么表达"，就必须使用此 skill。涵盖 Kernel、ACLNN、E2E 三种模式的用例编写。
---

# 编写 CSV 测试用例

## Step 1: 确定模式

三种模式对应不同的测试层级，使用不同的用例结构类。**CSV 表头字段决定了框架用哪个类解析**，选错模式会导致解析失败。

| 判断条件 | 模式 | 结构类 | 子命令 | 完整指南 |
|----------|------|--------|--------|----------|
| 无 `api_name` 列 | Kernel（算子编译链） | `TestcaseOp` | `ttk kernel` | `references/kernel-case.md` |
| `api_name` 以 `aclnn` 开头 | ACLNN（引擎 API 链） | `TestcaseAclnn` | `ttk aclnn` | `references/aclnn-case.md` |
| `api_name` 非 aclnn 开头 | E2E（框架 API 链） | `TestcaseE2e` | `ttk e2e` | `references/e2e-case.md` |

确定模式后，阅读对应的 reference 文件，内含该模式的必填字段、完整字段表、常见模式和示例。

## Step 2: 公共字段（9个，所有模式通用）

| 字段 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `testcase_name` | 是 | 自动生成 | 用例唯一名称 |
| `network_name` | 否 | None | 网络标签（如 `llama3_70b_train`） |
| `input_data_ranges` | 否 | ((None,None),) | 每输入随机范围 `[min,max]`（闭区间，min 和 max 为边界值）。None 时 min=-2，max=2。可指定第 3 个及以上的必现值，会混入随机数据中保证出现。min==max 时生成固定值。如 `"((-1,1),(0,10))"` 或 `"((-1,1,0),(0,10))"` |
| `precision_tolerances` | 否 | None | 每输出容差 `(rtol,atol)`。如 `"((0.001,0.001),)"` |
| `absolute_precision` | 否 | 1e-8 | 绝对精度容差 |
| `is_enabled` | 否 | True | False 跳过用例 |
| `remark` | 否 | None | 备注 |
| `soc_series` | 否 | None | SoC 过滤。`('Ascend910A','-Ascend310P')` |
| `priority` | 否 | 0 | 优先级数值（整数），配合 `--priority` 按范围筛选 |

## Step 3: 格式规则（所有模式通用）

CSV 解析器以逗号为字段分隔符。**shape 元组和 dtype 列表内部也包含逗号**，如果不加双引号包裹，CSV 解析器会把一个字段截断成多个字段，导致解析失败。

| 规则 | 正确 | 错误 |
|------|------|------|
| 含括号/逗号必须双引号 | `"((2,3),(2,3))"` | `((2,3),(2,3))` |
| dtype 用字符串 | `"('float32','int8')"` | `(float32,int8)` |
| 字典用双引号包裹，key 用单引号 | `"{'dim': -1}"` | `{'dim': -1}` |
| 不需要填写的字段留空即可，无需双引号 | `add_01,,add,...` （比如第 2 个字段 network_name 不填） | `add_01,"",add,...` 或 `add_01,None,add,...` |
| 可选输入/输出以 None 占位 | `"((128,1024),None,None)"` | `"((128,1024),,)"` 或 `"((128,1024),null,null)"` |
| 布尔值用 Python 格式 | `"{'transpose': True}"` | `"{'transpose': true}"` |

### dtype 字符串

`float16`/`fp16`, `float32`/`fp32`, `float64`/`fp64`, `bfloat16`/`bf16`, `int8`~`int64`, `uint8`~`uint64`, `bool`, `complex64`, `complex128`

## Step 4: 验证

写完 CSV 后用 `--validate` 校验（不下设备，三种模式均支持）：

```shell
python3 -m ttk kernel -i cases.csv --validate
python3 -m ttk aclnn -i cases.csv --validate
python3 -m ttk e2e -i cases.csv --validate

# 预览用例列表
python3 -m ttk list -i cases.csv
```

## 相关 Skill

- 写完用例怎么跑？→ `ttk-how-run-test`
- 用例跑失败了？→ `ttk-how-diagnose`
- 算子没有内置 Golden？→ `ttk-how-write-plugin`
