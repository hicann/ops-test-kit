# GEIR 用例编写

适用于 `python3 -m ttk geir`，使用 `GeirTestcase`，继承自 Kernel 的 `TestcaseOp`。共 27 个字段（Kernel 的 26 个字段 + GEIR 专属 1 个字段）。

Kernel 模式的 26 个字段说明详见 [Kernel用例编写](./Kernel_Case_Writing.md)。

## GEIR 专属字段

| 字段 | 类型 | 是否必填 | 默认值 | 说明 |
|------|------|---------|--------|------|
| `dyn_input_shapes` | SHAPELIKE_DYN_NESTED | 否 | `None` | 动态编译时输入张量的图编译 desc_shape。动态 shape 编译模式下，未指定时框架自动将所有正数维度替换为 `-1`；指定后按给定值构建图的 shape 描述。如 `"((128, -1), (1, 1024))"` 表示第一个输入第 0 维固定 128、第 1 维动态，第二个输入固定。 |

## dyn_input_shapes 与 input_shapes 的关系

| 字段 | 作用 | 示例 |
|------|------|------|
| `input_shapes` | 实际执行时的数据 shape | `"((128, 1024), (1, 1024))"` |
| `dyn_input_shapes` | 图编译时的 desc_shape，`-1` 表示该维度动态 | `"((128, -1), (1, 1024))"` |

未指定 `dyn_input_shapes` 时，动态模式下等价于将 `input_shapes` 的正数维度全部替换为 `-1`。

## 参考用例

GEIR 复用 Kernel 的 CSV 用例（字段与验证场景见 [Kernel 参考用例](./Kernel_Case_Writing.md)），额外用 `dyn_input_shapes` 列控制图编译时的动态维度。本通路独立示例：

| 文件 | 验证特性 | 关键列 |
|------|---------|--------|
| `geir/add.xlsx` | xlsx 多 sheet（T1/T2）输入 + `dyn_input_shapes` 图编译动态 shape | `dyn_input_shapes` |

> 直接使用 `examples/case_store/kernel/` 下的 CSV 跑 GEIR 无需改动；如需精确控制某输入维度的动态性，在末尾追加 `dyn_input_shapes` 列即可。
