### 目的

- 1个算子1个用例，用于用例书写借鉴。
- 作为门槛，TTK任何改动必须执行这里的用例。
- xlsx 示例验证 CSV 与 Excel 输入等价、多 sheet 选择（`--sheet`）与默认输出命名。

> 各通路每个用例的「验证特性 + 关键列」详见对应 Case Writing 指南：
> [Kernel](../../docs/Operator_Test_Guides/Kernel_Case_Writing.md) ·
> [GEIR](../../docs/Operator_Test_Guides/GEIR_Case_Writing.md) ·
> [ACLNN](../../docs/Operator_Test_Guides/ACLNN_Case_Writing.md) ·
> [E2E](../../docs/Operator_Test_Guides/E2E_Case_Writing.md)。

### Excel（.xlsx）多 sheet 示例

每通路一个 2-sheet 工作簿（T1/T2），验证 xlsx 输入与 `--sheet` 切换（默认首个工作表；默认输出名带实际 sheet 名，如 `add_T2_result.csv`，多 sheet 互不覆盖）：

运行示例：

```shell
python3 -m ttk kernel -i examples/case_store/kernel/add.xlsx            # 默认首个 sheet
python3 -m ttk kernel -i examples/case_store/kernel/add.xlsx --sheet T2
```
