# 确定性计算与batch一致性

`--deterministic-level`（缩写 `--dl`）控制一次 TTK 执行的确定性等级。level 3 仍先执行普通精度
校验；FA 算子的 case 内/跨 case batch 一致性则由 dump 驱动的第二阶段单独完成，避免把没有可切片
输出的框架摘要误判为一致。

## 等级与优先级

| 等级 | 执行含义 | 结果判据 |
|------|----------|----------|
| `0` | 不启用确定性选项 | 普通精度 |
| `1` | 同一用例的确定性执行 | 通路原有的同用例确定性检查 |
| `2` | 编译期与执行期确定性选项 | 通路原有的强确定性检查 |
| `3` | 确定性执行；若完整声明 batch relation，则可进入第二阶段 | 第一阶段普通精度；第二阶段 raw dump slice 比对 |

可以在 CSV 的 `attributes` 中为单个用例写
`'batch_deterministic_level': 3`。当命令行 `--dl` 为 `0`（默认）时，它与该用例指定
`--dl=3` 等价；非零命令行值始终优先，例如 `--dl=1` 会覆盖 CSV 的 level 3。该属性只决定
确定性执行等级，不会自动构成 batch relation。

只有 `batch_axis`、`batch_slice_info`、`batch_seed` 三个字段都存在且可解析时，才构成 batch
relation。三者都没有或只填了其中一部分时，level 3 仍正常执行确定性/精度流程，但不进入第二阶段，
也不会因为缺字段失败；执行前会按用例输出一次 warning，列出缺少的字段并提示继续普通 level-3
执行。prepare/validate 不执行主算子，不输出该执行提示。
