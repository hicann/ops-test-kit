# 确定性计算与批一致性

通过 `--deterministic-level`（缩写 `--dl`）控制 NPU 执行的确定性等级，用于验证算子输出的可复现性和跨用例一致性。

## 等级说明

| 等级 | 参数值 | 验证目标 | 判定方式 |
|------|--------|---------|---------|
| 关闭 | `0`（默认） | 无确定性要求 | — |
| 确定性计算 | `1` | 同一用例多次执行结果一致 | NPU 输出 MD5 一致 |
| 强一致 | `2` | 编译期确定性 + 执行期确定性 | GE 侧 `ge.deterministicLevel=1` + 设备侧 |
| 批一致性 | `3` | 跨用例输出切片一致 | 同 `batch_consistency_id` 分组，切片 MD5 比对 |

## 1. 确定性计算（level=1）

### 1.1 适用场景

验证同一用例在 NPU 上多次执行结果完全一致（位级），用于排查随机性引入的精度波动。

### 1.2 命令

```bash
python3 -m ttk aclnn -i cases.csv --dl=1
python3 -m ttk e2e  -i cases.csv --dl=1
python3 -m ttk geir -i cases.csv --dl=1
```

### 1.3 工作机制

- **ACLNN**：调用 `set_deterministic_level(1)`，多次执行后比对输出 MD5
- **E2E**：调用 `torch_npu.npu.set_deterministic_level(1)`，影响框架层执行
- **GEIR**：编译期注入 `ge.deterministicLevel=1` 到 GE 图选项
- **Kernel**：不支持

### 1.4 结果判定

执行完成后，TTK 比对多次运行的输出字节 MD5。不一致时标记为失败并在日志中打印差异。

## 2. 批一致性（level=3）

### 2.1 适用场景

验证不同用例的输出切片是否位级一致。典型用途：同一算子以不同 batch 切片方式执行，验证切片拼接后结果与完整执行一致。

### 2.2 前置条件

CSV 中必须配置以下字段，用于定义切片关系：

| CSV 字段 | 作用 | 示例 |
|---------|------|------|
| `batch_seed` | 分组标识，相同 seed 的用例归为一组 | `(100,)` |
| `batch_axis` | 切片所在轴（嵌套列表） | `(([0],),)` |
| `batch_slice_info` | 切片范围 (start, stop, step) | `(([[0,5,1]],),)` |

`batch_consistency_id` 由 `batch_seed` + 切片长度自动生成。相同 seed 且切片长度相同的用例归入同一比对组。

### 2.3 CSV 示例

```csv
testcase_name,op_name,tensor_view_shapes,batch_seed,batch_axis,batch_slice_info,...
slice_0,add,"((5,8),)",(100,),(([0],),),(([[0,5,1]],),),...
slice_1,add,"((5,8),)",(100,),(([0],),),(([[5,10,1]],),),...
full,add,"((10,8),)",(100,),,,...
```

上例中 `slice_0` 和 `slice_1` 各取前 5 行和后 5 行，`full` 取全部 10 行。三者 `batch_seed` 相同，切片长度一致（5+5=10），归入同一比对组。

### 2.4 命令

```bash
python3 -m ttk e2e -i cases.csv --dl=3
python3 -m ttk aclnn -i cases.csv --dl=3
```

> **GEIR 不支持 level=3**：执行时会打印 warning 并忽略，不影响正常精度比对。

### 2.5 工作机制

1. 所有用例正常执行，输出 `output_bytes` 和 `batch_consistency_id` 被收集
2. 执行结束后，按 `batch_consistency_id` 分组
3. 每组内按 `batch_axis` / `batch_slice_info` 提取输出切片
4. 计算各切片 MD5，组内 MD5 全部一致则 PASS，否则 FAIL

### 2.6 结果输出

批一致性结果在测试结束后统一打印：

```
Batch consistency: 2/3 groups passed
  [PASS] group=a1b2c3... members=['slice_0', 'slice_1', 'full']
  [FAIL] group=d4e5f6... members=['slice_a', 'slice_b']
```

## 3. 参数约束

| 约束 | 说明 |
|------|------|
| `--dl=3` 与 GEIR | 不兼容，GEIR 会忽略并打印 warning |
| `--dl=3` 与 NPUSim | 不兼容，仿真模式强制 `deterministic_level=0` |
| `--dl=3` 与 `--no-prof` | 不兼容，`--no-prof` 跳过执行 |
| level=3 需要 `--run>=1` | 至少执行一次才能收集输出 |

## 4. 通路支持

| 通路 | level=1 | level=2 | level=3 |
|------|---------|---------|---------|
| Kernel | 不支持 | 不支持 | 不支持 |
| GEIR | 支持 | 不支持（no-op） | 不支持（忽略） |
| ACLNN | 支持 | 支持 | 支持 |
| E2E | 支持 | 支持 | 支持 |

> Kernel 的 testcase 未定义 `batch_consistency_id` 等 CSV 字段，level=3 无法分组比对。
