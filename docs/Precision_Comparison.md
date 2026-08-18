# 精度比对方法

通过 `--compare` 参数选择精度比对方法，或由 TestSpec 的 `tolerance` 字段按 dtype 逐输出路由。不同方法适用于不同的算子类型和精度要求。

## 方法总览

| 方法 | 参数值 | 公式核心 | 适用场景 | 默认容差 |
|------|--------|---------|---------|---------|
| 统计相对误差 | `stat_rel_err`（默认） | `|a-g| / (|g|+ε)` 的均值/最大值 | 浮点算子常规测试 | 按 dtype 阈值表 |
| 数值近似 | `close` | `|a-g| <= atol + rtol*|g|` | 逐点 isclose 比对 | rtol=0.001/0.0001, atol=1e-8 |
| 余弦相似度 | `cosine` | `dot(a,g) / (|a|*|g|)` | 大规模向量整体趋势 | 1-cos ≤ 0.01 |
| 二进制精确 | `binary` | 逐 bit SHA-256 比对 | 整型运算/精确结果 | 无容差 |
| 重量化 | `requant` | ULP 差 ≤ 1 | float8 类型（e5m2/e4m3fn/hifloat8） | ptol=0.001 |
| 量化 | `quant` | `|a-g| <= 1`（1 LSB） | 浮点输入 + int4/int8 量化输出 | ptol=0 |
| 三方交叉校验 | `cross_check` | NPU/XPU 误差比值 | 多硬件精度对齐 | 按 level 预设 |

> `a` = NPU 输出，`g` = Golden

## 1. 统计相对误差（stat_rel_err）

默认方法。按 dtype 统计相对误差的均值（mere）和最大值（mare），结合社区标准阈值判定。

### 公式

```
rel_err = |a - g| / (|g| + 1e-7)
mere = mean(rel_err)       # 平均相对误差
mare = max(rel_err)        # 最大相对误差
```

### 判定

```
passed = mere < threshold AND mare < 10 * threshold
```

- `mere` 超过 `threshold` → FAIL
- `mare` 超过 `10 * threshold` → FAIL（防止个别大误差被均值掩盖）

### dtype 阈值表

| dtype | threshold | 说明 |
|-------|-----------|------|
| `float16` | 2^-10 (≈0.000977) | 半精度 |
| `bfloat16` | 2^-7 (≈0.007812) | BF16 精度较低，阈值放宽 |
| `float32` | 2^-13 (≈0.000122) | 单精度 |
| `float8_e4m3fn` | 2^-3 (≈0.125) | FP8 E4M3 |
| `float8_e5m2` | 2^-2 (≈0.25) | FP8 E5M2 |
| 其他 | 2^-13 | 回退到 float32 级 |

### 特殊处理

- NaN/Inf 位置不一致 → 直接 FAIL（不计算 mere/mare）
- 双方同位置均为 NaN → 视为一致
- 双方同位置均为 Inf 且同号 → 视为一致

### 使能方式

```bash
# CLI（显式指定）
python3 -m ttk kernel -i cases.csv --compare stat_rel_err

# Spec.tolerance（按 dtype 路由，需 --plugin）
# 未指定 --compare 且无 Spec.tolerance 时自动生效
```

## 2. 数值近似（close）

使用 `np.isclose()` 逐元素比对，支持相对误差（rtol）和绝对误差（atol）双判据。

### 公式

```
passed = |a - g| <= atol + rtol * |g|
```

### 默认容差

| dtype | rtol | ptol | atol |
|-------|------|------|------|
| `float16` / `bfloat16` / `complex32` | 0.001 | 0.001 | 1e-8 |
| `float32` / 其他浮点 | 0.0001 | 0.0001 | 1e-8 |

- `rtol`：相对误差容差（大值域）
- `atol`：绝对误差容差（小值域，接近 0 时 rtol 失效）
- `ptol`：允许的不匹配元素比例（如 0.001 = 允许 0.1% 元素超出容差）

### 判定

```
precision = (golden_size - mismatch_count) / golden_size
passed = (1 - precision) <= ptol
```

### CSV 自定义容差

```csv
precision_tolerances,"((0.001, 0.001),)"
absolute_precision,1e-8
```

- `precision_tolerances`：每个输出的 `(rtol, ptol)` 对
- `absolute_precision`：全局 atol

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare close

# Spec.tolerance
# class AddTestSpec:
#     tolerance = {"float32": {"standard": "close"}}
```

## 3. 余弦相似度（cosine）

计算输出与 Golden 的余弦相似度，衡量向量方向一致性而非数值精度。

### 公式

```
cosine = dot(a, g) / (norm(a) * norm(g))
passed = (1 - cosine) <= rtol
```

### 默认容差

| 参数 | 默认值 |
|------|--------|
| `rtol` | 0.01（即 cosine ≥ 0.99） |

### 适用场景

- 大规模向量（如 embedding、attention 输出）
- 关注整体趋势而非逐点精度
- 不适合需要精确数值的场景

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare cosine

# Spec.tolerance
# class AddTestSpec:
#     tolerance = {"float16": {"standard": "cosine"}}
```

## 4. 二进制精确（binary）

逐 bit 比对 NPU 输出与 Golden，无容差概念。

### 判定

```
SHA-256(output_bytes) == SHA-256(golden_bytes) → PASS
```

不一致时，按 `itemsize` 分组定位差异元素索引。双方同位置均为 NaN 时视为相等。

### 适用场景

| 场景 | 说明 |
|------|------|
| 纯搬移算子 | 输出应与输入逐 bit 一致 |
| 纯比较算子 | 如 Equal、Less、Greater 等 |
| 整型运算 | int8/int32/uint 等精确结果 |
| 索引/掩码输出 | 非 0 即 1 的场景 |
| float4 / int4 | 打包格式逐 bit 比对 |

### 跨 dtype 处理

仅支持整型间的跨 dtype 比对（自动提升到公共类型）。浮点 vs 整型等不兼容组合直接拒绝。

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare binary

# Spec.tolerance
# class AddTestSpec:
#     tolerance = {"float16": {"standard": "binary"}}
```

## 5. 重量化（requant）

为 float8 类型设计，以 ULP（Unit in Last Place）为判据。

### 公式

```
diff = |view_as_int8(a) - view_as_int8(g)|    # FP8 按 int8 视图比较 ULP 差
passed = (diff <= 1 的元素占比) >= (1 - ptol)
```

### 默认容差

| dtype | ptol |
|-------|------|
| `float8_e5m2` / `float8_e4m3fn` / `hifloat8` | 0.001（允许 0.1% 元素 ULP 差 > 1） |

### 适用场景

- float8 量化算子
- FP8 精度对齐验证

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare requant

# Spec.tolerance
# class AddTestSpec:
#     tolerance = {"float8_e4m3fn": {"standard": "requant"}}
```

> 通常无需手动指定——`float8_e5m2` / `float8_e4m3fn` / `hifloat8` 会自动切换到 `requant`。

## 6. 量化（quant）

用于浮点输入 + int4/int8 量化输出的算子，判据为绝对误差 ≤ 1 LSB。

### 公式

```
diff = |int(a) - int(g)|
bad = count(diff > 1)
passed = (bad / golden_size) <= ptol
```

### 默认容差

| 参数 | 默认值 |
|------|--------|
| `ptol` | 0（不允许任何元素超差） |

### 适用场景

- 量化算子（如 PerTokenQuant）
- 浮点输入 → int4/int8 输出
- 量化末步 `round(x / scale)` 是阶跃函数，浮点末位 1 ULP 差异会导致结果跳到相邻整数格，逐位相等既不可能也无意义

### 使能方式

只能通过 Spec.tolerance 声明，不通过 `--compare` 指定。声明时会校验输出 dtype 为 int4/int8 且输入全为浮点：

```python
class PerTokenQuantTestSpec:
    __spec__ = "per_token_quant"
    tolerance = {
        "int8": {"standard": "quant", "ptol": 0.001},
    }
```

## 7. 三方交叉校验（cross_check）

将 NPU 输出、Golden、第三方（XPU）输出三方做误差比值比对。完整使用方法详见 [XPU 三方交叉校验与性能采集](./XPU_Cross_Check.md)。

### 公式

```
rel_npu   = |a - g| / (|g| + ε)          # NPU 相对误差
rel_party = |b - g| / (|g| + ε)          # XPU 相对误差

mare_ratio = max(rel_npu) / max(rel_party)     # 最大误差比
mere_ratio = mean(rel_npu) / mean(rel_party)   # 平均误差比
rmse_ratio = rmse(a,g) / rmse(b,g)             # RMSE 比
```

### 判定

大值域（`|g| ≥ small_value`）：三个比值均不超过 limit → PASS

小值域（`|g| < small_value`）：NPU 误差计数 / XPU 误差计数 ≤ 2.0 → PASS

### Level 预设

| Level | mare_ratio | mere_ratio | rmse_ratio | 适用 |
|-------|-----------|-----------|-----------|------|
| L0 | 10.0 | 2.0 | 2.0 | 宽松（初期对齐） |
| L1 | 5.0 | 1.5 | 1.5 | 默认 |
| L2 | 2.0 | 1.2 | 1.2 | 严格（接近对齐） |

### small_value 阈值

| dtype | small_value | small_value_atol |
|-------|------------|-----------------|
| `float16` | 2^-11 | 2^-16 |
| `bfloat16` | 2^-8 | 2^-16 |
| `float32` | 2^-14 | 2^-30 |

> `cross_check` 仅支持 `float16` / `bfloat16` / `float32`。

### 使能方式

```bash
# CLI
python3 -m ttk kernel -i cases.csv --compare cross_check --config ttk.conf.yaml --plugin /path/to/assets.py

# Spec.tolerance（声明 level 和 ratio）
# class AddTestSpec:
#     tolerance = {"bfloat16": {"standard": "cross_check", "level": "L2"}}
```

> 完整使用方法详见 [XPU 三方交叉校验与性能采集](./XPU_Cross_Check.md)。

## 自动切换规则

部分 dtype 会自动切换比对方法，无需手动指定：

| 数据类型 | 自动切换到 | 原因 |
|---------|-----------|------|
| `float8_e5m2` / `float8_e4m3fn` / `hifloat8` | `requant` | FP8 需 ULP 判据 |
| `float4` / `int4` | `binary` | 打包格式需逐 bit |
| `float8_e8m0` | `binary` | 无浮点语义 |
| 整型 / 布尔 | `binary` | 精确结果 |

## 优先级与路由

未显式指定 `--compare` 时，按以下优先级决定每个输出的比对方法：

```
1. CLI --compare 参数（最高优先级）
2. Spec.tolerance[dtype].standard（需 --plugin）
3. dtype 自动切换规则
4. stat_rel_err（默认兜底）
```

`quant` 例外：只能通过 Spec.tolerance 声明，`--compare` 中的浮点判据不会覆盖 quant 声明。

## 容差自定义

### CSV 字段

| 字段 | 作用 | 格式 |
|------|------|------|
| `precision_tolerances` | 每输出的 `(rtol, ptol)` | `((0.001, 0.001),)` |
| `absolute_precision` | 全局 atol | `1e-8` |

### Spec.tolerance

```python
class AddTestSpec:
    __spec__ = "add"
    tolerance = {
        "float16": {"standard": "stat_rel_err", "threshold": 0.002},
        "float32": {"standard": "close"},
        "bfloat16": {"standard": "cross_check", "level": "L2"},
        "int8": {"standard": "quant", "ptol": 0.001},
    }
```

## 方法选择建议

| 场景 | 推荐方法 | 参数 |
|------|---------|------|
| 浮点算子常规测试 | 统计相对误差 | `--compare stat_rel_err`（默认） |
| 逐点 isclose | 数值近似 | `--compare close` |
| 大规模向量整体趋势 | 余弦相似度 | `--compare cosine` |
| 整型/索引/掩码/纯搬移/纯比较 | 二进制精确 | `--compare binary` |
| float8 量化 | 重量化 | `--compare requant`（通常自动） |
| int4/int8 量化输出 | 量化 | Spec.tolerance 声明 `quant` |
| 多硬件精度对齐 | 三方交叉校验 | `--compare cross_check`（需 XPU） |
