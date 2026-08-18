# Dump 数据调试

通过 `--dump` 系列参数保存输入、输出和 Golden 数据到磁盘，便于事后分析精度问题、复现失败用例、或离线比对。

## 适用场景

| 场景 | 命令组合 | 说明 |
|------|---------|------|
| 主动 Dump 全量数据 | `--dump full` | 保存输入+输出+Golden |
| 选择性 Dump | `--dump in,golden` | 只保存指定类别 |
| 失败时自动 Dump | `--dump-on-fail` | 精度比对失败时自动保存全部数据 |
| 离线数据准备 | `--no-prof --dump in,golden` | prepare 阶段，详见[离线数据准备与导入](./Offline_Data_Prepare_and_Import.md) |

## 1. Dump 类别

| 参数值 | 保存内容 | 时机 |
|--------|---------|------|
| `in` | 输入 tensor（含 tiling data） | 输入生成后、设备执行前 |
| `out` | NPU 输出 tensor | 设备执行后、精度比对前 |
| `golden` | Golden tensor | Golden 生成后 |
| `full` | 以上全部 | 各阶段分别保存 |

可组合指定，逗号分隔：`--dump in,golden`。

## 2. Dump 格式

通过 `--dump-format` 指定保存格式：

| 格式 | 参数值 | 说明 | 适用分析方式 |
|------|--------|------|-------------|
| 二进制 | `bin`（默认） | 原始字节，无 shape 信息 | 十六进制查看、按 dtype 手动解析 |
| NumPy | `npy` | `.npy` 文件，含 dtype 和 shape | `numpy.load()` 直接加载 |
| PyTorch | `pt` | `.pt` 文件，含 dtype 和 shape | `torch.load()` 直接加载 |
| 终端打印 | `print` | 直接输出到终端 | 快速目视检查 |

### 2.1 格式选择建议

- **npy**：最推荐，Python 生态通用，`numpy.load()` 即可用
- **pt**：需要 Torch 张量语义（含自定义 dtype）时使用
- **bin**：最小体积，但需额外信息（dtype/shape）才能解析
- **print**：仅用于小规模数据快速查看

### 2.2 特殊 dtype 处理

`complex32` 在 npy/pt 格式下按 `float16` 存储；`int4`/`float4` 在 bin 格式下为打包字节，npy/pt 格式下自动解包。

## 3. 文件命名与路径

### 3.1 输出路径

Dump 文件默认保存到 TTK 工作目录（`root_path`）。E2E 模式可通过环境变量 `NPU_DUMP_PATH` 覆盖：

```bash
export NPU_DUMP_PATH=/data/dump
python3 -m ttk e2e -i cases.csv --dump full
```

### 3.2 文件命名规则

```
<testcase_name>_<phase>_<index>.<ext>
```

| 通配 | 说明 | 示例 |
|------|------|------|
| `<testcase_name>` | 用例名 | `add_01` |
| `<phase>` | 阶段标识 | `input` / `output` / `golden` |
| `<index>` | tensor 序号（从 0 开始） | `0` / `1` |
| `<ext>` | 格式后缀 | `bin` / `npy` / `pt` |

Kernel 模式额外保存 tiling data：

```
add_01_dyn_input_0.npy          # 动态 shape 输入
add_01_dyn_input_1.npy
add_01_dyn_tiling_data.npy      # 动态 tiling
add_01_dyn_output_0.npy         # 动态输出
add_01_dyn_output_1.npy
add_01_golden_0.npy             # Golden
```

GEIR 模式文件名带 `geir` 前缀：

```
add_01_geir_input_0.npy
add_01_geir_output_0.npy
add_01_geir_golden_0.npy
```

失败时自动 Dump 的文件名带 `fail_` 前缀：

```
add_01_fail_input_0.npy
add_01_fail_output_0.npy
add_01_fail_golden_0.npy
```

## 4. 失败时自动 Dump

### 4.1 命令

```bash
python3 -m ttk kernel -i cases.csv --dump-on-fail
```

### 4.2 行为

- 精度比对**失败**时自动保存该用例的全部数据（输入+输出+Golden）
- 比对**通过**时不保存
- 不需要同时指定 `--dump`，`--dump-on-fail` 独立生效
- 文件名带 `fail_` 前缀，与正常 dump 区分

### 4.3 与 `--dump` 组合

可同时使用：`--dump full --dump-on-fail`。此时所有用例都保存正常数据，失败用例额外保存一份 `fail_` 前缀副本。

## 5. 典型调试流程

### 5.1 单用例定位

```bash
# 1. 先跑一遍，用 --dump-on-fail 抓失败数据
python3 -m ttk kernel -i cases.csv -t add_01 --dump-on-fail --single-log

# 2. 用 npy 格式重新 dump，方便 Python 加载
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy

# 3. Python 中加载分析
python3 -c "
import numpy as np
out = np.load('add_01_dyn_output_0.npy')
golden = np.load('add_01_golden_0.npy')
diff = np.abs(out - golden)
print('max diff:', diff.max())
print('argmax:', np.unravel_index(diff.argmax(), diff.shape))
"
```

### 5.2 全量回归抓取

```bash
# 全量用例失败时自动 dump，npy 格式
python3 -m ttk kernel -i cases.csv --dump-on-fail --dump-format npy
```

### 5.3 固定随机种子复现

```bash
python3 -m ttk kernel -i cases.csv -t add_01 --dump full --dump-format npy --seed 42
```

`--seed` 固定输入数据生成和 Golden 计算的随机性，确保可复现。

## 6. 参数约束

| 约束 | 说明 |
|------|------|
| `--dump` 与 `--no-prof` | `--no-prof` 只允许 `--dump in,golden`（prepare 模式） |
| `--dump-format print` 与 `--no-prof` | 不兼容，prepare 需要可回读格式（bin/npy/pt） |
| `--dump-on-fail` 与 `--no-prof` | 不兼容，`--no-prof` 跳过比对 |
| `--dump-on-fail` 与 `--validate` | 不兼容，`--validate` 跳过执行和比对 |

## 7. 通路支持

所有通路（Kernel / ACLNN / GEIR / E2E）均支持 `--dump` 和 `--dump-on-fail`。
