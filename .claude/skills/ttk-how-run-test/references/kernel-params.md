# Kernel 模式专用参数

## 三种执行方式

| 方式 | 说明 | 适用场景 |
|------|------|---------|
| **动态 shape 编译**（默认） | 编译时维度为符号（-1），运行时适配不同 shape | 需要覆盖多种 shape 的通用测试 |
| **静态 shape 编译**（`-c`） | 按 CSV 中的具体维度值编译，生成专用 kernel | 对特定 shape 做性能调优 |
| **二进制执行**（`-b release`） | 直接使用 CANN 安装目录下预编译好的二进制 kernel，跳过编译 | 验证发布态 kernel 的精度和性能 |

## 命令示例

```shell
# 动态 shape 编译+执行（默认）
python3 -m ttk kernel -i cases.csv

# 关闭动态 shape 编译
python3 -m ttk kernel -i cases.csv -d false

# 静态 shape 编译+执行
python3 -m ttk kernel -i cases.csv -c

# 二进制模式（使用 CANN 安装目录下预编译二进制）执行
python3 -m ttk kernel -i cases.csv -b release

# 仅编译不执行
python3 -m ttk kernel -i cases.csv --co
```

## 编译控制

| 参数 | 缩写 | 说明 | 默认 |
|------|------|------|------|
| `--dynamic` | `-d` | 动态 shape 编译；`-d false` 关闭 | **开启** |
| `--const` | `-c` | 静态 shape 编译 | 关闭 |
| `--binary` | `-b` | 二进制编译；`-b release` 使用发布内核 | 关闭 |
| `--compile-only` | `--co` | 仅编译不执行 | 关闭 |
| `--no-prof` | | 不下设备执行（类似 dry-run，编译、输入生成、Golden生成等流程正常执行） | 开启 |
| `--compile-opts` | | 编译选项（KEY=VALUE 格式，可多次指定） | 无 |
| `--tiling-run` | `--tr` | Tiling 运行次数 | 3 |

## 内存控制

| 参数 | 说明 | 默认 |
|------|------|------|
| `--reuse-hbm` | 每个用例默认下发 3 次到 NPU 执行，复用同一块 HBM 内存以使能 L2 Cache | 关闭 |
| `--reserve-hbm` | 预留 HBM 内存（MB） | 无 |
| `--limit` / `-l` | 用例内存上限（GB），输入输出字节数之和超过则跳过 | 30 |

## 执行控制

| 参数 | 说明 | 默认 |
|------|------|------|
| `--run` | 执行次数 | 板端 3 次 |
| `--warmup` | Profiling 前预热 | 开启 |
| `--npu-timeout` | NPU 执行超时（ms） | 无限制 |
| `--clear-atomic` | 强制在算子执行前清零输出和 workspace | 关闭 |
| `--clear-ub` | 执行前将 UB 填充为指定值（默认清零） | 关闭 |
| `--clear-l1` | 执行前将 L1 填充为指定值（默认清零） | 关闭 |
| `--simt-ub` | SIMT 模式 UB 大小 | 无 |
| `--simt-stack-dcu` | SIMT 模式 DCU 栈大小 | 无 |
| `--force-block-dim` | 强制指定 block_dim | 无 |
