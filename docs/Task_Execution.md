# Task Execution

[toc]

---

# Command Overview

```
python3 -m ttk {kernel,aclnn,e2e,info,list} [options]
```

| Subcommand | Purpose |
|-----------|---------|
| `kernel` | AscendC kernel compile + NPU execute + precision compare |
| `aclnn` | aclnn\* C API call + precision compare |
| `e2e` | PyTorch framework API end-to-end test (NPU/GPU/CPU) |
| `info` | Query local Ascend NPU device info |
| `list` | Preview test case names from CSV |

```shell
python3 -m ttk -v                    # Show version
python3 -m ttk kernel --help         # Show help
```

# Quick Start

```shell
# Kernel
python3 -m ttk kernel -i examples/case_store/kernel/mat_mul_v3.csv
python3 -m ttk kernel -i examples/case_store/kernel/add.csv -d -s
python3 -m ttk kernel -i examples/case_store/kernel/add.csv --co

# ACLNN
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv

# E2E
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --backend npu
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --backend cpu

# Device info & case preview
python3 -m ttk info
python3 -m ttk list -i cases.csv --op add
```

# Case Selection

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--testcase` | `-t` | Filter by case name (comma-separated) | `-t add_01,add_02` |
| `--testcase-index` | `--ti` | Filter by index | `--ti=1,3,5` or `--ti=1-10` |
| `--testcase-count` | `--tc` | Randomly pick N cases | `--tc=10` |
| `--operator` | `--op` | Filter by operator name | `--op add,mat_mul_v3` |
| `--exclude-operator` | `--no-op` | Exclude operator name | `--no-op concat_d` |
| `--priority` | | Filter by priority range | `--priority=1-3` |
| `--rerun` | | Rerun failed cases | `--rerun=precision_status` |

# Device & Parallelism

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--device` | `--dev` | Number of devices to use | All |
| `--device-whitelist` | | Device whitelist | None |
| `--device-blacklist` | | Device blacklist | None |
| `--process-count` | `--pc` | Processes per device | 1 |
| `--platform` | `--plat` | SoC version | Auto-detect |
| `--proc-timeout` | | Per-case timeout (seconds) | 0 (unlimited) |
| `--limit` | `-l` | Per-process HBM memory cap (bytes) | None |

# Precision Control

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--compare` | Comparison method | `close`/`cosine`/`binary`/`requant` | `close` |
| `--input-dist` | Input distribution | `uniform`/`normal` | `uniform` |
| `--seed` | Random seed (reproducible) | Integer | Random |
| `--golden-mode` | Golden generation mode | `Enable`/`Disable`/`Promote` | `Enable` |

> See [Result Analysis](./Result_Analysis.md) for comparison method details.

# Debug & Diagnostics

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--dump` | Dump data: `full`/`in`/`out`/`golden` | `--dump full` |
| `--dump-format` | Dump format: `bin`/`npy`/`pt`/`print` | `--dump-format npy` |
| `--dump-on-fail` | Auto-dump all data on precision failure | `--dump-on-fail` |
| `--single-log` | One log file per test case | `--single-log` |
| `--plugin` | External plugin path | `--plugin /path/to/plugin.py` |

# Output

| Parameter | Short | Description | Example |
|-----------|-------|-------------|---------|
| `--output` | `-o` | Output result CSV path | `-o results.csv` |
| `--title` | | Custom output columns | `--title testcase_name,precision_status` |
| `--csv-preserve` | | Preserve original CSV headers | `--csv-preserve` |

# Kernel Compilation Control

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--dynamic` | `-d` | Dynamic compilation | **On** |
| `--static` | `-s` | Static compilation | Off |
| `--const` | `-c` | Const compilation | Off |
| `--binary` | `-b` | Binary mode; `--binary=release` for released kernels | Off |
| `--auto` | | Auto-select compile mode from op info | Off |
| `--compile-only` | `--co` | Compile only, skip execution | Off |
| `--no-prof` | | Disable profiling | On |
| `--compile-opts` | | Compile options | None |
| `--core-type` | `--ct` | Core type: `AiCore`/`VectorCore` | Auto |
| `--impl-mode` | | Operator implementation mode | None |
| `--tiling-run` | `--tr` | Tiling run times | 3 |
| `--cce` | | Compile CCE file | Off |
| `--reuse-hbm` | | Reuse HBM across processes | Off |
| `--reserve-hbm` | | Reserve HBM (MB) | None |
| `--clear-atomic` | | Clear atomic-write region before exec | Off |
| `--clear-ub` / `--clear-l1` | | Clear UB / L1 before exec | Off |
| `--simt-ub` / `--simt-stack-dcu` | | SIMT-mode UB / DCU stack size | None |
| `--force-block-dim` | | Force `block_dim` value | None |

# E2E-Specific Parameters

| Parameter | Description | Options | Default |
|-----------|-------------|---------|---------|
| `--backend` | Hardware backend (NPU/GPU/CPU) | `npu`/`gpu`/`cpu` | Auto-detect |
| `--validate` | Validate CSV cases only, skip device execution | On/Off | Off |

> E2E mode runs through a unified Backend abstraction (`framework_api/backends/`); all three backends share the same case parsing and precision comparison pipeline. The CPU backend is commonly used as the Golden source.

# ACLNN-Specific Parameters

The ACLNN subcommand reuses the common parameters and has no mode-specific options.

# General Parameters

| Parameter | Short | Description | Default |
|-----------|-------|-------------|---------|
| `--input` | `-i` | CSV test case file (required) | |
| `--output` | `-o` | Output result CSV path | None |
| `--seed` | | Random seed | Random |
| `--print` | | Print summary info | On |
| `--no-memory-check` | | Skip host memory check | Off |
| `--proc-no-reuse` | | New process per case | Off |
| `--task-prof` | | Task-level profiling switch | On |
| `--po` / `--progress-output` | | Progress output path | None |
| `--run` | | Execution count | 3 (onboard) / 1 (model) |
| `--warmup` | | Warmup before profiling | On |
| `--npu-timeout` | | NPU execution timeout (ms) | Unlimited |
