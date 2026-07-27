# Kernel Operator Test Guide

[toc]

---

# Environment Setup

Python 3.8+, PyTorch (for golden computation), CANN toolkit installed.

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit && pip install -r requirements.txt
```

# Write Test Cases

See [Test Case Generation](../Test_Case_Generation.md) for all CSV fields.

Example `add.csv`:

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
add_01,,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
```

With compilation params (MatMulV3):

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
matmul_512_1_1792__1792_256,llama3_70b_train,mat_mul_v3,"((512, 1792), (1792, 256), None, None)","('bfloat16', 'bfloat16', 'float32', 'int8')","('ND',)","((512, 256),)","('bfloat16',)","('ND',)","((512, 1792), (1792, 256), None, None)","('ND',)","((512, 256),)","('ND',)","{'transpose_x1': False, 'transpose_x2': False, 'offset_x': 0, '#enable_pad': 1}","((-1, 1),)","((0.001, 0.001),)",1e-08,(),(),True,,,0,,(),()
```
```

More examples in `examples/case_store/kernel/`.

# Precision Testing

```shell
python3 -m ttk kernel -i add.csv
python3 -m ttk kernel -i add.csv -d
python3 -m ttk kernel -i add.csv --dev 0
python3 -m ttk kernel -i add.csv -o results.csv
```

## Execution Flow

```
Read CSV -> Compile/match kernel and run tiling -> Generate inputs -> Generate golden (CPU) -> Execute on NPU -> Precision compare -> Output results
```

## Compile Modes

| Flag | Mode | Description |
|------|------|-------------|
| `-d` (default) | Dynamic shape | Compile with dynamic shapes, run tiling, then execute |
| `-c` | Static shape | Compile with fixed shapes |
| `-b release` | Binary | Use pre-compiled release kernels |

```shell
python3 -m ttk kernel -i add.csv --co          # Compile only
python3 -m ttk kernel -i add.csv --no-prof     # Disable profiling
```

# Two-Stage Input/Golden Execution

Use the exact prepare pair to generate input and CPU golden without executing the
target Kernel, then replay the data on the target device:

```shell
python3 -m ttk kernel -i add.csv --plugin /path/to/kernel_assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/add

python3 -m ttk kernel -i add.csv --plugin /path/to/kernel_assets \
  --manual-data-dirs /data/add
```

Use the same `-d`, `-c`, or `-b release` selection in both commands. A standalone
`--no-prof` remains the original Kernel dry run; `--co` stops before input/golden
generation and cannot be combined with manual-data prepare or replay. See
[Manual-Data Prepare and Replay](../Manual_Data_Prepare_and_Replay.md) for file
formats, directory transfer, plugin requirements, and validation rules.

# Performance Testing

```shell
python3 -m ttk kernel -i add.csv
python3 -m ttk kernel -i add.csv --run=5
python3 -m ttk kernel -i add.csv --warmup
```

# Multi-Card Parallel

```shell
python3 -m ttk kernel -i add.csv               # All available cards
python3 -m ttk kernel -i add.csv --dev=2
python3 -m ttk kernel -i add.csv --pc=2
python3 -m ttk kernel -i add.csv --device-whitelist=0,1
```

# Common Examples

```shell
python3 -m ttk kernel -i examples/case_store/kernel/mat_mul_v3.csv
python3 -m ttk kernel -i examples/case_store/kernel/split.csv -c
python3 -m ttk kernel -i examples/case_store/kernel/concat_d.csv
python3 -m ttk kernel -i add.csv -t add_01 --dump-on-fail
python3 -m ttk kernel -i add.csv --rerun=precision_status
python3 -m ttk kernel -i add.csv --plugin /path/to/my_golden.py
```
