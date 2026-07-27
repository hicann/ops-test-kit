# ACLNN Operator Test Guide

[toc]

---

# Environment Setup

Python 3.8+, CANN toolkit (with aclnn headers and shared libraries).

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit && pip install -r requirements.txt
```

# Write Test Cases

ACLNN mode uses `api_name` (not `op_name`). See [Test Case Generation](../Test_Case_Generation.md).

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
aclnnCat_float,aclnnCat,"(((3,3),(3,2)),(3,5),)","(('float32','float32'),'float32')",{'dim': -1},"(1,)"
```

## Key Fields

- **tensor_view_shapes**: Nested for TensorList. `(((3,3),(3,2)),(3,5),)` = first input is TensorList of 2 tensors.
- **output_tensor_indexes**: Which positions are outputs. `"(1,)"` = 2nd param is output.
- **attributes**: Non-tensor params. E.g. `{'dim': -1}`
- **scalar_dtypes**: Scalar param types. E.g. `('float32',)`

## Inplace Operation

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes,output_inplace_indexes
aclnnInplaceFill_01,aclnnInplaceFillTensor,"((3,4,5),)","('float32',)","{'value': 1.5}","(0,)","(0,)"
```

# Precision Testing

```shell
python3 -m ttk aclnn -i aclnn_cat.csv
python3 -m ttk aclnn -i aclnn_cat.csv --compare cosine
python3 -m ttk aclnn -i aclnn_cat.csv --dev 0
python3 -m ttk aclnn -i aclnn_cat.csv -o results.csv
```

## Execution Flow

```
Read CSV -> Generate input tensors/scalars -> Call aclnn* C API -> Generate golden (CPU) -> Precision compare -> Output results
```

## Separate Data Preparation and Device Execution

```shell
# Prepare without calling the aclnn* target API or compare.
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/aclnn_cat --plat Ascend950

# Restore tensors/scalars/golden, run the target API, and compare.
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/assets \
  --manual-data-dirs /data/aclnn_cat
```

Prepare does not query device count or compile clear/warmup helper kernels, but it
still requires CANN/OPP for CSV and ACLNN API metadata parsing. Pass the target
`--plat` on a host without SoC detection. Both stages require the same CSV data
contract. See [Manual-Data Prepare and Replay](../Manual_Data_Prepare_and_Replay.md)
for complete constraints.

# Common Examples

```shell
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_add.csv
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_convolution.csv
python3 -m ttk aclnn -i aclnn_cat.csv --dump-on-fail
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/my_golden.py
```
