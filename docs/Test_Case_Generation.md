# Test Case Generation (CSV Format)

[toc]

---

# Introduction

TTK uses CSV files to define test cases in batch. The first row is the header (column names), and each subsequent row represents one test case. Different test modes use different column sets.

**Mode auto-detection logic** (based on CSV header):
- Header contains `api_name` and the first non-empty `api_name` value does NOT start with `aclnn` -> **E2E mode** (`TestcaseE2e`)
- Header contains `api_name` and the first non-empty value starts with `aclnn` -> **ACLNN mode** (`TestcaseAclnn`)
- Header does NOT contain `api_name` -> **Kernel mode** (`TestcaseOp`)

---

# Common Fields (All Modes)

These 9 fields are shared by Kernel, ACLNN, and E2E modes.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `testcase_name` | STRING | **Yes** | auto-generated | Unique test case name. Auto-generated as `auto_testcase_name_N` if missing. |
| `network_name` | STRING | No | `None` | Network/model name tag (e.g. `llama3_70b_train`). |
| `input_data_ranges` | FLOAT_RANGE_NESTED | No | `((None, None),)` | Per-tensor data range for random input generation. Each element is `[min, max]` (closed interval, min and max are inclusive boundary values). None defaults to min=-2, max=2. 3rd and beyond are guaranteed values mixed into random data. min==max generates a fixed value. Nested for TensorList support. E.g. `"((-1, 1), (0, 10))"` or `"((-1, 1, 0), (0, 10))"` |
| `precision_tolerances` | FLOAT_RANGE_NESTED | No | `None` | Per-output precision tolerance pairs `(rtol, atol)`. E.g. `"((0.001, 0.001),)"` |
| `absolute_precision` | FLOAT_OR_NESTED | No | `1e-8` | Default absolute precision tolerance. Can be a single float or a nested container for per-output control. |
| `is_enabled` | BOOL | No | `True` | Set `False` to skip this test case. |
| `remark` | STRING | No | `None` | Free-form comment. |
| `soc_series` | STRING_TUPLE | No | `None` | SoC filter. Prefix with `-` to exclude. E.g. `('Ascend910A', '-Ascend310P')` |
| `priority` | INT | No | `0` | Priority level for selective execution. |

---

# Kernel Mode CSV (26 fields)

For `python3 -m ttk kernel`. Uses `TestcaseOp`.

Concrete shapes are specified directly via `input_shapes` / `output_shapes`, and dynamic shapes are derived automatically by replacing positive dimensions with `-1`. The **count and order** of inputs/outputs are determined by the operator definition files — they must match exactly. TensorList grouping is expressed via nesting in shape fields, e.g. `(((3,3),(3,2)),(3,5))` for TensorList(2) + single tensor. **When to use TensorList format**: when the op info has ParamType = DYNAMIC — even a single tensor must use TensorList nesting.

**Op Definition & Info Sources**:
- **Op definition**: `xxx_def.cpp` (e.g. `add_def.cpp`), defines INPUT/OUTPUT/ATTR declarations
- **Op info**: `aic-{soc_series}-ops-info-{repo[4:]}.json`, contains each input/output's name, supported dtypes, supported formats, paramType (optional/dynamic), ValueDepend, etc.
  - Builtin op info path: `{ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/config/{soc_series}/{repo}/aic-{soc_series}-ops-info-{repo[4:]}.json`
  - Custom op info path: `{ASCEND_OPP_PATH}/vendors/customize/op_impl/ai_core/tbe/config/{soc_series}/aic-{soc_series}-ops-info.json`
  - E.g. add op builtin info on `ascend910b`: `$ASCEND_OPP_PATH/built-in/op_impl/ai_core/tbe/config/ascend910b/ops_math/aic-ascend910b-ops-info-math.json`

## Identity

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `op_name` | STRING | **Yes** | *(none)* | Operator name, prefer the `opInterface.value` from op info (underscore format). If opInterface is not configured, convert camelCase to underscore. E.g. `add`, `mat_mul_v3`. |

## Input/Output Shapes

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `input_shapes` | SHAPE_NESTED | **Yes** | `()` | Input tensor shapes. Supports TensorList nesting. E.g. `"((128, 1024), (1, 1024))"` for two tensors, or `"(((3,3),(3,2)),(3,5))"` for TensorList(2) + single tensor. For optional inputs (paramType=optional), use `None` as placeholder; corresponding dtype/format must also be `None`. |
| `input_dtypes` | DTYPE_NESTED | **Yes** | `()` | Input data types. Nested for TensorList. Supports broadcast: `('float32',)` fills all inputs. E.g. `"('float32', 'float32')"`. |
| `output_shapes` | SHAPE_INFER_NESTED | **Yes** | `None` | Output tensor shapes. Supports TensorList nesting. E.g. `"((128, 1024),)"`. |
| `output_dtypes` | DTYPE_NESTED | **Yes** | *(none)* | Output data types. Nested for TensorList. Supports broadcast: `('float32',)` fills all positions. E.g. `"('float32',)"`. |

## Format & Original Shapes

| Field | Type | Required | Default | Fallback | Description |
|-------|------|----------|---------|----------|-------------|
| `input_formats` | DTYPE_NESTED | No | `('ND',)` | *(none)* | Input tensor formats. E.g. `('ND', 'NCHW')`. Supports broadcast: `('ND',)` fills all positions. |
| `input_ori_shapes` | SHAPE_NESTED | No | -> `input_shapes` | `input_shapes` | Original input shapes (before format transformation). |
| `input_ori_formats` | DTYPE_NESTED | No | `('ND',)` | `input_formats` | Original input formats (for format transformation). Supports broadcast. |
| `output_formats` | DTYPE_NESTED | No | `('ND',)` | `output_ori_formats` | Output tensor formats. Supports broadcast. |
| `output_ori_shapes` | SHAPE_INFER_NESTED | No | `None` | `output_shapes` | Original output shapes. |
| `output_ori_formats` | DTYPE_NESTED | No | `('ND',)` | `output_formats` | Original output formats. Supports broadcast. |

## Attributes

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `attributes` | DICT | No | `{}` | Operator attributes (compile-time and runtime). E.g. `{'transpose_x1': True, 'transpose_x2': False}`. Attributes with names matching op_info input names are extracted as special tensor overrides. When an input has ValueDepend of OPTIONAL or REQUIRED in the op info, its tensor value **must** be specified in attributes (e.g. ReduceMin's `axes`). For non-ValueDepend inputs whose values cannot be randomly generated (e.g. index parameters, shape size parameters, size parameters whose values correlate with other input shapes), specifying values in attributes is a convenient alternative to writing a custom input function for small tensors (e.g. grouped_matmul's `group_list`). |

## Special Properties

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `output_inplace_indexes` | INT_TUPLE | No | `()` | In-place output indexes (input tensor indexes to overwrite). No need to set manually: auto-detected from op info (when output name matches input name, output overwrites that input memory). |
| `output_shape_unknown_indexes` | INT_TUPLE | No | `()` | Indices of **output tensors** with unknown shapes at compile time. E.g. `"(0,)"` means the 0th output shape is undetermined. |

## Options

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dump_file_prefix` | STRING | No | `None` | Custom filename prefix for dumped data files. |
| `manual_input_binaries` | EVAL | No | `()` | Manual input binary file paths. Evaluated as Python expression. Nested for TensorList. |
| `manual_golden_binaries` | EVAL | No | `()` | Manual golden output binary file paths. Nested for TensorList. |

### Example Cases

The `examples/case_store/kernel/` directory provides examples for various scenarios:

| File | Scenarios Covered |
|------|-------------------|
| `abs.csv` | Basic case, multiple dtypes |
| `add.csv` | Multiple dtypes, broadcast, input_data_ranges |
| `concat_d.csv` | TensorList input (DYNAMIC) |
| `mat_mul_v3.csv` | Optional inputs (None placeholder), attributes |
| `non_zero.csv` | Unknown output shape (output_shape_unknown_indexes) |
| `reduce_min.csv` | ValueDepend tensor input (axes via attributes) |
| `split.csv` | TensorList output, dynamic parameters |
| `zeros_like.csv` | Basic case |

---

# ACLNN Mode CSV (24 fields)

For `python3 -m ttk aclnn`. Uses `TestcaseAclnn`.

Includes all [Common Fields](#common-fields-all-modes) plus the following:

## Identity

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_name` | STRING | **Yes** | *(none)* | ACLNN API name. E.g. `aclnnAdd`, `aclnnCat`. |

## Tensor Properties

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tensor_view_shapes` | SHAPE_NESTED | **Yes** | `()` | Tensor view shapes with TensorList nesting. E.g. `"(((3,3),(3,2)),(3,5))"` for TensorList input. |
| `tensor_dtypes` | DTYPE_NESTED | **Yes** | `()` | Tensor data types with nesting. E.g. `"('float32','float32')"` or `"(('float32','float32'),'float32')"`. |
| `tensor_formats` | DTYPE_NESTED | No | `('ND',)` | Tensor formats with nesting. E.g. `(('ND','ND'),'ND')`. |
| `tensor_storage_shapes` | SHAPE_NESTED | No | `()` | Tensor storage shapes (for non-contiguous tensors). Falls back to view shapes. |
| `tensor_view_offsets` | INT_NESTED | No | `()` | Tensor view offsets into storage. |
| `tensor_view_strides` | SHAPE_NESTED | No | `()` | Tensor view strides. Auto-computed from shape if not specified. |

## Output Properties

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `output_tensor_indexes` | INT_TUPLE | No | *(auto)* | Indices indicating which tensors are outputs. Auto-filled from param naming conventions if not set: tensor params ending with `Ref`, `Out`, `OutOptional`, `Output`, `OutputOptional`, or named `output` are detected as outputs. Backward/Grad APIs exclude `gradOutput`, `gradOut`, `grad_output`, `attentionOut`, `dOut`, and non-last `output`. Falls back to last tensor if no match. |
| `output_inplace_indexes` | INT_TUPLE | No | `()` | In-place output indexes (input tensor indexes to overwrite). No need to set manually: auto-detected from op info (when output name matches input name, output overwrites that input memory). |

## Attribute & Scalar Properties

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `attributes` | DICT | No | `{}` | API attribute parameters. E.g. `{'dim': -1}`. Also supports specifying small-shape tensor values by parameter name (for tensors whose values cannot be randomly generated, e.g. index params, shape/size params whose values correlate with other input shapes). E.g. `{'group_list': [1, 1, 1]}`. |
| `scalar_dtypes` | DTYPE_NESTED | No | `()` | Scalar parameter data types. Nested for ScalarList. **No compression/broadcast** (scalars have no shape, count is inferred from dtypes element count only). E.g. `('int64',)` or `(('int32','int32'),)`. |
| `scalar_data_ranges` | FLOAT_RANGE_NESTED | No | `((None, None),)` | Per-scalar data range `(min, max)`. None defaults to min=-2, max=2. Scalar values are randomly generated by default; to specify concrete values, set them in `attributes` (optional). |

## Dump & Manual Data

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `dump_file_prefix` | STRING | No | `None` | Custom filename prefix for dumped data files. |
| `manual_tensor_binaries` | EVAL | No | `()` | Manual tensor binary file paths. Evaluated as Python expression. |
| `manual_golden_binaries` | STRING_TUPLE | No | `()` | Manual golden output binary file paths. |

### Non-Contiguous Tensors

Control non-contiguous tensor layout via `tensor_view_strides`, `tensor_view_offsets`, and `tensor_storage_shapes`. The view shape and actual storage shape can differ:

```csv
tensor_view_shapes,"((2,3),)"
tensor_view_strides,"((4,1),)"
tensor_view_offsets,"(5,)"
tensor_storage_shapes,"((3,6),)"
```

This means: view is 2x3, stride (4,1) means row stride=4, offset=5 starts from the 5th element in storage, and storage shape is 3x6. When strides/offsets/storage_shapes are not specified, tensors default to contiguous layout.

### Specifying Tensor Values via Attributes

Specify small-shape tensor values in the `attributes` dict by the parameter name defined in the API header. This applies to tensors whose values cannot be randomly generated by the framework (e.g. index params, shape/size params whose values correlate with other input shapes). For small-shape tensors, this is more convenient than `manual_tensor_binaries` or custom input plugins:

```csv
attributes,"{'group_list': [1, 1, 1]}"
```

You can also specify tensor values purely for testing purposes.

---

# E2E Mode CSV (19 fields)

For `python3 -m ttk e2e`. Uses `TestcaseE2e`.

**Parameter Source**: E2E test case parameters (tensor count/order/dtype, keyword arguments) come from the framework API's function signature. Check PyTorch official docs or torch_npu extension docs for parameter definitions.

**API Types**:

| api_name Example | Type | Description |
|------------------|------|-------------|
| `torch.add` | Module function | Direct call |
| `torch.nn.functional.relu` | Submodule function | Call function in submodule |
| `torch.Tensor.relu_` | Tensor method | Called on tensor instance (in-place) |

Includes all [Common Fields](#common-fields-all-modes) plus the following:

## Identity

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `api_name` | STRING | **Yes** | *(none)* | Framework API path. E.g. `torch.add`, `torch.nn.functional.relu`, `torch.Tensor.relu_`. |

## Tensor Properties

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `tensor_view_shapes` | SHAPE_NESTED | **Yes** | *(none)* | Input tensor shapes with TensorList nesting. E.g. `"((2,3,4),(2,3,4))"`. |
| `tensor_dtypes` | DTYPE_NESTED | **Yes** | *(none)* | Input tensor data types. E.g. `"('float32','float32')"`. Supports broadcast: `"('float32',)"` fills all positions. |
| `tensor_formats` | DTYPE_NESTED | No | `()` | Tensor formats. Supports broadcast. |
| `tensor_storage_shapes` | SHAPE_NESTED | No | `()` | Tensor storage shapes for non-contiguous tensors. Falls back to view shapes. |
| `tensor_view_offsets` | INT_NESTED | No | `()` | Tensor view offsets into storage. |
| `tensor_view_strides` | STRIDE | No | `()` | Tensor view strides. Auto-computed from shape if not specified. |

## Output & Attributes

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `output_tensor_indexes` | INT_TUPLE | No | *(auto)* | Indices of output tensors. In-place Tensor methods (`torch.Tensor.xxx_`) auto-fill `(0,)`. APIs with `out` parameter require manual specification of output tensor index. |
| `attributes` | DICT | No | `{}` | Framework API keyword arguments. E.g. `{'alpha': 1.0}`. Required non-tensor parameters in the API signature **must** be provided. |
| `golden_api` | STRING | No | `""` | Alternative API for golden computation. E.g. `torch.nn.functional.conv2d`. Set to `"disable"` to skip golden generation. |

---

# Data Type Strings

| Data Type | String(s) |
|-----------|-----------|
| 16-bit float | `float16`, `fp16` |
| 32-bit float | `float32`, `fp32` |
| 64-bit float | `float64`, `fp64` |
| BFloat16 | `bfloat16`, `bf16` |
| 8-bit signed int | `int8` |
| 16-bit signed int | `int16` |
| 32-bit signed int | `int32` |
| 64-bit signed int | `int64` |
| 8-bit unsigned int | `uint8` |
| 16-bit unsigned int | `uint16` |
| 32-bit unsigned int | `uint32` |
| 64-bit unsigned int | `uint64` |
| Boolean | `bool` |
| 64-bit complex | `complex64` |
| 128-bit complex | `complex128` |

# CSV Value Format Reference

| Type | Format | Example |
|------|--------|---------|
| Shape tuple | Double-quoted nested tuple | `"((128, 1024), (1, 1024))"` |
| Dtype tuple | Double-quoted string tuple | `"('float32', 'float32')"` |
| Nested shape (TensorList) | Double-quoted triple-nested tuple | `"(((3,3),(3,2)),(3,5))"` |
| Dict | Raw Python dict | `{'dim': -1}` |
| Numeric | Raw value | `1e-8`, `0`, `True` |
| Empty | Empty field | Two consecutive commas `,,` |
| None input | `None` keyword | `None` |

> **Note**: Fields containing special characters (parentheses, commas, quotes) must be wrapped in double quotes.

# Fallback Chains

Some fields automatically fall back to other headers when not explicitly set:

| Field | Fallback |
|-------|----------|
| `input_ori_shapes` | `input_shapes` |
| `output_ori_shapes` | `output_shapes` |
| `input_ori_formats` | `input_formats` |
| `output_ori_formats` | `output_formats` |
| `input_formats` | *(no fallback, default `('ND',)`)* |
| `output_formats` | `output_ori_formats` |

# Complete Examples

See `examples/case_store/` for ready-to-use CSV examples:

```
examples/case_store/
├── kernel/                    # Kernel mode examples
│   ├── abs.csv                # Abs operator
│   ├── add.csv                # Add operator (multi-dtype)
│   ├── concat_d.csv           # ConcatD (TensorList via nesting)
│   ├── mat_mul_v3.csv         # MatMulV3 (llama3 shapes)
│   ├── non_zero.csv           # NonZero (unknown output shape)
│   ├── split.csv              # Split (const inputs)
│   └── zeros_like.csv         # ZerosLike
├── aclnn/                     # ACLNN mode examples
│   ├── aclnn_add.csv          # With scalar params
│   ├── aclnn_cat.csv          # TensorList concat
│   ├── aclnn_convolution.csv  # Complex attributes
│   ├── aclnn_inplace_fill_tensor.csv  # Inplace operation
│   ├── aclnn_masked_select.csv
│   ├── aclnn_nonzero_v2.csv
│   └── aclnn_split_tensor.csv  # TensorList output
└── e2e/                       # E2E mode examples
    ├── torch_add.csv          # torch.add
    └── torch_npu_conv2d.csv   # torch_npu.npu_conv2d (with golden_api)
```
