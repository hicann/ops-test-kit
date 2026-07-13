# E2E Operator Test Guide

[toc]

---

> Hardware vendor names appearing in this document are for illustration only; TTK is configuration-driven and supports any hardware accelerator that conforms to the interface.

# Environment Setup

- Python 3.8+, PyTorch 2.0+
- NPU backend: additionally install [torch_npu](https://gitcode.com/Ascend/pytorch)

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirements.txt
pip install ".[e2e-npu]"
pip install ".[e2e-xpu]"
```

# Write Test Cases

E2E mode uses `api_name` as PyTorch API path. See [Test Case Generation](../Test_Case_Generation.md).

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
add_f32_01,torch.add,"((2,3,4),(2,3,4))","('float32','float32')",{'alpha':1.0}
```

## API Types

| Type | api_name Example |
|------|-----------------|
| Function | `torch.add` |
| Module function | `torch.nn.functional.relu` |
| Tensor method | `torch.Tensor.relu_` |

## With `out` Parameter

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
add_out,torch.add,"((2,3),(2,3),(2,3))","('float32','float32','float32')",{'alpha':1.0},"(2,)"
```

## Using `golden_api`

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,golden_api
npu_conv2d_f16,torch_npu.npu_conv2d,"((1,3,224,224),(64,3,7,7),(64,))","('float16','float16','float16')","{'stride':[2,2],'padding':[3,3]}",torch.nn.functional.conv2d
```

# Backends

E2E mode is driven by a unified Backend abstraction (`ttk.core_modules.framework_api.backends`). All backends share the same case parsing, input generation, and precision-comparison pipeline. The CPU backend is commonly used as the Golden source. The backend is auto-selected per the configured hardware segment (`yaml` `frameworks.torch.<seg>`); `--cpu` forces the CPU backend.

| Backend | Dependencies |
|---------|-------------|
| NPU | torch + torch_npu |
| MLU | torch (MLU) |
| CPU | torch |

# Precision Testing

```shell
# auto-selects available backend per configured hardware segment
python3 -m ttk e2e -i torch_add.csv
# force CPU backend
python3 -m ttk e2e -i torch_add.csv --cpu
```

## Execution Flow

```
Read CSV -> Generate input tensors -> Call API on test backend -> Call golden API on CPU -> Precision compare -> Output results
```

# Common Examples

```shell
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv
python3 -m ttk e2e -i examples/case_store/e2e/torch_npu_conv2d.csv
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --cpu
python3 -m ttk e2e -i torch_add.csv -t add_f32_01
python3 -m ttk e2e -i torch_add.csv --seed 42 -o results.csv
python3 -m ttk e2e -i torch_add.csv --validate                   # CSV-only validation (common to all modes)
```
