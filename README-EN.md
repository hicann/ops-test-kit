# TTK (ops-test-kit) Operator Test Framework

TTK (Tensor Tool Kit) is a full-pipeline, automated, batch operator testing framework provided by the [CANN](https://hiascend.com/software/cann) operator library. It helps developers quickly perform batch operator functional verification, performance evaluation, and Golden value comparison, improving operator development quality and efficiency.

* **Rich operator test types**: Supports Kernel (AscendC), ACLNN (aclnn\* C API), and E2E (PyTorch/torch\_npu framework API) testing
* **Multiple hardware backends**: E2E mode runs through a unified Backend abstraction supporting NPU, GPU, and CPU as device under test or reference
* **Batch CSV-driven**: Define test cases via CSV files and run them in batch with a single command
* **Multi-card parallel execution**: Supports multi-NPU parallel testing for improved efficiency
* **Multiple precision comparison methods**: Supports numeric approximation, cosine similarity, binary exact, and requantization comparison
* **Extensible plugin system**: Custom Golden / input generation functions, with separate registries for Kernel and ACLNN/E2E modes

[中文文档](./README.md)

## Basic Usage Guide

- [Test Case Generation (CSV Format)](/docs/Test_Case_Generation.md)
- [Task Execution](/docs/Task_Execution.md)
- [Result Analysis](/docs/Result_Analysis.md)

## Operator Test Guides

- [Kernel Operator Test Guide](/docs/Operator_Test_Guides/Kernel_Test_Guide.md)
- [ACLNN Operator Test Guide](/docs/Operator_Test_Guides/ACLNN_Test_Guide.md)
- [E2E Operator Test Guide](/docs/Operator_Test_Guides/E2E_Test_Guide.md)

## FAQ & Troubleshooting

- [FAQ](/docs/FAQ/FAQ.md)

# Quick Start

## Installation

- Install CANN from the Ascend community: [Official Link](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python 3.8+ recommended

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## Write Test Cases

Taking the Kernel mode Add operator as an example, create a CSV file `add.csv`:

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
add_01,,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
```

> For detailed CSV format, see [Test Case Generation](/docs/Test_Case_Generation.md)

## Run Tests

```shell
# Kernel mode: compile + execute + precision comparison
python3 -m ttk kernel -i add.csv

# ACLNN mode
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv

# E2E mode (NPU/GPU/CPU; backend auto-detected if --backend omitted)
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --backend npu

# Show device info
python3 -m ttk info
```

> For more parameters, run `python3 -m ttk kernel --help`

## Test Results

Test results including per-case precision status and overall pass rate are printed to the terminal.

> See [Result Analysis](/docs/Result_Analysis.md) for details.

## Directory Structure

```
ops-test-kit/
├── README.md              # Project documentation
├── LICENSE                # License file
├── pyproject.toml         # Project config (dependencies, build system)
├── requirements.txt       # Runtime dependencies
├── ttk/                   # Python main module (95% of code)
├── csrc/                  # C/C++ extension source (compiled for Python use)
├── tests/                 # Test code
├── examples/              # Example code
│   └── case_store/        # Example CSV test cases
│       ├── kernel/        # Kernel mode examples
│       ├── aclnn/         # ACLNN mode examples
│       └── e2e/           # E2E mode examples
├── docs/                  # Documentation
└── scripts/               # Build/dev helper scripts
```

## Related Information

- [Contributing Guide](CONTRIBUTING.md)
- [Security Statement](SECURITY.md)
- [License](LICENSE)
- [SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## Contact Us

- **Issue Tracking**: [GitCode Issues](https://gitcode.com/cann/ops-test-kit/issues)
- **Community Discussion**: [GitCode Discussions](https://gitcode.com/cann/ops-test-kit/discussions)
- **Technical Articles**: [GitCode Wiki](https://gitcode.com/cann/ops-test-kit/wiki)
