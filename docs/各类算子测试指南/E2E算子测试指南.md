# E2E算子测试指南

[toc]

---

# 环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch 2.0+
- NPU后端需额外安装 [torch_npu](https://gitcode.com/Ascend/pytorch)
- 完成CANN包安装

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt

# NPU后端
pip install ".[e2e-npu]"

# GPU后端
pip install ".[e2e-gpu]"
```

# 测试用例编写

## 基本格式

E2E模式CSV的核心字段为 `api_name`，值为PyTorch API的完整路径。详细字段说明可参考：[用例生成](../用例生成.md)

以torch.add为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes
add_f32_01,torch.add,"((2,3,4),(2,3,4))","('float32','float32')",{'alpha':1.0}
```

## API类型

E2E模式支持多种PyTorch API调用方式：

| API类型 | api_name示例 | 说明 |
|---------|-------------|------|
| 函数调用 | `torch.add` | 直接调用函数 |
| 模块函数 | `torch.nn.functional.relu` | 调用模块中的函数 |
| Tensor方法 | `torch.Tensor.relu_` | 通过Tensor实例调用 |

## 带out参数的API

部分API支持 `out` 参数将结果写入指定张量。通过 `output_tensor_indexes` 指定输出位置：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
add_out,torch.add,"((2,3),(2,3),(2,3))","('float32','float32','float32')",{'alpha':1.0},"(2,)"
```

## 使用golden_api

当被测API（如torch_npu扩展API）和Golden计算API（如标准PyTorch API）不同时，通过 `golden_api` 指定Golden计算函数：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,tensor_formats,attributes,golden_api
npu_conv2d_f16,torch_npu.npu_conv2d,"((1,3,224,224),(64,3,7,7),(64,))","('float16','float16','float16')","(29,4,2)","{'stride':[2,2],'padding':[3,3],'dilation':[1,1],'groups':1}",torch.nn.functional.conv2d
```

执行时TTK会：
1. 在待测后端（如NPU）调用 `api_name` 指定的API
2. 在CPU上调用 `golden_api` 指定的API生成Golden
3. 比对两者的输出

# 精度测试

## 后端选择

E2E 模式由统一的 Backend 抽象层（`ttk.core_modules.framework_api.backends`）驱动 NPU/GPU/CPU 三种后端，三者共享同一套用例解析、输入生成与精度比对逻辑。CPU 后端通常用作 Golden 计算源。

| 后端 | 参数值 | 依赖 |
|------|--------|------|
| NPU | `npu` | torch + torch_npu |
| GPU | `gpu` | torch (CUDA) |
| CPU | `cpu` | torch |

未传 `--backend` 时按 NPU > GPU > CPU 优先级自动检测可用后端。

## 执行命令

```shell
# NPU后端
python3 -m ttk e2e -i torch_add.csv --backend npu

# GPU后端
python3 -m ttk e2e -i torch_add.csv --backend gpu

# CPU后端（常用于Golden生成）
python3 -m ttk e2e -i torch_add.csv --backend cpu
```

## 执行流程

E2E模式的执行流程如下：

```
读取CSV → 生成输入张量 → 在待测后端调用API → 在CPU调用Golden API → 精度比对 → 输出结果
```

## 精度对比

```shell
# 使用余弦相似度比对
python3 -m ttk e2e -i torch_add.csv --backend npu --compare cosine

# 失败时自动Dump数据
python3 -m ttk e2e -i torch_add.csv --backend npu --dump-on-fail

# Dump数据为npy格式
python3 -m ttk e2e -i torch_add.csv --backend npu --dump full --dump-format npy
```

精度对比方法的详细说明请参考：[结果分析](../结果分析.md)

# 常用场景示例

```shell
# torch.add 基础测试
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --backend npu

# torch_npu.npu_conv2d（使用golden_api）
python3 -m ttk e2e -i examples/case_store/e2e/torch_npu_conv2d.csv --backend npu

# 指定用例运行
python3 -m ttk e2e -i torch_add.csv --backend npu -t add_f32_01

# 固定随机种子
python3 -m ttk e2e -i torch_add.csv --backend npu --seed 42

# 输出结果
python3 -m ttk e2e -i torch_add.csv --backend npu -o results.csv

# 仅校验CSV用例格式（不下设备，所有模式通用）
python3 -m ttk e2e -i torch_add.csv --validate
```
