# E2E 模式专用参数

## 专属选项

| 参数 | 说明 | 可选值 | 默认 |
|------|------|--------|------|
| `--backend` | 硬件后端 | `npu`/`gpu`/`cpu` | 自动检测（NPU > GPU > CPU） |

## 后端依赖

| 后端 | 依赖 |
|------|------|
| NPU | `torch` + `torch_npu` |
| GPU | `torch`（CUDA） |
| CPU | `torch` |

## 执行流程

```
读取 CSV → 生成输入张量 → 在待测后端调用 API → 在 CPU 调用 Golden API → 精度比对 → 输出结果
```

## API 类型

| api_name 示例 | 说明 |
|-------------|------|
| `torch.add` | 函数调用 |
| `torch.nn.functional.relu` | 模块函数 |
| `torch.Tensor.relu_` | Tensor 方法 |

## golden_api

当被测 API 与 Golden 计算 API 不同时，通过 CSV 的 `golden_api` 字段指定：
- NPU 调用 `api_name`（如 `torch_npu.npu_conv2d`）
- CPU 调用 `golden_api`（如 `torch.nn.functional.conv2d`）
