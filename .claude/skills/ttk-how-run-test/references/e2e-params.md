# E2E 模式专用参数

## 专属选项

| 参数 | 说明 | 默认 |
|------|------|------|
| `--cpu` | 强制 CPU 执行（跳过 NPU 自动探测） | 关闭 |
| `--no-prof` | 仅准备输入/golden，不执行主 API | 关闭 |
| `--fullgraph` | Graph 模式（torch.compile）；0=禁用，1=启用 | 0 |
| `--aclgraph` | aclgraph 模式（torch.compile reduce-overhead） | 关闭 |
| `-d`/`--dynamic` | 动态图 | 默认关闭 |
| `-c`/`--const` | 静态图 | 默认关闭 |

> **仿真**：E2E 支持 `--backend npusim` 无卡仿真，但仅 **eager** 执行（graph/aclgraph/fullgraph 禁用），且仅支持 aclnn 算子——非 aclnn 的 legacy 自定义算子（如 `torch_npu.npu_conv2d`）会被 torch_npu 拒绝。详见 `npusim-params.md`。

## 后端依赖

| 后端 | 依赖 |
|------|------|
| NPU | `torch` + `torch_npu`（默认自动探测） |
| CPU | `torch`（`--cpu` 强制） |

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
