# XPU 三方交叉校验与性能采集

通过远端 XPU（GPU/MLU/CPU 等非 NPU 加速器）运行同一算子，与 NPU 输出做交叉比对，或仅采集 XPU 侧性能数据。适用于多硬件精度对齐、算子迁移验证、性能基准建立等场景。

## 适用场景

| 场景 | 命令组合 | 说明 |
|------|---------|------|
| 三方交叉校验 | `--compare cross_check` + `--config` + `--plugin` | XPU 输出作为第三方参考，与 NPU 输出/Golden 做误差比值 |
| XPU 性能采集 | `--xpu-perf` + `--config` | 只采集 XPU 侧 `device_ms`，不取数据，不影响精度比对 |

> 两个场景都需要先部署 xpu-server，详见 [xpu-server 部署指南](../ttk/remote/server/README.md)。

## 1. 前置准备

### 1.1 部署 xpu-server

在 XPU 机器上启动服务（本机测试最简）：

```bash
python -m xpu_server.xpu_server --port 9090
```

跨机/CI 等场景参见 [xpu-server 部署指南](../ttk/remote/server/README.md) 的「场景 1-4」。

### 1.2 配置 TTK worker 端

在工作目录创建 `ttk.conf.yaml`，或通过 `--config` 指定：

```yaml
remote:
  endpoints:
    - host: 127.0.0.1      # XPU 机器地址
      port: 9090
  # mTLS（跨机加密时启用，本机可省略）
  # tls_ca: /opt/ttk-certs/ca.crt
  # tls_cert: /opt/ttk-certs/client.crt
  # tls_key: /opt/ttk-certs/client.key

frameworks:
  torch:
    xpu:                   # XPU 硬件 profile
      torch_lib: mlu       # torch binding（如 torch_mlu）
      profiler: builtin
```

### 1.3 配置 third_party（cross_check 必需）

在算子 TestSpec 插件中声明 `third_party`，告诉 TTK 用哪个 provider 和 API 作为参考：

```python
__spec__ = {"add": "AddTestSpec"}


class AddTestSpec:
    third_party = {
        "torch": "torch.add",  # provider=torch, api=torch.add（点分 API 路径）
    }
    # 或用类形式（复杂自定义场景）：dict 值传类对象，__call__ 按名收输入
    # class ThirdPartyImpl:
    #     def __call__(self, x, y, **kwargs):
    #         return [x + y]
    # third_party = {"torch": ThirdPartyImpl}
```

没有 `third_party` 时，`cross_check` 会因无参考输出而 `GOLDEN_FAILURE`。

## 2. 三方交叉校验

### 2.1 命令

```bash
python3 -m ttk kernel -i cases.csv \
  --plugin /path/to/assets.py \
  --config ttk.conf.yaml \
  --compare cross_check
```

### 2.2 工作流程

```
CSV 用例 → NPU 编译执行 → output_bytes
                 ↓
       remote client dispatch → xpu-server 执行同一算子 → third_party outputs
                 ↓
       cross_check：output / golden / third_party 三方误差比值
                 ↓
       结果写入 precision_status + xpu_metrics
```

### 2.3 比对逻辑

`cross_check` 计算 NPU 输出与第三方输出的误差比值，支持 `mare`/`mere`/`rmse` 三种度量，按 level 预设容差判定通过/失败。详细容差规则参见 [精度比对方法](./Precision_Comparison.md)。

### 2.4 provider 过滤

CSV 中多个算子可能配置了不同 provider，可用 `--provider` 缩小范围：

```bash
python3 -m ttk kernel -i cases.csv --compare cross_check --provider torch --config ttk.conf.yaml
```

`--provider` 是测试过滤器，只缩小 dispatch 范围，不覆盖 spec 中的 `third_party` 配置。未设置时使用 spec 的第一个 provider。

### 2.5 complex32 输入输出的传递

`complex32` 无 numpy 存储类型，wire 上以 **fp16 + 尾维 `[2]`（real, imag 交错）** 布局传输。X-Input-Schema 的每个条目在物理 `dtype` 之外携带 `logical_dtype`（CSV 声明值），server 端据此把该布局还原成**逻辑 shape 的 `torch.complex32` 张量**再喂给三方 API：

```python
import torch

__spec__ = {"complex_mul": "ComplexMulTestSpec"}


class ComplexMulTestSpec:
    """complex_mul：complex32 输入的 third_party 参考实现"""

    class TorchRefImpl:
        # 参数按名绑定（契约同 golden 类形式）：x1/x2 输入喂给 __call__。
        # complex32 已由 server 还原为逻辑 shape 的 torch.complex32，
        # 无需处理 "fp16 + 尾维 [2]" 的存储布局。
        def __call__(self, x1, x2, **kwargs):
            return [x1 * x2]  # 输出 complex32，server 自动转交错 fp16 回传

    third_party = {"torch": TorchRefImpl}
    tolerance = {"complex32": {"standard": "cross_check"}}
```

- 同一算子混跑 `float16` / `complex32` 用例时，在 `__call__` 里按 `x1.dtype == torch.complex32`（或 `x1.is_complex()`）分支即可——逻辑 dtype 已随请求传递，不再依赖尾维启发式。
- 字符串形式只接受点分 API 路径（如 `"torch.abs"`）；自定义实现传类或可调用对象。

输出方向同理：三方返回的 complex32 张量由 server 转回 fp16 `[..., 2]` 交错布局回传，与 NPU 输出 / golden 的存储约定对齐后参与比对。第三方实现若返回 complex64/complex128（numpy 原生复数），元素数与交错布局不匹配会导致比对错位——complex32 输出请返回 complex32（或自行 `view(torch.float16)`）。

### 2.6 失败处理

- xpu-server 不可达 → `xpu_results={}`，cross_check 输出 `GOLDEN_FAILURE`
- provider 解析失败 → 同上
- XPU 执行报错 → 该 provider 标记为 `FAIL`，不参与比对

## 3. XPU 性能采集

### 3.1 命令

```bash
python3 -m ttk kernel -i cases.csv \
  --config ttk.conf.yaml \
  --xpu-perf
```

### 3.2 与 cross_check 的区别

| 维度 | `--xpu-perf` | `--compare cross_check` |
|------|-------------|------------------------|
| 数据传输 | 不取 XPU 输出 | 取 XPU 输出做比对 |
| 精度影响 | 不影响 | 替换比对方法 |
| 结果列 | `xpu_metrics`（device_ms） | `precision_status` + `xpu_metrics` |
| 需要 third_party | 否 | 是 |

`--xpu-perf` 可与默认 `mixed` 比对同时使用：NPU 侧正常做 Golden 比对，XPU 侧额外采集性能数据。

### 3.3 结果输出

`xpu_metrics` 列写入结果 CSV，格式示例：

```json
{"torch": {"device_ms": 0.45, "status": "PASS"}}
```

## 4. 参数约束

| 参数 | cross_check | xpu-perf |
|------|------------|----------|
| `--compare` | 必须 `cross_check` | 任意（默认 `mixed`） |
| `--config` | 必须（含 endpoints） | 必须（含 endpoints） |
| `--plugin` | 必须（含 `third_party`） | 可选 |
| `--provider` | 可选过滤 | 可选过滤 |
| `--no-prof` | 不兼容 | 不兼容 |
| `--validate` | 不兼容 | 不兼容 |

## 5. 通路支持

| 通路 | cross_check | xpu-perf |
|------|------------|----------|
| Kernel | 支持 | 支持 |
| ACLNN | 支持 | 支持 |
| GEIR | 支持 | 支持 |
| E2E | 支持 | 支持 |

> E2E 模式下，若 `op_name` 为点分 API 路径（如 `torch.add`）且未配置 `third_party`，server 会直接按 dotted 路径解析执行，无需额外 spec。
