# TestSpec — 算子测试规范

> **不依赖任何测试框架或工具，Ascend 生态统一的算子测试规范。**

## 快速上手

以 `abs` 算子为例，四种测试路径的 spec 注册汇总：

| 测试路径 | 注册名 | 参数名来源 | Golden 参数类型 | 无 spec 时默认行为 |
|---------|--------|----------|----------------|-------------------|
| Kernel | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 无 golden，报 UNSUPPORTED |
| GEIR | `abs`（CSV `op_name`） | 算子 `def.cpp` 形参 | numpy.ndarray | 无 golden，报 UNSUPPORTED |
| ACLNN | `aclnnAbs`（CSV `api_name`） | `aclnn_abs.h` 形参 | torch.Tensor | 自动从 torch 同名 API 映射 |
| E2E | `torch.abs`（CSV `api_name`） | torch API 签名 | torch.Tensor | 框架自动在 CPU 跑同 API 作 golden |

> 注册名即 `get_spec_attr` 的查找 key：Kernel/GEIR 用 `op_name`，ACLNN/E2E 用 `api_name`。GEIR 复用 Kernel 的 spec，无需额外编写。类名按 PascalCase+TestSpec 约定或通过 `__spec__` dict 显式注册。

### Kernel

```python
__spec__ = {"abs": "AbsTestSpec"}

import torch

class AbsTestSpec:
    """Abs 算子测试规范（kernel 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(torch.from_numpy(x)).numpy()]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

### ACLNN

```python
__spec__ = {"aclnnAbs": "AclnnAbsTestSpec"}

import torch

class AclnnAbsTestSpec:
    """Abs 算子测试规范（aclnn 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(x)]

    third_party = {"torch": "torch.abs"}
    tolerance = {"float32": {"standard": "binary_equal"}}
```

### E2E

```python
__spec__ = {"torch.abs": "TorchAbsTestSpec"}

import torch

class TorchAbsTestSpec:
    """Abs 算子测试规范（e2e 流程）"""
    def golden(x, **kwargs):
        return [torch.abs(x)]

    tolerance = {"float32": {"standard": "binary_equal"}}
```

## Spec 字段一览

所有字段可选，按需声明：

| 字段 | 用途 | 合法类型 | 适用路径 |
|------|------|---------|---------|
| `golden` | CPU 真值 | `str` / 函数 / 类 | Kernel / GEIR / ACLNN / E2E |
| `third_party` | 三方标杆 | `str` / `dict` / 类 | Kernel / GEIR / ACLNN / E2E |
| `compare` | 自定义比对 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `pre_compare` | 比对前处理 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `customize_inputs` | 自定义输入 | 函数 | Kernel / GEIR / ACLNN / E2E |
| `pre_npu` | input之后、主设备API之前的算子自定义阶段 | 函数 | ACLNN / E2E |
| `tolerance` | 精度标准 | `dict(dtype→{standard, ...})` | Kernel / GEIR / ACLNN / E2E |
| `torch_graph` | Graph 模式 | `torch.nn.Module` 子类 | E2E（仅 torch API） |
| `describe` | 算子描述信息 | 函数 | 全路径 |

所有属性可选。`golden` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `golden = "numpy.abs"` | numpy 有直接对应 API |
| 函数 | `def golden(x, *, axis=-1):` | 中等复杂度，`*` 前=输入，`*` 后=属性 |
| 类 | `class Golden: def __call__(self, x):` | 需要状态管理，`__init__` + `__call__` + `__del__` |

> golden 在 CPU 计算，Kernel/GEIR 传入 numpy.ndarray，ACLNN/E2E 传入已 H2D 的 torch.Tensor。

`third_party` 三种形式：

| 形式 | 示例 | 适用场景 |
|------|------|---------|
| 字符串 | `third_party = "torch.nn.functional.softmax"` | 单 API 对标 |
| dict | `third_party = {"torch": "...", "tf": "..."}` | 多 vendor / 多框架 |
| 类 | `third_party = MyImplClass` | 小算子拼接，需要 `__call__` |

> third_party 调用三方 API（如 torch），Kernel/GEIR 会将 numpy 转 torch 后调用，ACLNN/E2E 直接传入设备侧 torch.Tensor。

### 类形式参数绑定（Impl class）

> `golden` / `third_party` 用类形式时适用。框架按参数名从算子输入/属性池中按名匹配，分别传给 `__init__` 和 `__call__`。

- **参数名必须匹配算子定义**：`__init__` 与 `__call__` 的参数（除 `self`、`**kwargs`）并集须为算子输入/属性的子集，不匹配则抛 `UnknownParamError`。
- **按方法声明分发**：每个输入/属性只传给声明了该参数名的方法；若 `__init__` 和 `__call__` 同时声明同名参数，则两者都收到。
- **带默认值的参数可不传**：算子定义中未出现的参数，若声明了默认值则取默认值（如 `def __call__(self, x, axis=-1)` 中 `axis` 未传时取 -1）。
- **`device` 为框架保留参数**：有输入 tensor 时设备信息取自 `tensor.device`，无需声明；无输入 tensor 的算子（如 `range`/`eye`）可声明 `device` 参数，由框架注入目标设备。

```python
class AddImpl:
    def __init__(self, x):       # x 可进 __init__
        self.x = x
    def __call__(self, x, y):    # 同名 x 两边都喂；y 只喂 __call__
        return [self.x + x + y]

class EyeImpl:
    def __init__(self, n, device):    # 无输入 tensor: device 框架注入
        self.n, self.device = n, device
    def __call__(self):
        import torch
        return [torch.eye(self.n, device=self.device)]
```

`compare` / `pre_compare` 消费规则：

- ACLNN 必须用 CSV 中的精确 `aclnn*` `api_name` 注册；Kernel 必须用 CSV 中的精确 raw `op_name` 注册。
- Kernel 会对每个已启用的 dynamic、const 和 binary 模式分别调用 hook，关闭模式哨兵不进入 hook。
- 需要读取当前或 replay 恢复输入时，compare 必须显式声明仅关键字参数 `compare_context`；仅声明 `**kwargs` 不会收到该参数。

`customize_inputs` 自定义输入生成：

- 参数名与算子定义一致，返回修改后的输入（结构与原输入一致）。
- Kernel/GEIR 传入 numpy.ndarray，ACLNN/E2E 传入 torch.Tensor。
- 若只需调整随机数据范围，可不写此字段，直接在 CSV 中设置 `input_data_ranges`。

```python
def customize_inputs(x, min_val, max_val, **kwargs):
    if min_val[0] == max_val[0]:
        min_val[0] = numpy.min(x)
        max_val[0] = numpy.max(x)
    return (x, min_val, max_val)
```

`pre_npu` 算子前置阶段（仅 ACLNN/E2E）：

- 未声明时完全保持原执行链路；prepare 阶段不会执行该 hook；普通执行和 replay 在 input
  生成/恢复完成后、主设备 API 之前执行一次；
- `pre_npu` 使用和主 API 一样的参数组装规则：API 形参按 ParamPlan 传入，CSV 中不是 API
  形参的属性仍以普通 `kwargs` 传入；该阶段不预设任何算子业务；
- 返回 `None` 或 `PreNpuResult(stop=False)`时继续主 API；返回
  `PreNpuResult(stop=True, reason=...)`时，本 case 以 PASS 停在该阶段，不再执行主 API、Golden 或 compare；
- 显式接收通用 `context` 的 hook 可以调用 `context.run_profiled(stage_name, operation)`，按本轮
  `--task-prof`、`--warmup` 和 `--run` 采集该自定义操作；ACLNN/E2E 使用各自原生 profiler，结果目录默认保留；
- ACLNN hook 还可通过 `context.run_aclnn(...)` 调用辅助 ACLNN API。框架管理 device、stream、executor、
  同步和资源释放；`output_names` 必须指向 C 签名中的 `aclTensor*` 参数并提供同 shape 的 host placeholder，
  暂不支持把 `aclTensorList*` 作为输出复制回来。E2E hook 应直接调用对应 framework API；

`context` 通用运行时上下文：

- 它不是 `pre_npu` 专用参数。当前 ACLNN/E2E 的函数形式 `input`/`customize_inputs`、`golden`、
  `pre_npu`、`pre_compare` 和 `compare` 都可以声明可按关键字传入的
  `context: TtkContext`。参数名和类型标注必须同时存在才会注入；`TtkContext | None`、
  `Optional[TtkContext]` 也受支持；容器类型和混合业务 Union 不会被当作注入请求。未标注的历史业务 `context`、
  只写 `**kwargs` 的老 hook 都不会被覆盖。
  Kernel/GEIR 当前不创建或注入 `context`，不能假定同一 hook 在这两条通路可使用它；
- `api_name`、`testcase_name`、`case_type`标识当前调用；`input_tensors/input_scalars`是当前最终生成或 replay
  恢复的值。算子 hook 只应按自己的显式合同原地写专用 placeholder，其他业务 tensor 不应借 context 隐式改写。
  `attributes`和`csv_fields`分别是解析后的 CSV attribute 和原始 CSV 行，二者均为深度只读映射；普通 CSV
  attribute 仍通过原有 ParamPlan 进入 hook kwargs，不应从 context 重读后覆盖它；
- `options`是 CLI 经校验、归一化后的公开 `SWITCHES` 深度只读快照，不是原始 argparse 对象，也不保留命令行字符串。
  hook 可以记录这些值或选择自己的私有流程，但不得修改快照或借此覆盖 CSV 业务参数；
- `manual_data_mode`、`manual_data_writes_goldens`、`manual_case_dir`和`manual_data_format`说明本次是普通、prepare
  还是 replay，以及当前唯一可定位或 replay 已命中的 testcase 数据目录和格式。`manual_case_dir` 只是通用目录，
  框架不规定插件在其中使用的文件名、扩展名、schema 或生命周期；这些字段可供 input、Golden 或 compare 独立使用，
  不与 pre-NPU 绑定；
- `state`是当前 testcase 私有的普通可变字典，框架不预留 key、不解释 value，也不要求任何 hook 使用它。input、
  pre-NPU、Golden、pre-compare、compare 可以传递任意算子私有对象，也可以完全不传；同一 spec 内的多个 hook 应使用
  `<operator>.<purpose>` 形式的私有 key，避免插件自己的功能互相覆盖。框架不限定 value 的来源、用途或类型，
  也不会把任何算子业务定义为公共 context 协议；
- `state`只在当前进程、当前 testcase 生命周期内共享。正常返回（包括 input-only prepare）后，框架不再持有该
  `context`，其中对象随引用释放；算子如果把 context/value 另存到模块全局、外部闭包，或自行构造引用环，框架无法
  代替算子回收。大对象在最后一个消费阶段结束后可以主动 `pop`，但不能早于后续 hook 使用；
- 插件需要跨 prepare/replay 保存私有状态时，可自行在 `manual_case_dir` 下创建文件或目录。TTK 只管理并严格校验
  完整匹配 `input_<slot>_*`、`scalar_<slot>_*`、`golden_<slot>_*` 命名规则的框架数据文件；其他条目不复制、不搬运、
  不恢复、不校验，replay 扫描时直接忽略。prepare 启动时仍会先清理整个同名旧 testcase，插件必须根据本轮 input
  重新生成所需条目。框架数据发布不会改动本轮插件已经生成的条目；插件自行维护其名称、schema、版本、完整性、原子
  写入和清理。只有伪装成框架完整命名的目录、软链或其他非普通条目才会被严格拒绝。多个 replay 根只有在命中具体
  case 后才能确定 `manual_case_dir`；直接执行同时给多个根时该字段为 `None`。跨服务器复制对应 testcase 目录即可同时
  迁移框架数据和插件私有条目；
- `run_aclnn(...)` 只在 ACLNN `pre_npu` hook 执行期间有效，用于调用辅助 ACLNN API；
  `run_profiled(...)` 只在 ACLNN/E2E `pre_npu` hook 执行期间有效。两个 runner 都会在 hook 返回或抛异常后清空，
  其他阶段或事后持有的 context 无法再次调用。

`run_profiled(...)` 返回只读语义的 `RuntimeProfile`：`enabled` 表示本轮是否启用性能采集，`repeat_count` 是实际
执行次数，`elapsed_us` 是设备总耗时均值，`result_path` 是保留的 profiler 根目录，`kernels` 包含每个设备 kernel 的
name、总耗时、调用次数、均值、最大值和最小值。`--task-prof false` 时 operation 仍执行一次，但不预热、不重复、
不启动 profiler，返回 `enabled=False`。启用采集时使用固定目录 `msprof/pre_npu/<case>/<stage>/`，重复运行同一
case/stage 沿用 TTK 现有覆盖语义；先前返回的 `result_path` 只描述当次使用的位置，不保证后续运行后内容仍对应旧结果。

常用 CLI 到 context 的映射如下。属性型 switch 使用稳定别名，避免 hook 依赖 `SWITCHES` 的私有存储名：

| CLI | `context` / `options` 中的值 | 说明 |
| --- | --- | --- |
| `--manual-data-dirs` | `manual_data_dirs`、`options["manual_data_dirs"]` | 规范化后的绝对 `Path` 元组；replay 可有多个搜索根 |
| `-o` / `--output` | `options["output_file_name"]` | 主结果 CSV 的已解析路径值；算子可据此定位自己的 companion 结果，但不得改写框架主结果 |
| `--task-prof` | `options["task_prof"]`、`options["TASK_PROFILING"]` | 两者都是布尔值；前者是推荐稳定别名 |
| `--run` | `options["run_time"]` | 已应用默认值后的实际执行次数，不暴露私有 `_run_time` |
| `--compile-only` | `options["compile_only"]` | 已应用 property 规则后的布尔值 |
| `--dump` / `--dump-format` / `--dump-on-fail` | `options["dump_config"]` 的 `mode` / `file_format` / `dump_on_fail` | `mode` 是归一化位掩码，不是原始逗号字符串 |
| `--golden-mode` / `--compare` | `options["golden_mode"]` / `options["compare_method"]` | 当前 Golden 和比较模式 |
| `--warmup` / `--deterministic-level` | `options["warmup"]` / `options["deterministic_level"]` | 性能 warmup 与确定性控制彼此独立 |
| `--plugin` | `options["plugin_path"]` | 规范化后的插件路径元组 |
| `--no-prof` | `manual_data_mode="prepare"`，并由 `manual_data_writes_goldens` 区分 `in` 与 `in,golden` | 没有独立 `options["no_prof"]`；框架暴露的是校验后的阶段语义 |

除表中的稳定别名外，`SWITCHES.__slots__` 中不以 `_` 开头的字段仍按原字段名出现在 `options`，例如
`selected_testcases`、`random_seed`、`input_distribution`、`run_timeout`、`validate_only`。这只是当前运行配置快照，
不是承诺把未来每个 argparse 名称原样变成 context key；需要长期依赖的新 key 应先定义稳定别名和回归测试。

`context`的字段只解决运行时观察、算子私有 state 和辅助调用，不改变参数优先级：CSV 显式值仍优先于 pytest
参数处理、input、Golden、compare 和 assets 调度。使用 context 的 hook 仍应把业务 attribute 声明为普通参数或从
`**kwargs`读取，而非把 context 当成第二套参数来源。

#### 通用自定义阶段示例

下面的 hook 只演示公共能力。自定义动作、state key、停止字段、附属文件和 companion 结果格式都由算子定义；
TTK 不预设任何算子业务。插件附属文件应只保存跨进程确实需要、版本化且可校验的紧凑状态，
不应重复保存 input/Golden 中已有的大 tensor。`context.state` 用于同一 testcase 当前进程内的阶段传递，普通附属文件
用于 prepare/replay 或多服务器交接。

```python
from ttk.test_spec import PreNpuResult, TtkContext


def prepare_runtime_state(*, context: TtkContext, **kwargs):
    profile = context.run_profiled("runtime_state", build_runtime_state)
    context.state["my_operator.runtime_profile"] = profile
    if kwargs.get("runtime_state_only") is True:
        return PreNpuResult(stop=True, reason="runtime state ready")
    return None


class MyOperatorTestSpec:
    pre_npu = prepare_runtime_state
```

算子可以执行任何必须位于 input 和主 API 之间的准备动作，也可以完全不使用 state、附属文件、profiling 或停止能力；
但不能借此修改 CSV 显式业务语义或把主 API 伪装成已执行。

`tolerance` 精度标准：

按 dtype 声明每个输出的精度比对标准。`standard` 字段只接受 2.1 官方标准：

| standard 值 | 含义 | `--compare` 传参 |
|-------------|------|------------------|
| `stat_rel_err` | 统计相对误差（默认） | `stat_rel_err` |
| `binary_equal` | 逐 bit 相等 | `binary`、`bin` |
| `cross_check` | 交叉比对（需配合 `third_party`） | `cross_check` |
| `quant` | 量化比对（待支持） | `requant` |
| `isclose` | numpy.isclose 容差比对 | `close` |
| `cosine` | 余弦相似度 | `cosine` |

> 前四行为 2.1 官方标准，可写入 `Spec.tolerance`；`isclose` 和 `cosine` 为 CLI 框架增强，仅通过 `--compare` 指定。CLI 可用值列中的名字均可直接用于 `--compare`。

`torch_graph` Graph 模式（仅 E2E）：

声明一个 `torch.nn.Module` 子类，用于 E2E 的 Graph 模式构图测试。仅适用于 torch API。

```python
import torch.nn as nn

class SoftmaxGraph(nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

class TorchSoftmaxTestSpec:
    torch_graph = SoftmaxGraph
```

`describe` 算子描述信息：

返回算子的描述性信息（如约束、备注等），供框架展示。函数形式，返回 str。

## 命名与注册

### 类命名约定

`op_name` → 类名 = `PascalCase + TestSpec`：

| op_name | 类名 |
|---------|------|
| `abs` | `AbsTestSpec` |
| `softmax_v2` | `SoftmaxV2TestSpec` |
| `layer_norm` | `LayerNormTestSpec` |

> 仅处理纯 snake_case（如 `softmax_v2` → `SoftmaxV2`）。对于已是 PascalCase 或含数字拼接的算子名（如 `BatchMatMul`、`matmul_3d`），需用 `__spec__` 显式注册。

### `__spec__` 显式注册

```python
# 值为类名字符串（非类对象）；可写在文件顶部
__spec__ = {"abs": "AbsTestSpec"}
```

### 文件与发现规则

- **文件命名**：spec 文件名不以 `_` 开头，否则 loader 跳过该文件。
- **静态索引**：首次加载时，loader 遍历目录下所有 `.py` 文件，通过 `ast.parse` 解析模块级 `__spec__` 赋值，构建 `op → (file, classname)` 索引。此过程仅做语法解析，不执行文件代码，无副作用。
- **惰性执行**：仅当某个 op 被查询命中时，loader 才加载（exec）其所在的 spec 文件并缓存模块对象；未命中的文件在本次会话中不会被执行。
- **`__spec__` 位置不限**：因索引阶段只读 AST 不执行代码，`__spec__` 写在文件顶部或底部效果相同；建议写在顶部，使 op→class 映射一目了然。

## 消费方式

```python
from ttk.test_spec import get_spec_attr, get_spec_class_meta

golden_fn = get_spec_attr("softmax_v2", "golden", ("path/to/specs",))
tp = get_spec_attr("softmax_v2", "third_party", ("path/to/specs",))
meta = get_spec_class_meta("softmax_v2", ("path/to/specs",))   # {spec_file, class_name}
```

## 更多示例

见 `examples/` 目录：

| 文件 | 场景 |
|------|------|
| `01_minimal_golden.py` | 最简 golden（单个函数） |
| `02_golden_three_forms.py` | golden 三种形式：字符串/函数/类 |
| `03_third_party_three_forms.py` | third_party 三种形式：字符串/dict/类 |
| `04_golden_tolerance_compare.py` | golden + tolerance + compare 组合 |
| `05_pre_compare_customize_inputs.py` | pre_compare 两种模式、customize_inputs、多输出 compare |
| `06_golden_multi_path.py` | 同一算子在 Kernel/ACLNN/E2E 不同路径下的 golden 编写 |
