# 离线数据准备与导入

两阶段执行把输入生成、可选的 CPU Golden 生成与目标设备执行拆开，适用于
准备机和执行机不是同一台服务器的场景，也支持只准备 input。

```text
prepare input：   CSV -> input 生成/修正 -> 只保存 input
prepare 完整：CSV -> input 生成/修正 -> CPU Golden -> 保存两者
replay：          CSV -> 恢复 input -> NPU API -> 恢复或生成 Golden -> compare
```

支持 `ttk e2e`、`ttk aclnn` 和 `ttk kernel`。两个阶段都以同一份 CSV 为用例来源，
按 `testcase_name` 匹配数据；框架不会根据数据目录反向生成用例。

## 1. 阶段选择

| 模式 | 参数 | Input | Golden | 设备执行 | Compare |
| --- | --- | --- | --- | --- | --- |
| 普通执行 | 不传两阶段参数 | 生成 | 生成 | 执行 | 执行 |
| prepare input | `--no-prof --dump in` | 生成并保存 | 不执行 | 跳过 | 跳过 |
| prepare in,golden | `--no-prof --dump in,golden` | 生成并保存 | 生成并保存 | 跳过 | 跳过 |
| replay | `--manual-data-dirs DIR...` | 从文件恢复 | 存在时恢复；不存在时在设备执行后生成 | 执行 | 执行 |

E2E 和 ACLNN 支持 input-only 和完整 prepare。input-only 会替换整个同名 case
目录，不会把旧 Golden 与新 input 混用。不支持 Golden-only prepare。
`--dump-format` 支持 `bin`、`npy`、`pt`，默认值为 `bin`。

### Kernel 的 `--no-prof` 与 `--co`

Kernel 单独使用 `--no-prof` 时保持原有 dry-run 行为：完成 input、Golden 和 workspace
生成，但关闭 dynamic、const、binary 执行。只有精确组合
`--no-prof --dump in,golden` 才进入本流程的 prepare。
Kernel 不支持 input-only prepare，因为其 output 分配可能在设备执行前依赖
Golden 或动态 output 信息。

`--co/--compile-only` 在编译或 tiling 后、input/Golden 生成前返回，不能产生两阶段数据，
因此不能与 prepare 或 replay 组合。Kernel prepare 仍执行当前模式所需的编译或 binary
匹配和 tiling，只在设备锁和目标 Kernel 执行前返回。

## 2. 快速开始

以下命令中的 prepare 和 replay 必须使用兼容的 CSV 与 assets。

### 2.1 E2E

```bash
# prepare
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# replay
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

如果只准备 input，不希望执行 CPU Golden 回调：

```bash
python3 -m ttk e2e \
  -i /path/to/cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/input_result.csv
```

E2E prepare 可以增加 `--cpu` 强制 CPU backend；replay 是设备阶段，不能使用 `--cpu`。

### 2.2 ACLNN

```bash
# prepare
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  --plat Ascend950 \
  -o /path/to/prepare_result.csv

# replay
python3 -m ttk aclnn \
  -i /path/to/aclnn_cases.csv \
  --plugin /path/to/assets \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

ACLNN 同样支持 `--no-prof --dump in`。该数据被 replay 时，先执行目标 API，
再基于恢复的 input 生成缺失的 Golden。

ACLNN prepare 不调用主 API，也不查询设备数量和编译 warmup 辅助 Kernel。CSV 与 API
元信息解析仍依赖 CANN/OPP；准备机无法探测 SoC 时应显式传 `--plat`。

### 2.3 Kernel

两个阶段必须选择相同的 dynamic、const 或 release binary 模式。以下示例使用 release：

```bash
# prepare：编译/binary匹配和tiling后保存数据，不执行目标Kernel
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/manual \
  -o /path/to/prepare_result.csv

# replay：恢复数据后执行release Kernel和compare
python3 -m ttk kernel \
  -i /path/to/kernel_cases.csv \
  --plugin /path/to/kernel_assets \
  --plat Ascend910_9362 \
  -d=false -c=false -b=release \
  --manual-data-dirs /data/manual \
  -o /path/to/replay_result.csv
```

框架没有内置 Kernel input/Golden 时，assets 必须为 CSV 的 `op_name` 注册
`__input__["kernel"]` 和 `__golden__["kernel"]`。只有 E2E TestSpec 的 assets 不能替代
raw Kernel 回调。

## 3. 数据目录

### 3.1 prepare 输出目录

显式传入 `--manual-data-dirs DIR` 时，prepare 写入该目录。省略时按以下顺序确定：

1. 只有一个目录 plugin：`<plugin>/manual_data`；
2. 只有一个 `.py` plugin：`<plugin父目录>/manual_data`；
3. 没有 plugin：`<当前工作目录>/manual_data`。

prepare 只允许一个输出目录。多个 plugin 路径时必须显式指定输出目录。

### 3.2 replay 搜索目录

```bash
python3 -m ttk e2e -i cases.csv \
  --manual-data-dirs /data/current /data/archive
```

TTK 按参数顺序为每个 case 查找目录。第一个同名目录一旦命中便立即使用；该目录损坏时
直接失败，不会跳到后续目录，也不会回退随机生成。

### 3.3 case 目录名

只含 `[A-Za-z0-9_.-]` 且不超过 120 字符的 `testcase_name` 直接作为目录名。其他名称
会安全化，并追加完整原名 SHA-256 的前 12 位。prepare/replay 使用完全相同的原始名称
时会得到相同目录；这不是模糊匹配，也不能从目录名反推原始长名称。

同名 case 重新 prepare 时，旧目录先失效；这也防止 input-only prepare 遗留旧
Golden。新数据在隐藏临时目录中写入并回读校验，
全部成功后才发布；失败不会留下可 replay 的半成品。

## 4. 文件协议

完整 case 可以包含以下文件；input-only case 不含任何 `golden_*` 文件：

```text
<manual-data-dir>/<case-directory>/
├── input_0_bfloat16.bin
├── input_1_int32.bin
├── input_2_none.bin
├── scalar_0_float32.bin
└── golden_0_bfloat16__shape_2x8x128.bin
```

input、scalar、`None` marker 以及 npy/pt Golden 使用普通文件名：

```text
<input|scalar|golden>_<从0开始的扁平索引>_<numpy-dtype|none>.<bin|npy|pt>
```

非 `None` 的 bin Golden 额外编码 prepare 时的输出 shape：

```text
golden_<索引>_<dtype>__shape_<shape-token>.bin
```

`shape-token` 使用 `x` 分隔维度，例如 `(2, 0, 3)` 写成 `2x0x3`，标量 `()` 写成
`scalar`。replay 会在加载数值前比较该 shape 与设备 output 或 CSV output shape，
因此可以发现“错误 shape 但元素总数相同”的回归。非 `None` 的 bin Golden 缺少 shape
后缀时直接拒绝；必须重新执行 prepare 生成该 case 目录。

### 4.1 槽位和 `None`

- input、scalar、Golden 的索引必须分别从 0 连续；
- optional `None` 使用同格式的零字节 `*_none.<format>`，不能省略；
- Golden 的 `*_none` marker 表示抑制该 output slot 的比较；设备 API 仍可物化该输出，
  replay 会把 Golden 恢复为 `None`，由 custom/default compare 标记为 suppressed；
- `*_none.npy` 和 `*_none.pt` 是 TTK marker，不是 NumPy/PyTorch 原生文件；
- 零元素 Tensor 仍用真实 dtype 文件名，例如 `input_0_float32.bin`，与 `None` 不同；
- input 保存最终 backing storage，view shape/stride/offset 与 TensorList/ScalarList 分组由
  当前 CSV 重建。

### 4.2 格式

| 格式 | 保存方式 | shape 来源 |
| --- | --- | --- |
| `bin` | 连续原始字节 | input/scalar 来自 CSV；Golden 来自文件名并与设备/CSV shape 核对 |
| `npy` | NumPy 数组 | 文件内 shape；custom dtype 使用等宽 `voidN`，逻辑 dtype 由文件名恢复 |
| `pt` | CPU Torch 数据 | 文件内 shape；Torch 不支持的 dtype 在同一 pt 内保存 raw bytes 和 shape |

prepare 每次只写一种格式。若交付目录同时存在多种完整副本，replay 按
`bin > npy > pt` 选择一整套数据，不会跨格式拼接。高优先级格式只要出现但不完整或损坏，
当前 case 直接失败，不回退低优先级副本。

目录内不允许 JSON、日志、checksum、子目录、符号链接或其他 sidecar。每个非 None 文件
写完后立即按 dtype、shape 和完整字节回读；校验失败时不发布 case。

## 5. replay 行为

replay 命中数据后始终跳过：

- 随机输入生成和 attributes 数据覆盖；
- input plugin；

完整数据目录中存在 Golden 时，replay 跳过 CPU Golden API 或 Golden plugin。
input-only 数据目录中不存在任何 Golden 时，replay 会先执行设备 API，再基于
恢复的 input 调用当前 Golden 链路，最后 compare。部分 Golden 文件缺失不会被
当成 input-only，而是直接报数据不完整。

replay 仍执行 CSV/ParamPlan、API 或 Kernel 解析、必要的编译/tiling、wrapper、设备执行、
compare 和结果落表。依赖 Spec、wrapper、tolerance、pre-compare 或 custom compare 时，
仍应传入对应 `--plugin` 和 `PYTHONPATH`。

prepare 在 Golden 回调前复制最终 tensor backing storage和 ACLNN scalar。即使自定义
Golden 原地修改输入，保存的仍是目标 API 在普通执行中应看到的数据。

### Compare

两阶段执行不替换 compare：

- E2E、ACLNN和Kernel的pre-compare、自定义compare保持各自正常优先级；
- 没有自定义hook时，ACLNN/Kernel的tolerance和CSV精度字段保持原解析路径；
- 使用 CSV `rtol/ptol/atol` 进行 close 比较时仍需显式传 `--compare close`；
- replay 可以调整精度标准，无需重新 prepare。

custom compare 不应依赖 prepare 进程中 input/Golden plugin 写入的模块全局变量。跨进程或
跨服务器 replay 不会恢复该状态；必要信息应由 Golden、CSV 或可重建配置提供。
若判定需要输入侧数据，compare应显式声明仅关键字参数`compare_context`。direct和
replay都可从其`input_tensors`、ACLNN `input_scalars`（Kernel为空tuple）、CSV `attributes`和原始
`csv_fields`读取当前用例状态；仅声明`**kwargs`的hook不会收到该上下文。

## 6. provider 扩展

`register_manual_data_directory_provider(provider)` 可为个别 case 提供数据路径。provider
接收 `(testcase, case_type, switches)`，返回一个路径、多个有序路径或 `None`。replay
搜索顺序为：

```text
provider返回路径 -> --manual-data-dirs批量路径
```

这允许扩展 CSV 字段覆盖个别 case，其余 case 继续使用 CLI 批量目录。当前没有内置专用
CSV 表头；扩展代码可从 `testcase.original_dict` 读取字段。prepare 始终写 CLI/默认目录，
不使用 provider。provider 也不能绕过 replay 的设备阶段和参数约束。

ACLNN 的 `manual_tensor_binaries/manual_golden_binaries` 是另一组历史字段，不会自动接入
本 provider 或两阶段协议。

## 7. 参数约束

- `-i/--input` 和 `-o/--output` 每次各接收一个 CSV；
- `--plugin` 可在一个字符串中用逗号分隔多个搜索路径；
- `--manual-data-dirs` 在 prepare 最多一个，在 replay 可有多个；
- prepare 禁止 output/full dump、`print`、`--dump-on-fail` 和 `--validate`。完整 prepare
  还禁止 `--golden-mode Disable`，input-only prepare 允许使用。E2E 还禁止 graph
  选项；Kernel 只接受 `--dump in,golden`，并禁止 `--compile-only`；
- replay 禁止 `--no-prof`、`--validate`、`--golden-mode Disable`，E2E 禁止 `--cpu`，
  Kernel 禁止 `--compile-only`；
- `--seed` 只影响 prepare 的随机输入。replay 读取文件，不用 seed 重新生成或匹配数据；
- 两个阶段可用相同 `-t/--testcase` 只处理指定 case。

## 8. 交付验收

至少执行以下检查：

| 测试项 | 验收点 |
| --- | --- |
| direct 基线 | 同一 CSV、plugin、compare 配置可普通执行 |
| 三种格式 | `bin/npy/pt` 分别 prepare 和 replay |
| 两种 prepare | E2E/ACLNN input-only 和完整 prepare 都不执行设备 |
| 缺少 Golden | replay input-only 数据时先执行设备，再生成 Golden 并 compare |
| 目录迁移 | 完整复制数据根到另一绝对路径后 replay |
| shape | 修改 device output 为同 numel 错 shape 时，新 bin/npy/pt 都明确失败 |
| compare | custom compare 和 CSV close 容差在 replay 中生效 |
| 格式优先级 | 整套选择 `bin > npy > pt`，损坏高优先级不回退 |
| 失败保护 | case、槽位、None marker、dtype、shape 或字节错误明确失败 |

prepare 成功时结果为 `MANUAL_DATA_PREPARED/PASS`。完成目录迁移后，应核对 direct 与
replay 的 case 集合、设备执行状态、精度状态和自定义 compare 结果。

## 9. 已知边界

本协议不使用 manifest 或 sidecar，因此不能自动校验：

- 同名 case 是否仍对应同一 API、attributes、wrapper 或 plugin 版本；
- input bin 在 shape 改变但 dtype 和总元素数不变时是否来自旧 CSV；
- 文件同大小替换或位翻转，因为没有 checksum；
- E2E、ACLNN、Kernel 的同名兼容目录是否被混用；
- seed、生成环境和协议版本历史。

修改 API、attributes、input shape/view、wrapper 或 assets 后应重新 prepare。不同入口、
命令或数据版本应使用不同 manual-data 根目录，不要手工重命名 case 内文件。

常见错误：

| 提示 | 检查项 |
| --- | --- |
| `prepared testcase ... was not found` | CSV 的完整 `testcase_name` 和搜索根目录 |
| `slot count ... != CSV` | 文件是否漏传、索引是否连续、CSV 是否变化 |
| `filename dtype ... != CSV storage dtype` | prepare/replay 的 tensor/scalar dtype |
| `saved shape ... != device output shape` | 设备输出 shape 是否回归，是否使用旧 CSV/assets |
| `byte size ... != expected` | 文件是否截断，dtype/shape 是否变化 |
| `unexpected file in manual-data case` | 删除日志、JSON、子目录或其他 sidecar |
