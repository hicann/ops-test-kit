---
name: ttk-how-run-test
description: 运行 TTK 测试（首次/正常发起一次执行）。只要用户提到运行测试、跑算子、执行 kernel/aclnn/e2e 测试，或需要构造 ttk 命令、查看设备、ttk info、ttk list，就必须使用此 skill。即使用户只是说"帮我跑一下 add"、"怎么运行算子"、"加个编译选项"、"用 2 卡跑"等口语化表述，也应触发。跑完后失败/结果不对要排障用 ttk-how-diagnose。
---

# 运行 TTK 测试

## Step 1: 确认环境

TTK 启动时自动检测 CANN 环境（`_env.py`），按优先级查找 `ASCEND_CUSTOM_PATH` → `ASCEND_TOOLKIT_HOME` → `ASCEND_HOME_PATH` → `ASCEND_OPP_PATH` → 默认路径，找到后自动 source `setenv.bash` 并配置 LD_LIBRARY_PATH、PYTHONPATH、ulimit 等。通常无需手动设置环境变量。

仅在 CANN 装在非标准路径时，需设置 `ASCEND_CUSTOM_PATH`：
```shell
export ASCEND_CUSTOM_PATH=/path/to/cann
```

验证环境：
```shell
python3 -m ttk -v        # 版本检查
python3 -m ttk info       # 设备信息（NPU卡数、CANN版本）
python3 -m ttk list -i cases.csv  # 预览用例列表
python3 -m ttk list -i cases.csv --op add  # 按算子名筛选
```

## Step 2: 选择模式

四层架构对应关系见 AGENTS.md。根据 CSV 表头自动判断模式：

| 模式 | 判断条件 | 子命令 |
|------|---------|--------|
| Kernel | CSV 无 `api_name` 字段 | `python3 -m ttk kernel` |
| GEIR | CSV 无 `api_name` 字段 | `python3 -m ttk geir` |
| ACLNN | CSV 有 `api_name`，值以 `aclnn` 开头 | `python3 -m ttk aclnn` |
| E2E | CSV 有 `api_name`，值非 `aclnn` 开头 | `python3 -m ttk e2e` |

## Step 3: 构造命令

每次只能执行一条 `ttk` 命令，不支持在同一个目录下并发多条（如 `python3 -m ttk kernel -i a.csv &` 和 `python3 -m ttk kernel -i b.csv &`），因为进程间会冲突。多个同模式 CSV 可用逗号拼接：`-i a.csv,b.csv`，或逐个依次拉起。

`-i` 指定 CSV 文件是唯一的必填参数。

### Kernel 模式

```shell
python3 -m ttk kernel -i cases.csv
```

Kernel 支持动态 shape 编译（默认）、静态 shape 编译（`-c`）、二进制执行（`-b release`）三种方式，详见 `references/kernel-params.md`。

### GEIR 模式

```shell
python3 -m ttk geir -i cases.csv
```

GEIR 模式通过 GE 图编译+执行测试算子，复用 Kernel 模式的 CSV 和 golden 资产。支持在线编译（默认）和二进制复用（`-b release`），详见 `references/geir-params.md`。

### ACLNN 模式

```shell
python3 -m ttk aclnn -i cases.csv
```

ACLNN 子命令复用全部通用参数，无模式专属选项，详见 `references/aclnn-params.md`。

### E2E 模式

```shell
python3 -m ttk e2e -i cases.csv
```

E2E 默认自动探测 NPU；`--cpu` 强制 CPU 执行。详见 `references/e2e-params.md`。

## 通用选项速查

### 用例筛选

| 参数 | 缩写 | 示例 |
|------|------|------|
| `--testcase` | `-t` | `-t add_01,add_02` |
| `--testcase-index` | `--ti` | `--ti 1-10` |
| `--testcase-count` | `--tc` | `--tc 10` |
| `--operator` | `--op` | `--op add,mat_mul_v3` |
| `--priority` | | `--priority 1-3` |
| `--rerun` | | `--rerun precision_status` |

### 设备与进程

| 参数 | 缩写 | 说明 | 默认 |
|------|------|------|------|
| `--device` | `--dev` | 使用设备数 | 全部 |
| `--device-whitelist` | | 指定设备 | 无 |
| `--device-blacklist` | | 排除设备 | 无 |
| `--process-count` | `--pc` | 每卡进程数 | 自动（CPU核数×80%÷卡数，上限4） |
| `--platform` | `--plat` | SoC 版本 | 自动 |
| `--proc-timeout` | | 单用例超时(秒) | 0 |

### 精度控制

| 参数 | 说明 | 默认 |
|------|------|------|
| `--compare` | `close`/`stat_rel_err`/`cosine`/`binary`/`requant`/`cross_check` | Spec.tolerance 路由（需 `--plugin`），否则 `stat_rel_err` |
| `--seed` | 随机种子。取相同值时，同一个用例每次执行生成的输入数据完全一致 | 随机 |
| `--golden-mode` | `Enable`/`Disable`/`Promote` | `Enable` |
| `--input-dist` | `uniform`/`normal` | `uniform` |

> 完整容差优先级（`--compare` > `Spec.tolerance` > CSV > 默认）见 `ttk-how-diagnose/references/precision-debug.md`。

### 调试

| 参数 | 说明 | 默认 |
|------|------|------|
| `--dump-on-fail` | 精度失败时 Dump 全部数据 | 关闭 |
| `--dump in,out,golden` | 指定 Dump 内容 | 关闭 |
| `--dump-format npy` | Dump 格式：`bin`/`npy`/`pt`/`print` | `bin` |
| `--single-log` | 每个用例独立日志文件 | 关闭 |
| `--plugin path.py` | 加载自定义插件 | 无 |
| `--validate` | 仅校验 CSV 格式和字段（不执行编译、输入生成、Golden生成等流程） | 关闭 |

### 输出

| 参数 | 缩写 | 说明 |
|------|------|------|
| `--output` | `-o` | 输出结果 CSV |

## 快速示例

```shell
# 运行自带示例
python3 -m ttk kernel -i examples/case_store/kernel/add.csv
python3 -m ttk geir -i examples/case_store/kernel/add.csv
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv

# 调试单个失败用例
python3 -m ttk kernel -i cases.csv -t case_name --dump-on-fail --single-log

# 输出结果到 CSV
python3 -m ttk kernel -i cases.csv -o results.csv
```

## 详细参数

各模式完整参数列表见 references/：
- Kernel 专用参数：`references/kernel-params.md`
- GEIR 专用参数：`references/geir-params.md`
- ACLNN 专用参数：`references/aclnn-params.md`
- E2E 专用参数：`references/e2e-params.md`

完整文档：`docs/Task_Execution.md`（路径相对于 ops-test-kit 仓库根目录）

## 相关 Skill

- 用例怎么写？→ `ttk-how-write-case`
- 测试跑失败了？→ `ttk-how-diagnose`
- 算子没有内置 Golden？→ `ttk-how-write-plugin`
