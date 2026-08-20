# 任务执行

# 命令行总览

```
python3 -m ttk {kernel,aclnn,e2e,geir,info,list} [选项]
```

| 子命令 | 用途 |
|--------|------|
| `kernel` | AscendC 算子内核编译 + NPU 执行 + 精度比对 |
| `geir` | GE 图编译 + 执行 + 精度比对 |
| `aclnn` | aclnn\* C API 调用 + 精度比对 |
| `e2e` | 框架 API 端到端测试 |
| `info` | 查询本机 Ascend NPU 设备信息 |
| `list` | 预览 CSV 中的测试用例列表 |

查看版本：

```shell
python3 -m ttk -v
```

查看帮助：

```shell
python3 -m ttk kernel --help
python3 -m ttk aclnn --help
python3 -m ttk e2e --help
python3 -m ttk geir --help
```

# 设备信息与用例预览

```shell
# 查看设备信息
python3 -m ttk info

# 预览用例列表
python3 -m ttk list -i examples/case_store/kernel/add.csv

# 按算子名过滤预览
python3 -m ttk list -i cases.csv --op add
```

# 通用参数

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--input` | `-i` | 无 | CSV 用例文件路径（必填） | `-i add.csv` |
| `--config` | | 无 | ttk 配置 YAML 路径（覆盖 `~/.config/ttk/` 和 `./ttk.conf.yaml`） | `--config ttk.conf.yaml` |

## 用例筛选

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--testcase` | `-t` | 无 | 按用例名筛选（逗号分隔） | `-t add_01,add_02` |
| `--testcase-index` | `--ti` | 无 | 按用例索引筛选 | `--ti=1,3,5` 或 `--ti=1-10` |
| `--testcase-count` | `--tc` | 无 | 随机选取N个用例 | `--tc=10` |
| `--operator` | `--op` | 无 | 按算子名筛选 | `--op add,mat_mul_v3` |
| `--exclude-operator` | `--no-op` | 无 | 排除算子名 | `--no-op concat_d` |
| `--priority` | | 无 | 按优先级筛选 | `--priority=1-3` |
| `--provider` | | 无 | 三方provider过滤（如torch/tf），缩小dispatch范围 | `--provider torch` |
| `--rerun` | | 无 | 重跑失败项 | `--rerun=precision_status` |

## 设备与并行

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--device` | `--dev` | 全部 | 使用设备数量 | `--dev 2` |
| `--device-whitelist` | | 无 | 设备白名单 | `--device-whitelist 0,1` |
| `--device-blacklist` | | 无 | 设备黑名单 | `--device-blacklist 2` |
| `--process-count` | `--pc` | 1 | 每张卡进程数 | `--pc 4` |
| `--proc-no-reuse` | | 关闭 | 每个用例创建新进程 | `--proc-no-reuse` |
| `--no-memory-check` | | 关闭 | 跳过主机内存检查 | `--no-memory-check` |
| `--platform` | `--plat` | 自动检测 | SoC版本 | `--plat Ascend910B3` |
| `--proc-timeout` | | 0（不限） | 单用例超时（秒） | `--proc-timeout 120` |
| `--limit` | `-l` | 30 | 单用例HBM内存上限（GB），超过则跳过 | `-l 16` |

## 执行控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--run` | | 3（仿真模式 1） | 执行次数 | `--run=5` |
| `--task-prof` | | 开启 | 任务级 Profiling 开关 | `--task-prof=false` |
| `--warmup` | | 开启 | Profiling 前预热 | `--warmup=false` |
| `--npu-timeout` | | 无限制 | NPU 执行超时（ms） | `--npu-timeout 60000` |
| `--no-prof` | | 关闭 | 只生成输入和 Golden，不执行目标 API | `--no-prof` |
| `--validate` | | 关闭 | 仅校验 CSV 用例格式，不执行编译、输入生成、Golden 生成等流程 | `--validate` |

## 精度控制

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--compare` | | `stat_rel_err` | 精度对比方法。未指定时按 Spec.tolerance 路由（需 `--plugin`）。可选：`close`/`stat_rel_err`/`cosine`/`binary`/`requant`/`cross_check` | `--compare cosine` |
| `--input-dist` | | `uniform` | 输入数据分布，可选：`uniform`/`normal` | `--input-dist normal` |
| `--seed` | | 随机 | 随机种子（可复现） | `--seed 42` |
| `--golden-mode` | | `Enable` | Golden生成模式，可选：`Enable`/`Disable`/`Promote` | `--golden-mode Disable` |
| `--deterministic-level` | `--dl` | `0` | 确定性等级：`0`=关闭，`1`=确定性计算（MD5一致），`2`=强一致，`3`=批一致性（跨用例切片比对） | `--dl=1` |

> 精度比对方法的公式、容差及选择建议详见[精度比对方法](./Precision_Comparison.md)。确定性计算详见[确定性计算与批一致性](./Deterministic_Compute.md)。

## 调试与诊断

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--dump` | | 关闭 | Dump数据：`full`/`in`/`out`/`golden` | `--dump full` |
| `--dump-format` | | `bin` | Dump格式：`bin`/`npy`/`pt`/`print` | `--dump-format npy` |
| `--dump-on-fail` | | 关闭 | 精度失败时Dump全部数据 | `--dump-on-fail` |
| `--manual-data-dirs` | | 无 | prepare输出目录或replay有序搜索目录 | `--manual-data-dirs /data/op` |
| `--single-log` | | 关闭 | 每个用例独立日志文件 | `--single-log` |
| `--plugin` | | 无 | 外部插件路径 | `--plugin /path/to/plugin.py` |
| `--xpu-perf` | | 关闭 | 采集三方XPU性能（需remote XPU配置） | `--xpu-perf` |

## 结果输出

| 参数 | 缩写 | 默认值 | 说明 | 示例 |
|------|------|--------|------|------|
| `--output` | `-o` | 无 | 输出结果CSV路径（覆盖已有文件），与 `-a` 互斥 | `-o results.csv` |
| `--append` | `-a` | 无 | 追加结果到已有CSV（表头不匹配则覆盖），与 `-o` 互斥 | `-a results.csv` |
| `--title` | | 无 | 自定义输出列 | `--title testcase_name,precision_status` |
| `--csv-preserve` | | 关闭 | 保留原始CSV表头 | `--csv-preserve` |
| `--print` | | 开启 | 打印摘要信息 | `--print=false` |
| `--po` / `--progress-output` | | 无 | 进度输出路径 | `--po /tmp/progress.json` |

# 通路专属参数

各通路专属参数详见对应指南：

- **Kernel编译控制**（`-d`/`-c`/`-b`/`--co`/`--compile-opts`等）：[Kernel测试指南](./Operator_Test_Guides/Kernel_Test_Guide.md)
- **E2E后端选择**（`--cpu`/Backend抽象层）：[E2E测试指南](./Operator_Test_Guides/E2E_Test_Guide.md)
- **GEIR编译模式**（`-c`/`-d`/`-b`）：[GEIR测试指南](./Operator_Test_Guides/GEIR_Test_Guide.md)
- ACLNN复用通用参数，无专属选项

# 高阶使用场景

| 文档 | 核心参数 |
|------|---------|
| [离线数据准备与导入](./Offline_Data_Prepare_and_Import.md) | `--no-prof --dump in,golden` + `--manual-data-dirs` |
| [XPU 三方交叉校验](./XPU_Cross_Check.md) | `--compare cross_check` / `--xpu-perf` + `--config` |
| [确定性计算与批一致性](./Deterministic_Compute.md) | `--deterministic-level` / `--dl` |
| [Dump 数据调试](./Dump_Debug.md) | `--dump` + `--dump-format` + `--dump-on-fail` |
| [性能测试](./Performance_Testing.md) | `--warmup` + `--run` |
| [NPUSim 仿真测试](./NPUSim/npusim_usage.md) | `--backend npusim` + `--sim-*` |
