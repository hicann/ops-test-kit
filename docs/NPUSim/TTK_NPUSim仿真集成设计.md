# TTK × NPUSim 仿真测试集成设计文档

> 状态：已落地（2026-08-10 起 record/report 改用 CANN 自带 cannsim，不再依赖 npu-simulator 仓库源码）
> 关联仓库：`ops-test-kit`（TTK，改动方）
> 日期：2026-08-07（2026-08-10 更新）
>
> **使用者请参阅**：[TTK NPUSim 使用指南](./TTK_NPUSim使用指南.md)（命令、参数、排障）

## 1. 背景与目标

TTK（`ops-test-kit`）目前只能在真实 Ascend 芯片上执行算子。NPUSim（`npu-simulator`，SoC 级仿真工具）能在无卡环境下以二进制兼容方式运行同一 kernel，产出 bit 级精度结果与指令流水图。

**目标**：让 TTK 的 **kernel** 与 **aclnn** 测试模式可选地通过 NPUSim 仿真执行代替真实芯片，输出：

1. TTK 现有精度比对结果（复用 golden + comparison 体系 → 结果 CSV 的 `precision_status` 等列）
2. 仿真流水图（复用 NPUSim `report` 的 `trace_core*.json` Chrome tracing + HTML 报告，显式开关触发）

**已确认决策**：

1. 集成形态：`ttk kernel` / `ttk aclnn` 命令内加 `--backend npusim` 开关切换执行后端。
2. wrapper（user_app）形式：**Python wrapper**（NPUSim 原生支持 `.py`，自动注入 `TASK_QUEUE_ENABLE=0`）。
3. 流水图：默认不生成，`--sim-report` 显式触发。

## 2. 关键事实（已实证）

### NPUSim 侧

| 事实 | 说明 |
|---|---|
| user_app 直接给 `.py` 路径即可 | `launch_spec.py` 判非 ELF → `bash -c` 执行，`is_python=True` 注入 `TASK_QUEUE_ENABLE=0`；退出码 0 = PASS（`runner.py:179`） |
| record 持久产物仅 `instr.bin` + `.soc-version` | `log/`、`log_ca/`、`run_log/` 跑完被 `artifacts.py::cleanup_internal_dirs` 清理 |
| **NPUSim 不产出设备端 tensor 数值** | 必须由 user_app（TTK 生成的 wrapper）自行回读算子输出并写盘，TTK 再读取比对 |
| report 输入优先级 | helper DB → `instr.bin` → `log_ca/`；引擎 BiProfRunner（需 plotly≥5） |
| report 产物 | `trace_core{N}.json`（Chrome tracing）、`kernel_<N>_reports/core_<M>/` 下各 `*_report.html`、聚合 `index.html`/`SUMMARY.md`、`*_vf_report.json` |
| 环境依赖 | `ASCEND_TOOLKIT_HOME`（需 source setenv）、camodel 库（toolkit 自带，record 注入 `LD_LIBRARY_PATH`） |
| SoC 映射 | `Ascend950 → Ascend950PR_9589`；`SOC_INFO` max_cores=32 |
| **版本坑（历史）** | 早期（9.1.0-beta.3 之前）CANN 自带 `cannsim` 的 instr.bin 记录格式不兼容（424→432 字节），曾用 `PYTHONPATH=<npu-simulator> python3 -m cannsim.main` 调仓库源码。**2026-08-10 起**：CANN 9.1.0-beta.3 / 9.2.0 自带 cannsim（v0.1.0）与仓库源码行为一致（record/report 全参数、`cannsim_` 产物前缀），record/report 均直接走 `$ASCEND_TOOLKIT_HOME/bin/cannsim`，已彻底移除仓库依赖 |

### TTK 侧

- CLI 分发：`ttk/cli/kernel.py|aclnn.py` → `bridge.py::args_to_switches()` 灌入 `SWITCHES` → `run_with_switches()` → `NpuInstance`（kernel/aclnn 共用）。
- **执行接缝点**：
  - kernel：`npu/op/profiling.py:725` `do_profiling()` → `OnlineRtsProfiling.do()`（`npu/op/rts_sequence.py:49-138`）→ `RTSInterface`（`runtime/rts_interface.py`，ctypes 加载 libruntime.so）→ `rtKernelLaunchWithFlagV2`。返回 `RTSProfilingResult(cycle, output_bytes, oob)`。
  - aclnn：`npu/op_api/profiling.py:505` `do_profiling()` → `AclOpExecutor._acl_sequence()` → `AclInterface`（`aclnn/acl_interface.py`，libascendcl.so + libnnopbase.so）→ `aclnnXxxGetWorkspaceSize` + `aclnnXxx`。返回 `ApiProfilingResult(success, api_prof, op_prof, output_bytes, output_view_shapes)`。
- 比对：`comparison/comparison.py:27-90` `compare(outputs, goldens, dtypes, standards)`；选择逻辑 `comparison/resolve.py:107-134`；结果 `EachCompareResult(precision, is_pass, log, standard, metrics)`。
- **已有 model 语义可复用**：`MODE.ASCEND_CAMODEL`（`classes.py:33`）时 `use_device()=False`、`run_time=1`、`RTSInterface(camodel=True)` 加载 `libruntime_camodel.so`、跳过 clear_l1/ub、`rts_stream=None`。
- Manual Data 两阶段：`--no-prof --dump in,golden` prepare / `--manual-data-dirs` replay（`bridge.py::configure_manual_data`、`ttk/core_modules/manual_data.py`），文件命名 `input_{idx}_{dtype}__shape_{dims}.{ext}`、`golden_{idx}...`，支持 bin/npy/pt。

## 3. 总体架构

新增 `ttk/core_modules/simulator/`（目录已存在，空壳）。**复用优先，不另起炉灶**：

- wrapper 直接 import TTK 现有 `RTSInterface` / `OnlineRtsProfiling` / `AclInterface`（纯 Python+ctypes，可放进子进程脚本）。
- 结果回填复用现有比对与结果 CSV 链路。
- 目录命名贴近 `manual_data.py` 规范。

```
ttk/core_modules/simulator/
├── __init__.py          # 导出 run_kernel_sim / run_aclnn_sim / apply_sim_backend / maybe_generate_sim_report
├── config.py            # SimOptions dataclass + 默认值
├── npusim_runner.py     # 定位并调用 CANN 自带 cannsim：record/report 子进程封装、env 组装
├── case_writer.py       # 把 worker 内存中的输入/golden/spec 落盘为 <sim_output>/<case>/
├── wrapper.py           # 生成 kernel/aclnn 两种 wrapper .py（字符串模板）
├── sim_profiling.py     # run_kernel_sim(context) / run_aclnn_sim(context, dev_id) 主逻辑
└── report.py            # --sim-report 时调用 npusim report
```

## 4. 关键改动

### 4.1 SWITCHES 新字段（`ttk/utilities/classes.py`）

- `__slots__` 新增：`backend`、`sim_soc_version`、`sim_output_dir`、`sim_report`、`sim_cores`、`sim_object_file`。
- 默认值：`backend="npu"`、`sim_soc_version="Ascend950"`、`sim_output_dir=""`（空 → `root_path/sim_output`）、`sim_report=False`。
- （2026-08-10 移除 `sim_npusim_home`——不再定位 npu-simulator 仓库。）
- 补 `__getstate__/__setstate__`（遍历 slots），保证 SWITCHES 可 pickle 进 wrapper。

### 4.2 CLI 参数（新增 `ttk/cli/sim_args.py`，kernel.py / aclnn.py 都 `add_sim_args(parser)`）

```
--backend {npu,npusim}    # 默认 npu
--sim-soc <Ascend950>     # 默认 Ascend950
--sim-output <dir>        # 仿真中间产物根目录（默认 root_path/sim_output）
--sim-report              # 仿真结束后生成流水图报告
--sim-cores <0-2,5>       # 传给 record/report 的 -n（默认空=全核）
--sim-obj <kernel.o>      # 可选，传给 record -f，报告含源码级信息
```

### 4.3 bridge 接线（`ttk/cli/bridge.py`）

- `args_to_switches()`：按 `hasattr(args, ...)` 模式灌 `--sim-*`。
- 新增 `apply_sim_backend(sw, args)`：当 `backend=="npusim"` 时做"仿真语义归一"：
  - `sw.mode = MODE.ASCEND_CAMODEL`（复用 model 语义：`use_device=False`、`run_time=1`、`rts_stream=None`、跳过 clear_l1/ub）
  - `sw.dev_plat = sw.sim_soc_version`（避免 `get_device_platform` 对 Ascend950 调 `get_npu_hw_info` 失败）
  - `sw.warmup = False`、`sw.TASK_PROFILING = False`、`sw.deterministic_level = 0`、默认 `sw.process_per_device = 1`（防多 camodel 打爆 CPU）
- `_handle_kernel` / `_handle_aclnn` 在 `args_to_switches` 后调用 `apply_sim_backend`。
- `configure_manual_data` **不改**：`--backend npusim` 不带 `--no-prof/--manual-data-dirs` 时它早返回，与透明仿真路径不冲突。

### 4.4 NpuInstance 短路（`ttk/core_modules/npu/instance_refactor.py`）

- `get_device_platform()`（:63）：开头分支——`backend=="npusim"` 时直接写 `sw.short_soc_version="Ascend950"` 并 return（跳过 `get_npu_hw_info`，工具包无 Ascend950.ini）。
- `setup_profile_object()`（:86）：`_compile_help_kernels()` 增加 `and sw.backend != "npusim"`（仿真不编译 warmup/clear 辅助核）。

### 4.5 执行接缝分支（核心）

**kernel** — `profile_process()`（`npu/op/profiling.py:381-386`）三连调改为：

```python
if switches.backend == "npusim":
    (context.dyn_prof_result, context.cst_prof_result,
     context.bin_prof_result) = run_kernel_sim(context)
else:
    # 原 do_profiling 三连调（不动）
```

**aclnn** — `profile_process()`（`npu/op_api/profiling.py:695`）改为：

```python
if switches.backend == "npusim":
    context.prof_result = run_aclnn_sim(context, dev_id)
else:
    context.prof_result = do_profiling(context, dev_id)
```

- `run_kernel_sim` 返回三个 `RTSProfilingResult(cycle=1.0, output_bytes=[bytes...], oob="OK")`（cycle 置 1.0 让 PASS 判定通过；真实性能在 report，后续可解析回填）。
- `run_aclnn_sim` 返回 `ApiProfilingResult(success=True, api_prof="SIM", op_prof="SIM", output_bytes=[...], output_view_shapes=[...], oob="OK")`。

### 4.6 Python wrapper

- 生成时机：`run_kernel_sim`/`run_aclnn_sim` 每次执行时写入 `<sim_out>/wrappers/<case>_<mode>.py`。
- 拉起：`npusim record` 的 user_app 直接给 wrapper 的**绝对路径**。
- wrapper 进程内 `PYTHONPATH` 前置 `sw.root_path`（TTK 源码，继承 CANN env；不含 npu-simulator 仓库），直接 `from ttk.core_modules.runtime import RTSInterface` 等，**不复制 ctypes 代码**。
- wrapper 收参：`sys.argv[1]` = 用例目录绝对路径。

**kernel wrapper**：反序列化 `switches.pkl` + 每模式 `param.pkl`（`RTSProfilingParam`，父进程 `__construct_profiling_param` 构造后 pickle；`param.kernel_dir` 绝对路径指向 `kernel_meta`）→ `device = RTSInterface(camodel=True)` → 依序执行启用的 dyn/cst/bin（`OnlineRtsProfiling(device, param).do()`）→ 每个输出写 `output_{i}.bin` + `result.json`（cycle/oob/ok）。

**aclnn wrapper**：`AclInterface` 可直接 import 复用；`Phase1ParamBuilder`/`AclOpExecutor` 强依赖 `TestcaseAclnn`（pickle 风险高）→ wrapper 内用精简驱动镜像其关键步骤（约 60 行）：读 `spec.json`（api_name/param_layout/tensors/scalars/attrs）→ `create_acl_tensor`/`create_acl_scalar` → 按 param_layout 组装 phase1 参数 → `acl_get_workspace` + `acl_execute` → 对 `output_tensor_indexes` 回读写 `output_{i}.bin` + `output_view_shapes.json`。**补单测**：对样例 op 断言 wrapper 参数构建与 `Phase1ParamBuilder` 一致，防逻辑漂移。

## 5. 数据流（`ttk kernel --backend npusim -i cases.csv`）

1. CLI → SWITCHES（backend=npusim, mode=CAMODEL）→ `NpuInstance`（平台短路、device_count=1）→ worker 池。
2. worker `profile_process`：正常生成输入、golden（CPU）、解析 tolerance——全部复用现有逻辑。
3. `run_kernel_sim(context)`：
   - 启用的 dyn/cst/bin 各构造 `RTSProfilingParam` → pickle 落盘 `<sim_out>/<case>/{switches.pkl,dyn/,cst/,bin/}`（未启用模式父进程置 `RTSProfilingResult.fail("SUPPRESSED")`）。
   - 生成 wrapper.py → `npusim_runner.run_record`（一次 record 跑全部启用模式，wrapper 依序执行并写 `output_*.bin`/`result.json`）。
   - 读回 → `RTSProfilingResult`。
4. `handle_profiling_result` + `comparing()` → `ProfilingReturnStructure` → 结果 CSV（`precision_status` 等列照常产出）。
5. `--sim-report` 时 `npusim report` 生成流水图（独立产物，失败仅 WARN，不阻断精度）。

## 6. 目录规范（一个用例）

```
<sim_output>/                      # sw.sim_output_dir
├── wrappers/
│   └── caseA_kernel.py / caseA_aclnn.py
└── caseA/
    ├── switches.pkl               # kernel：pickle SWITCHES
    ├── dyn/ cst/ bin/             # kernel：param.pkl + output_*.bin + result.json
    ├── spec.json                  # aclnn：api/param_layout/tensors/scalars/attrs
    ├── input_tensor_*.bin / scalar_*.bin
    ├── output_*.bin + output_view_shapes.json
    └── record_out/
        └── cannsim_<ts>_caseA/
            ├── instr.bin, .soc-version, cannsim.log   # instr.bin 在归档根目录（CANN cannsim 布局）
            └── (report/ 仅 --sim-report)
```

## 7. npusim 调用封装（`simulator/npusim_runner.py`）

- 调用形态：`python3 $ASCEND_TOOLKIT_HOME/bin/cannsim record <wrapper绝对路径> -s <soc> -o <case_dir>/record_out [-u <case_dir>] [-n <cores>] [-f <obj>]`——用 `sys.executable` 执行 CANN 自带 cannsim 入口脚本（与 camodel 同版本配套），不再依赖 npu-simulator 仓库源码。
- `locate_cannsim_executable`：优先 `$ASCEND_TOOLKIT_HOME/bin/cannsim`，其次 PATH 上的 `cannsim`；找不到抛 `RuntimeError`（提示安装含 cannsim 的 CANN）。
- env：继承 TTK 已 source 的 CANN 环境（`PYTHONPATH` 仅前置 `sw.root_path`，不含仓库）；设 `NPUSIM_NO_DELAY=1`/`CANNSIM_NO_DELAY=1` 跳过 record 阶段间 sleep。
- PASS 判定：**以 wrapper 写的 `result.json` 为准**（record 退出码因 camodel teardown 不可靠）；超时用 `sw.proc_timeout` 兜底。
- `run_report`：`-e <record_out/cannsim_*/> -o <case_dir>/report -n <cores>`；`--sim-report` 时调用。

## 8. 与 Manual Data 的整合

**透明模式**：用户只传 `--backend npusim` 即可跑（内部自动完成输入/golden 生成 → 序列化 → 仿真 → 回填），无需先跑 prepare。

兼容：若用户额外传 `--manual-data-dirs`，则走"外部 prepare + 仿真 replay"。`--backend npusim --no-prof` 校验互斥并报错提示。

## 9. 多用例/批量

- 每用例独立一次 record（kernel 不同无法合并，合并会使 instr.bin 混叠）。
- 沿用 TTK worker 池，但 `apply_sim_backend` 默认 `process_per_device=1`；`--proc-timeout` 透传防卡死。
- 仿真慢（单用例秒级到分钟级）→ 三 kernel 模式合并一次 record 缓解。

## 10. 风险与验证

**最高优先级风险**：TBE 编译产物能否在 Ascend950 仿真直跑（TTK 编译按 `dev_plat` 选 arch，Ascend950 平台表无 ini）。**第一步原型验证**：用手写最小 add 用例跑 `--backend npusim`，确认 `rtKernelLaunchWithFlagV2` 在 camodel 上可执行；若 arch 不匹配需 TTK 编译侧接受 Ascend950（唯一可能触及编译链路的点）。

| # | 风险 | 缓解 |
|---|---|---|
| 1 | TBE 编译产物能否在 Ascend950 仿真直跑 | 最高优先原型验证；必要时编译侧透传 Ascend950 |
| 2 | aclnn 算子 so 依赖 ops 包 | 缺失时返回 `ApiProfilingResult.fail("OP_API_NOT_FOUND")` 并提示装 ops 包 |
| 3 | report 需 plotly + CANN 自带 cannsim | `--sim-report` 走 `$ASCEND_TOOLKIT_HOME/bin/cannsim`（与 camodel 配套）；失败仅告警 |
| 4 | `RTSInterface(camodel=True)` 若 camodel 缺符号 | 回退 `camodel=False`（经 LD_LIBRARY_PATH 加载 camodel 的 libruntime.so shim） |
| 5 | 仿真无单值 cycle | v1 置 1.0（PASS）；后续从 report 解析回填 perf |
| 6 | aclnn wrapper 与 `Phase1ParamBuilder` 逻辑漂移 | 补参数构建对齐单测 |

## 11. 验证方式

1. **单元测试**：SWITCHES pickle 往返、sim 参数灌入、`apply_sim_backend` 归一语义、wrapper 模板生成、aclnn wrapper 参数构建对齐 `Phase1ParamBuilder`。
2. **最小原型（最高优先）**：`python3 -m ttk kernel --backend npusim -i examples/case_store/kernel/add.csv` 跑通 add，确认 camodel 可执行、输出回读、precision_status 产出。
3. **aclnn 原型**：`python3 -m ttk aclnn --backend npusim -i examples/case_store/aclnn/aclnn_add.csv`（需 ops 包）。
4. **流水图**：加 `--sim-report`，验证 `<case_dir>/report/trace_core*.json` 与 `index.html` 生成。
5. **批量/回归**：多用例 CSV + `--pc 1` 跑通，结果 CSV 各列正常。

## 12. 改动文件清单

**新增**：`ttk/core_modules/simulator/{__init__,config,npusim_runner,case_writer,wrapper,sim_profiling,report}.py`、`ttk/cli/sim_args.py`、`tests/core_modules/test_simulator_backend.py`
**修改**：`ttk/utilities/classes.py`（SWITCHES 字段 + pickle）、`ttk/cli/kernel.py`、`ttk/cli/aclnn.py`、`ttk/core_modules/npu/instance_refactor.py`、`ttk/core_modules/npu/op/profiling.py`、`ttk/core_modules/npu/op_api/profiling.py`、`ttk/core_modules/npu/op_api/input_generation.py`（修复 `_switche` 拼写 bug）

## 13. 开发进展与实证（截至 2026-08-07）

### 已实现并验证

- **架构落地**：`simulator/` 模块、`--backend npusim` 参数、SWITCHES 仿真字段、`apply_sim_backend` 归一、`get_device_platform` 平台映射、kernel/aclnn 的 `do_profiling` 接缝分支。
- **平台映射（关键修复）**：`Ascend950 → Ascend950PR_9589`（`SIM_PLATFORM_BY_SOC`）。TTK 的 `get_npu_hw_info` 用平台 ini 名查询，NPUSim `record -s` 用 SoC 名。CANN 平台表含 `Ascend950PR_9589.ini`（`Short_SoC_version=Ascend950`、`NpuArch=3510`）。
- **环境冒烟通过**：`npusim record` 拉起 .py user_app、camodel 初始化（`rtSetDevice` 约 25s）、`RTSInterface(camodel=True)` 加载 `libruntime_camodel.so`、TTK 编译的 `.o` 在 camodel 注册成功（`rtRegisterAllKernel` handle 非空）。
- **单测 14 项全过**：`tests/core_modules/test_simulator_backend.py`（参数归一、pickle、case 目录、wrapper 生成、结果解析）；相关回归 121 项通过。
- **修复 TTK 既有 bug**：`op_api/input_generation.py:121` `self._switche` → `self._switch`（输入生成在仿真路径暴露，真机未触发）。

### 关键实证（坑）

1. **camodel teardown SIGSEGV（进程级）**：用户程序跑完后 `atexit`/reset 触发 camodel 清理崩溃（exit=-11），`try/except` 捕获不了。**wrapper 一律以 `os._exit(0)` 收尾**（绕过 atexit），且 **PASS/FAIL 判定以 wrapper 写的 `result.json` 为准，不依赖 record 退出码**。
2. **`record` 以 user_app 脚本目录为 cwd**（`launch_spec._build_file_launch_spec`），wrapper 与 case 路径必须绝对路径；`-u <case_dir>` 传给 wrapper 的 `sys.argv[1]`。
3. **`record` 持久产物仅 `instr.bin` + `.soc-version`**；`log_ca/` 等被清理。
4. ~~CANN 自带 `cannsim` 命令是旧版~~（**2026-08-10 已推翻**）：9.1.0-beta.3 / 9.2.0 自带 cannsim（v0.1.0）与 camodel 配套、与仓库源码行为一致，record/report 均直接使用它，已删除仓库源码路径。

### 环境限制（当前 CANN 9.2.0，非仿真后端问题）

| 模式 | 现象 | 根因 |
|---|---|---|
| kernel | `DYN: OPTILING_FAILURE` | GE `ElewiseDSLTiling` 的 `DoAutoTiling` 对 Ascend950（dav-c310）失败（TBE 动态 tiling 不支持） |
| kernel | `CST: CST_COMPILE_FAILURE` | 同上，const 手工编译也失败 |
| aclnn | `aclnnAdd` / `aclnnCat` 执行失败（`ACLNN_EXECUTE_FAILED`，camodel 有 420 cycles 但算子返回错误） | Ascend950 下 aclnn 算子执行支持不完整（so 加载正常，执行返回错误） |

**结论**：仿真后端逻辑已正确工作（record 拉起、wrapper 执行、回读、比对链路全通）；端到端精度结果受限于当前环境的 **Ascend950 算子支持不完整**（TBE 编译链与 aclnn 算子执行均失败）。需要完整的 Ascend950 ops 环境（安装对应 ops 包）或针对性的算子编译支持才能产出真实精度比对。

### 已知待办

- aclnn wrapper 的 `TestcaseAclnn` pickle 已可用（端到端已走通 record），但算子执行受环境限制，需在支持 Ascend950 的环境回归。
- `--sim-report`（流水图）依赖 record 产出有效 `instr.bin`；当前算子执行失败时 record 无完整 instr.bin，流水图生成逻辑需在算子可用环境验证。
- wrapper 的 PASS 判定已按 `result.json` 实现；`cycle` 置 `"UNKNOWN"`（model 模式），后续可从 report 的 `SUMMARY.md`/`kernel_*_reports` 解析真实性能回填。

### 环境升级后的验证（2026-08-07，CANN 9.1.0-beta.3 + Ascend-cann-950-ops）

**结论：环境限制已解决。** 安装 `Ascend-cann-toolkit_9.1.0-beta.3` + `Ascend-cann-950-ops_9.1.0-beta.3`（`/home/developer/toolkit/Ascend`，source `cann/set_env.sh`）后：

| 模式 | 之前（9.2.0） | 之后（9.1.0-beta.3 + 950-ops） |
|---|---|---|
| kernel | `OPTILING_FAILURE` / `CST_COMPILE_FAILURE` | 编译成功（`BlockDim: 64`），仿真执行 → 回读 → **`DYN_GOLD: PASS`，`PRECISION_STATUS: PASS`** |
| aclnn | `aclnnAdd` 执行失败 | 算子执行成功 → **`GOLD: PASS`，`PRECISION_STATUS: PASS`** |
| 流水图 | — | `--sim-report` 生成 `trace_core0.json`（475 个 Chrome tracing 事件） |

端到端命令（kernel 与 aclnn 均 `PassRate: 100%`）：
```bash
source <new>/cann/set_env.sh
python3 -m ttk kernel --backend npusim -i cases.csv --sim-report   # kernel + 流水图
python3 -m ttk aclnn  --backend npusim -i cases.csv                # aclnn
```

### 环境升级暴露并修复的实现点

1. **`report` 必须用 CANN 自带 cannsim，而非仓库源码**（当时）：camodel（9.1.0）生成的 `instr.bin` 为 **424 字节**，仓库源码 reader 期望 **432 字节** → 仓库源码 report 报 `No executed instructions detected`。曾用 `_report_command()` 优先 `$ASCEND_TOOLKIT_HOME/bin/cannsim`、仓库源码作 fallback。**2026-08-10 起 record/report 均改用 CANN 自带 cannsim**（`$ASCEND_TOOLKIT_HOME/bin/cannsim`），彻底移除仓库依赖。CANN cannsim 的 `instr.bin` 落在**归档根目录**（`cannsim_*/instr.bin`，非 `record/` 子目录），`_instr_dir_of` 对 `-e` 输入目录直接返回（不存在嵌套 `record/` 时）。
2. **`output_bytes` 必须为 list 而非 tuple**：`npu/op/comparison.py:131` 原地赋值（`outputs[idx] = ...`），真机 `_copy_output_from_hbm` 返回 list；`_load_mode_result`/`_load_aclnn_result` 已对齐。
3. wrapper 模板 `output_%d.bin` 的 `Path / str % int` 优先级 bug（需括号），已修复。

### 2026-08-10 变更：record/report 改用 CANN 自带 cannsim

**背景**：早期因 CANN 自带 `cannsim` instr.bin 记录格式不兼容（424→432 字节），record 走 `PYTHONPATH=<npu-simulator> python3 -m cannsim.main` 调仓库源码。经实证，CANN 9.1.0-beta.3 / 9.2.0 自带 cannsim（v0.1.0）与仓库源码的 record/report 行为一致、且与 camodel 同版本配套，故**彻底移除 npu-simulator 仓库依赖**。

| 项 | 旧（仓库源码） | 新（CANN 自带 cannsim） |
|---|---|---|
| record 命令 | `python3 -m cannsim.main record ...`（PYTHONPATH 前置仓库） | `python3 $ASCEND_TOOLKIT_HOME/bin/cannsim record ...` |
| report 命令 | 优先 CANN cannsim，仓库源码 fallback | 一律 `python3 $ASCEND_TOOLKIT_HOME/bin/cannsim report ...` |
| 定位 | `locate_npusim_home`（`--sim-npusim` / `NPUSIM_HOME` / 兄弟目录） | `locate_cannsim_executable`（`ASCEND_TOOLKIT_HOME` → PATH） |
| PYTHONPATH | 前置 `<npu-simulator>:<ttk-root>` | 仅 `<ttk-root>` + CANN env |
| 产物目录 | `record_out/npusim_<ts>_<label>/record/instr.bin` | `record_out/cannsim_<ts>_<label>/instr.bin`（归档根目录） |
| 日志 | `npusim.log` | `cannsim.log` |
| CLI | `--sim-npusim` / `sim_npusim_home` | 已删除 |

**影响文件**：`ttk/core_modules/simulator/npusim_runner.py`（核心）、`ttk/cli/sim_args.py`、`ttk/utilities/classes.py`、`tests/core_modules/test_simulator_backend.py`（+6 测试）、`docs/NPUSim/TTK_NPUSim使用指南.md`。

**验证（2026-08-10，CANN 9.1.0-beta.3 + Ascend-cann-950-ops）**：
- 单测 25 过；pytest 全量 **1992 passed**；ruff 通过
- kernel 端到端 `add_example_00`：`PRECISION_STATUS: PASS`，record 用 CANN cannsim，产物 `cannsim_*`
- `--sim-report` 出 `trace_core0.json`（~4MB）
- aclnn `aclnnAdd_00`：`PRECISION_STATUS: PASS`

> 注意：CANN 9.2.0 缺 Ascend-950-ops 算子实现（如 `add_apt` 不在其 opp），编译阶段即失败；跑 npusim 需 source 含 950-ops 的环境（9.1.0-beta.3）。
