# Kernel算子测试指南


---

# 环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch（Golden计算使用）
- 完成CANN包安装，确保环境变量已source

```shell
# 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
source /usr/local/Ascend/cann/set_env.sh
# 指定路径安装
# source ${install_path}/cann/set_env.sh
```

2、自定义算子包（可选）

若测试自定义算子，请先完成编译部署。安装 .run 包时支持两种安装路径，包名格式为 `cann-ops-xxx-custom_linux-${arch}.run`（如 `cann-ops-math-custom_linux-x86_64.run`）：

**默认路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run
```

需先完成上述 CANN 环境变量 source。安装到 `${ASCEND_OPP_PATH}/vendors/` 下。

**指定路径安装**

```shell
bash cann-ops-xxx-custom_linux-${arch}.run --install-path=${install_path}
```

安装到 `${install_path}/vendors/${vendor_name}/` 下，需执行 `source ${install_path}/vendors/${vendor_name}/bin/set_env.bash` 使算子包生效，`set_env.bash` 自动设置 `ASCEND_CUSTOM_OPP_PATH` 环境变量。

> 若未安装直接使用解压目录，可手动执行 `export ASCEND_CUSTOM_OPP_PATH="${算子包路径}"`。

3、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

4、检查环境

```shell
python3 -m ttk info
```

确认设备信息正常后再执行测试。

# 测试用例编写

Kernel模式CSV字段说明详见 [Kernel用例编写](./Kernel_Case_Writing.md)。

以测试Add算子kernel为例，`add.csv` 完整文件如下：

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
add_01,,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
add_02,,add,"((969, 7188), (1,))","('float16', 'float16')","('ND',)","((969, 7188),)","('float16',)","('ND',)","((969, 7188), (1,))","('ND',)","((969, 7188),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
```

带编译参数的算子（如MatMulV3的transpose参数），通过 `attributes` 字段传入：

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
matmul_512_1_1792__1792_256,llama3_70b_train,mat_mul_v3,"((512, 1792), (1792, 256), None, None)","('bfloat16', 'bfloat16', 'float32', 'int8')","('ND',)","((512, 256),)","('bfloat16',)","('ND',)","((512, 1792), (1792, 256), None, None)","('ND',)","((512, 256),)","('ND',)","{'transpose_x1': False, 'transpose_x2': False, 'offset_x': 0, '#enable_pad': 1}","((-1, 1),)","((0.001, 0.001),)",1e-08,(),(),True,,,0,,(),()
```

更多用例编写示例可参考项目 `examples/case_store/kernel/` 目录下的CSV文件。

# 精度测试

执行精度测试时，使用 `kernel` 子命令：

```shell
# 基本执行（默认动态编译）
python3 -m ttk kernel -i add.csv

# 动态 shape 编译（默认）
python3 -m ttk kernel -i add.csv -d

# 指定设备
python3 -m ttk kernel -i add.csv --dev 0

# 输出结果
python3 -m ttk kernel -i add.csv -o results.csv
```

## 执行流程

Kernel模式的完整执行流程如下：

```
读取CSV → 编译或匹配内核并tiling → 生成输入数据 → 生成Golden（CPU） → NPU执行 → 精度比对 → 输出结果
```

## 编译模式

| 参数 | 编译模式 | 说明 |
|------|---------|------|
| `-d`（默认） | 动态 shape 编译 | 用动态shape编译，经tiling得到实际shape后执行 |
| `-c` | 静态 shape 编译 | 用固定shape编译后直接执行 |
| `-b release` | 二进制模式 | 使用预编译的发布内核 |

```shell
# 动态 shape 编译（默认）
python3 -m ttk kernel -i add.csv -d

# 关闭动态 shape 编译
python3 -m ttk kernel -i add.csv -d false

# 静态 shape 编译
python3 -m ttk kernel -i add.csv -c

# 二进制模式（使用发布内核）
python3 -m ttk kernel -i add.csv -b release
```

## 仅编译

```shell
# 仅编译不执行
python3 -m ttk kernel -i add.csv --co
```

## 编译控制参数

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--dynamic` | `-d` | 动态 shape 编译；`-d false` 关闭 | **开启** |
| `--const` | `-c` | 静态 shape 编译 | 关闭 |
| `--binary` | `-b` | 二进制模式；`-b release` 使用发布内核 | 关闭 |
| `--compile-only` | `--co` | 仅编译不执行 | 关闭 |
| `--no-prof` | | Kernel dry-run；与精确的 `--dump in,golden` 组合时进入手工数据 prepare | 关闭 |
| `--compile-opts` | | 编译选项（KEY=VALUE格式，可多次指定） | 无 |
| `--tiling-run` | `--tr` | Tiling运行次数 | 3 |
| `--reuse-hbm` | | 每个用例默认下发3次到NPU执行，复用同一块HBM内存以使能L2 Cache | 关闭 |
| `--reserve-hbm` | | 预留HBM内存（MB） | 无 |
| `--clear-atomic` | | 强制在算子执行前清零输出和workspace | 关闭 |
| `--clear-ub` | | 执行前将UB填充为指定值（默认清零） | 关闭 |
| `--clear-l1` | | 执行前将L1填充为指定值（默认清零） | 关闭 |
| `--clear-l0` | | 执行前将L0A/L0B/L0C初始化为指定值（默认清零；L0A/L0B填充指定值，L0C为matmul计算结果，值为0时全零） | 关闭 |
| `--simt-ub` | | SIMT 模式 UB 大小 | 无 |
| `--simt-stack-dcu` | | SIMT 模式 DCU 栈大小 | 无 |
| `--force-block-dim` | | 强制指定 block_dim | 无 |

示例：

```shell
# 编译选项
python3 -m ttk kernel -i cases.csv --compile-opts op_debug_config=oom,dump_cce --compile-opts enable_deterministic_mode=1

# 多次执行间复用HBM（使能L2 Cache）
python3 -m ttk kernel -i cases.csv --reuse-hbm
```

> 精度比对方法详见[精度比对方法](../Precision_Comparison.md)，Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 性能测试

Kernel模式默认会采集Profiling性能数据：

```shell
# 默认执行（含Profiling）
python3 -m ttk kernel -i add.csv

# 禁用Profiling
python3 -m ttk kernel -i add.csv --no-prof

# 设置执行次数（默认板端3次）
python3 -m ttk kernel -i add.csv --run=5
```

## 性能相关参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run` | 执行次数 | 板端3次 |
| `--warmup` | Profiling前预热 | 开启 |
| `--npu-timeout` | NPU执行超时（ms） | 无限制 |

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk kernel -i add.csv

# 使用2张卡
python3 -m ttk kernel -i add.csv --dev=2

# 每张卡2个进程
python3 -m ttk kernel -i add.csv --pc=2

# 指定使用卡0和卡1
python3 -m ttk kernel -i add.csv --device-whitelist=0,1
```

# 调试

```shell
# 调试单个用例
python3 -m ttk kernel -i add.csv -t add_01 --single-log

# 固定随机种子（可复现）
python3 -m ttk kernel -i add.csv --seed 42

# 仅校验CSV用例格式（不下设备）
python3 -m ttk kernel -i add.csv --validate
```

Dump 调试详见[Dump 数据调试](../Dump_Debug.md)。

# 常用场景示例

```shell
# 运行MatMulV3（llama3 shape）
python3 -m ttk kernel -i examples/case_store/kernel/mat_mul_v3.csv

# 运行Split算子（带常量输入）
python3 -m ttk kernel -i examples/case_store/kernel/split.csv -c

# 运行ConcatD算子（TensorList）
python3 -m ttk kernel -i examples/case_store/kernel/concat_d.csv

# 重跑精度失败的用例
python3 -m ttk kernel -i add.csv --rerun=precision_status

# 使用自定义Golden插件
python3 -m ttk kernel -i add.csv --plugin /path/to/my_golden.py

# Excel 多 sheet 用例（默认首个工作表；--sheet 指定工作表）
python3 -m ttk kernel -i examples/case_store/kernel/add.xlsx
python3 -m ttk kernel -i examples/case_store/kernel/add.xlsx --sheet T2
```

> 通用参数（用例筛选/设备并行/精度控制/调试/结果输出）见[任务执行](../Task_Execution.md)
