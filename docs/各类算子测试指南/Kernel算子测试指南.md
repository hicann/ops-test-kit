# Kernel算子测试指南

[toc]

---

# 环境准备

1、基本环境配置

- 建议Python 3.8+，安装PyTorch（Golden计算使用）
- 完成CANN包安装，确保环境变量已source
- 自定义算子请先完成编译部署

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2、TTK工具安装

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

# 测试用例编写

## 编写CSV用例文件

根据需要测试的算子输入参数信息，编写对应的CSV用例文件。详细字段说明可参考：[用例生成](../用例生成.md)

以测试Add算子kernel为例，`add.csv` 完整文件如下：

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
add_01,,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
add_02,,add,"((969, 7188), (1,))","('float16', 'float16')","('ND',)","((969, 7188),)","('float16',)","('ND',)","((969, 7188), (1,))","('ND',)","((969, 7188),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
```

## 带编译参数的算子

部分算子需要传递编译参数（如MatMulV3的transpose参数），通过 `attributes` 字段传入：

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

## 精度对比

```shell
# 使用余弦相似度比对
python3 -m ttk kernel -i add.csv --compare cosine

# 失败时自动Dump数据
python3 -m ttk kernel -i add.csv --dump-on-fail

# Dump输入和Golden到npy文件
python3 -m ttk kernel -i add.csv --dump in,golden --dump-format npy
```

精度对比方法的详细说明请参考：[结果分析](../结果分析.md)

# 输入和Golden两阶段执行

使用精确的 prepare 组合先生成 input 和 CPU Golden，但不执行目标 Kernel；随后在目标
设备恢复数据并执行：

```shell
python3 -m ttk kernel -i add.csv --plugin /path/to/kernel_assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/add

python3 -m ttk kernel -i add.csv --plugin /path/to/kernel_assets \
  --manual-data-dirs /data/add
```

两个命令使用相同的 `-d`、`-c` 或 `-b release` 模式。单独 `--no-prof` 仍表示原有
Kernel dry-run；`--co` 在 input/Golden 生成前停止，不能与手工数据 prepare 或 replay
组合。文件格式、目录迁移、Kernel plugin 要求和校验规则参见
[手工数据准备与回放](../手工数据准备与回放.md)。

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

# 常用场景示例

```shell
# 运行MatMulV3（llama3 shape）
python3 -m ttk kernel -i examples/case_store/kernel/mat_mul_v3.csv

# 运行Split算子（带常量输入）
python3 -m ttk kernel -i examples/case_store/kernel/split.csv -c

# 运行ConcatD算子（TensorList）
python3 -m ttk kernel -i examples/case_store/kernel/concat_d.csv

# 调试单个用例（失败时自动dump）
python3 -m ttk kernel -i add.csv -t add_01 --dump-on-fail

# 重跑精度失败的用例
python3 -m ttk kernel -i add.csv --rerun=precision_status

# 使用自定义Golden插件
python3 -m ttk kernel -i add.csv --plugin /path/to/my_golden.py
```
