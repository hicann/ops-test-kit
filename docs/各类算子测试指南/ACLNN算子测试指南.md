# ACLNN算子测试指南

[toc]

---

# 环境准备

1、基本环境配置

- 建议Python 3.8+
- 完成CANN包安装（包含aclnn头文件和动态库）
- 确保环境变量已source

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

## 基本格式

ACLNN模式CSV的核心字段为 `api_name`（而非Kernel模式的 `op_name`）。详细字段说明可参考：[用例生成](../用例生成.md)

以aclnnCat为例：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes
aclnnCat_float,aclnnCat,"(((3,3),(3,2)),(3,5),)","(('float32','float32'),'float32')",{'dim': -1},"(1,)"
```

## 关键字段说明

### tensor_view_shapes（张量形状）

支持嵌套结构表示多输入和TensorList：

```
普通多输入：  "((2,3),(4,5),)"        — 两个输入，shape分别为(2,3)和(4,5)
TensorList：  "(((3,3),(3,2)),(3,5),)" — 第一个输入是TensorList(包含两个tensor)，第二个输入是普通tensor
```

### output_tensor_indexes（输出张量索引）

指定哪些位置的参数是输出张量。索引从0开始：

- `"(1,)"` — 第2个参数是输出
- `"(2,)"` — 第3个参数是输出（如aclnnConvolution的weight输出）

### attributes（API属性）

API的非张量参数，以字典形式传入：

```csv
{'dim': -1}
{'stride':[2,2],'padding':[3,3],'dilation':[1,1],'groups':1}
```

### scalar_dtypes（标量参数）

当API有标量参数时，通过此字段指定标量的数据类型：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,scalar_dtypes
aclnnAdd_00,aclnnAdd,"((1,2,3,4),(1,2,3,4),(1,2,3,4))","('float32',)","('float32',)"
```

## 特殊场景

### 原地操作

通过 `output_inplace_indexes` 指定原地操作的输出：

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,output_tensor_indexes,output_inplace_indexes
aclnnInplaceFill_01,aclnnInplaceFillTensor,"((3,4,5),)","('float32',)","{'value': 1.5}","(0,)","(0,)"
```

### TensorList输出

部分算子（如aclnnSplitTensor）的输出是TensorList，在 `tensor_view_shapes` 和 `output_tensor_indexes` 中对应标记即可。

# 精度测试

```shell
# 基本执行
python3 -m ttk aclnn -i aclnn_cat.csv

# 指定设备
python3 -m ttk aclnn -i aclnn_cat.csv --dev 0

# 使用余弦相似度比对
python3 -m ttk aclnn -i aclnn_cat.csv --compare cosine

# 输出结果
python3 -m ttk aclnn -i aclnn_cat.csv -o results.csv
```

## 执行流程

ACLNN模式的执行流程如下：

```
读取CSV → 生成输入张量/标量 → 调用aclnn* C API → 生成Golden（CPU） → 精度比对 → 输出结果
```

## 输入/Golden与设备执行分离

```shell
# prepare：不调用aclnn*主API、不申请设备锁、不compare
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/assets \
  --no-prof --dump in,golden --dump-format bin \
  --manual-data-dirs /data/aclnn_cat --plat Ascend950

# replay：恢复tensor/scalar/Golden，执行aclnn*主API并compare
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/assets \
  --manual-data-dirs /data/aclnn_cat
```

prepare 不查询设备数量，也不编译清理或 warmup 辅助 Kernel，但 CSV 和 ACLNN API
元信息解析仍要求 CANN/OPP 环境。无卡环境无法探测 SoC 时必须传目标 `--plat`。
两个阶段必须使用相同 CSV 数据契约；plugin 或 Golden 逻辑变化后必须重新 prepare。

完整目录结构、`bin/npy/pt` typed data 文件校验和参数约束参见
[手工数据准备与回放](../手工数据准备与回放.md)。

# 多卡并行

```shell
# 使用全部可用NPU卡
python3 -m ttk aclnn -i aclnn_cat.csv

# 使用2张卡
python3 -m ttk aclnn -i aclnn_cat.csv --dev=2

# 指定使用卡0
python3 -m ttk aclnn -i aclnn_cat.csv --device-whitelist=0
```

# 调试

```shell
# 失败时自动Dump数据
python3 -m ttk aclnn -i aclnn_cat.csv --dump-on-fail

# Dump数据为npy格式
python3 -m ttk aclnn -i aclnn_cat.csv --dump full --dump-format npy

# 固定随机种子
python3 -m ttk aclnn -i aclnn_cat.csv --seed 42
```

# 常用场景示例

```shell
# 运行aclnnCat
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv

# 运行aclnnAdd（带标量参数）
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_add.csv

# 运行aclnnConvolution（复杂属性）
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_convolution.csv

# 运行aclnnInplaceFillTensor（原地操作）
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_inplace_fill_tensor.csv

# 使用自定义Golden插件
python3 -m ttk aclnn -i aclnn_cat.csv --plugin /path/to/my_golden.py
```
