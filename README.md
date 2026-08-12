# 🚀 TTK —— 算子测试工具

TTK（ops Test Tool Kit）是[CANN](https://hiascend.com/software/cann)算子库提供的全链路、自动化、批量化算子测试框架，帮助开发者快速完成算子批量功能验证、性能评估以及Golden值比对，提升算子开发质量和效率。

> 文档中出现的硬件厂商名仅作示例，TTK 通过配置驱动支持任意符合接口的硬件加速器。

* **支持丰富的算子测试类型**：支持Kernel（AscendC）、GEIR（GE图编译+执行）、ACLNN（aclnn* C API）、E2E（PyTorch/torch_npu框架API）等算子测试
* **支持多种硬件类型**：E2E 模式经统一 Backend 抽象层支持 NPU、MLU、CPU 等作为待测或标杆设备（按配置 hardware segment 自动选择可用后端）
* **批量CSV用例驱动**：通过CSV文件定义测试用例，一条命令批量运行
* **多卡并行执行**：支持多NPU设备并行测试，提升测试效率
* **多种精度对比方法**：支持统计相对误差（社区标准）、数值近似、余弦相似度、二进制精确、重量化、三方交叉校验等对比方法
* **可扩展插件系统**：支持自定义Golden生成函数、输入数据生成函数（Kernel 与 ACLNN/E2E 各自命名空间）

[English Documentation](./README-EN.md)

## 🏗️ 架构层级与测试覆盖

```
┌──────────────────────────────────────────────────┐
│  应用框架层  ✅ Torch · TensorFlow · ...          │  ← E2E（端到端，覆盖全链路）
├──────────────────────────────────────────────────┤
│  引擎层     ✅ GEIR · GE · ✅ ACLNN              │  ← GEIR（GE图编译+执行）/ ACLNN（引擎 API）
├──────────────────────────────────────────────────┤
│  算子层     ✅ AiCore · AiCpu                     │  ← Kernel（算子内核，覆盖编译+执行）
└──────────────────────────────────────────────────┘
```

## 🧰 基本使用指南

- [用例生成（CSV用例编写）](./docs/用例生成.md)
- [任务执行](./docs/任务执行.md)
- [手工数据准备与回放](./docs/手工数据准备与回放.md)
- [结果分析](./docs/结果分析.md)
- [NPUSim 仿真测试（--backend npusim）](./docs/NPUSim/TTK_NPUSim使用指南.md)

## 🧪 各类算子测试详细指南

- [Kernel算子测试指南](./docs/各类算子测试指南/Kernel算子测试指南.md)
- [GEIR算子测试指南](./docs/各类算子测试指南/GEIR算子测试指南.md)
- [ACLNN算子测试指南](./docs/各类算子测试指南/ACLNN算子测试指南.md)
- [E2E算子测试指南](./docs/各类算子测试指南/E2E算子测试指南.md)

## ❓ FAQ和问题自定位指南

- [FAQ一本通](./docs/FAQ和问题自定位指南/FAQ一本通.md)

# 🌟 快速入门

## 🔧 工具安装

- 按照昇腾社区安装CANN环境：[官方链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/softwareinst/instg/instg_quick.html)
- Python版本建议选择3.8及以上

```shell
git clone https://gitcode.com/cann/ops-test-kit.git
cd ops-test-kit
pip install -r requirements.txt
```

## ✍️ 编写用例

以Kernel模式测试Add算子为例，编写CSV用例文件 `add.csv`：

```csv
testcase_name,network_name,op_name,input_shapes,input_dtypes,input_formats,output_shapes,output_dtypes,output_formats,input_ori_shapes,input_ori_formats,output_ori_shapes,output_ori_formats,attributes,input_data_ranges,precision_tolerances,absolute_precision,output_inplace_indexes,output_shape_unknown_indexes,is_enabled,remark,soc_series,priority,dump_file_prefix,manual_input_binaries,manual_golden_binaries
add_01,,add,"((128, 1024), (1, 1024))","('float32', 'float32')","('ND',)","((128, 1024),)","('float32',)","('ND',)","((128, 1024), (1, 1024))","('ND',)","((128, 1024),)","('ND',)",{},"((0, 0), (0, 0))","((0.001, 0.001),)",1e-8,(),(),True,,,0,,(),()
```

> 更多用例格式说明可参考[用例生成](./docs/用例生成.md)章节

## ▶️ 执行测试

```shell
# Kernel模式：编译 + 执行 + 精度比对
python3 -m ttk kernel -i examples/case_store/kernel/add.csv

# GEIR模式：GE图编译 + 执行 + 精度比对
python3 -m ttk geir -i examples/case_store/kernel/add.csv

# ACLNN模式
python3 -m ttk aclnn -i examples/case_store/aclnn/aclnn_cat.csv

# E2E模式（按配置 hardware segment 自动选择可用后端；--cpu 强制 cpu）
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv
python3 -m ttk e2e -i examples/case_store/e2e/torch_add.csv --cpu

# 查看设备信息
python3 -m ttk info
```

> 更多使用参数说明可执行 `python3 -m ttk kernel --help` 查看

## 📊 测试结果

运行测试后会在终端输出每个用例的精度比对结果和整体通过率等信息。

> 更多结果分析说明可参考[结果分析](./docs/结果分析.md)章节

## 🔍目录结构

```
ops-test-kit/
├── README.md              # 项目说明文档
├── LICENSE                # 许可证文件
├── pyproject.toml         # 项目配置（依赖、构建系统等）
├── requirements.txt       # 生产环境依赖
├── ttk/                   # Python 主模块（95% 代码）
├── csrc/                  # C/C++ 扩展源码（编译后供 Python 调用）
├── tests/                 # 测试代码目录
├── examples/              # 使用示例代码
│   └── case_store/        # 示例CSV用例
│       ├── kernel/        # Kernel模式示例
│       ├── aclnn/         # ACLNN模式示例
│       └── e2e/           # E2E模式示例
├── docs/                  # 文档源文件
└── scripts/               # 构建/开发辅助脚本
```

## 🤖 AI 辅助（Agent Skills）

TTK 配备了 Agent Skills，为 CLI 类 AI 编程助手（Claude Code、OpenCode 等）提供 TTK 使用指导。Skills 位于 `.claude/skills/` 目录下。

### 方式一：在 ops-test-kit 目录下启动 Agent

Agent 自动加载 `.claude/skills/` 下的 Skills：

```shell
cd ops-test-kit
claude   # 或 opencode 等其他 CLI Agent
```

### 方式二：从其他项目中使用

如果你已在算子仓库等其他目录下启动了 Agent，直接告诉它读取 TTK 的入口文件即可：

```
读取 {ops-test-kit 路径}/AGENTS.md 获取 TTK 测试框架使用指南
```

Agent 会读取 AGENTS.md 中的技能索引，按需加载对应的 SKILL.md 和 reference 文件。

> **注意**：执行 `python3 -m ttk` 命令需要在 ops-test-kit 目录下运行。

### 方式三（托底）

如果前两种方式不适用，在项目的 `CLAUDE.md` 中添加以下内容，让 Agent 在每次会话中都能感知 TTK Skills：

```markdown
## TTK 算子测试框架

TTK（ops-test-kit）是昇腾 NPU 单算子测试框架，支持 Kernel/ACLNN/E2E 三种测试模式。
- 入口文件：{ops-test-kit 路径}/AGENTS.md
- Skills 目录：{ops-test-kit 路径}/.claude/skills/
- 执行 ttk 命令时需要 cd 到 ops-test-kit 目录
```

**内置技能**：

| 技能 | 用途 |
|------|------|
| ttk-how-run-test | 构造运行命令、查看设备、参数说明 |
| ttk-how-write-case | 编写 Kernel/ACLNN/E2E 模式的 CSV 用例 |
| ttk-how-diagnose | 诊断测试失败、精度问题、编译错误 |
| ttk-how-write-plugin | 编写自定义 Golden/Input 插件 |

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/blob/master/CANN/sigs/ops-basic)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-test-kit/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-test-kit/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-test-kit/wiki)获取技术文章，如系列化教程、优秀实践等。
