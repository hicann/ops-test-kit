# AGENTS.md - TTK 算子测试指南

## 定位

面向算子开发者的 TTK 使用指南。适用于 Claude Code、OpenCode 等 CLI 类 AI 编程助手。TTK 当前面向昇腾 NPU 单算子测试，支持 Kernel 编译执行、ACLNN API 测试、E2E 框架 API 对比三种测试模式。

## 架构层级

TTK 的三种测试模式对应昇腾技术栈的不同层级，越往上覆盖通路越全：

```
┌──────────────────────────────────────────────────┐
│  应用框架层  ✅ Torch · TensorFlow · ...          │  ← E2E（端到端，覆盖全链路）
├──────────────────────────────────────────────────┤
│  引擎层     GE · ✅ ACLNN                        │  ← ACLNN（引擎 API，覆盖编译+执行）
├──────────────────────────────────────────────────┤
│  算子层     ✅ AiCore · AiCpu                     │  ← Kernel（算子内核，覆盖编译+执行）
└──────────────────────────────────────────────────┘
```

不同层级使用不同的测试用例结构，模式选错会导致用例解析失败。根据 CSV 表头自动判断模式（详见各 skill）。

## 使用方式

执行 TTK 命令需要在本仓库（ops-test-kit）目录下运行：
- Kernel 模式：`python3 -m ttk kernel -i cases.csv`
- ACLNN 模式：`python3 -m ttk aclnn -i cases.csv`
- E2E 模式：`python3 -m ttk e2e -i cases.csv`（默认自动探测 NPU；`--cpu` 强制 CPU）

验证 NPU 环境：`python3 -m ttk info`

## 技能索引

| 任务 | 技能 | 触发场景 |
|------|------|---------|
| 运行测试 | ttk-how-run-test | 运行ttk、执行测试、查看设备、ttk info/list |
| 编写用例 | ttk-how-write-case | 写用例、生成CSV、CSV字段、TensorList、动态shape |
| 诊断失败 | ttk-how-diagnose | 测试失败、精度不通过、编译报错、执行超时、排障 |
| 编写插件 | ttk-how-write-plugin | TestSpec、__spec__、算子测试规范编写 |

> 读取 `.claude/skills/{技能名}/SKILL.md` 获取详细工作流程。

## 关键路径

- 用例示例：`examples/case_store/`
- 详细文档：`docs/`
- 插件示例：算子仓库 `tests/assets/` 目录下的 `golden.py` / `input.py`

## Commit 规范

- commit message 中不要添加 Co-Authored-By 行
