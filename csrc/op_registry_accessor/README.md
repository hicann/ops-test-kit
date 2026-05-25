# libttk_op_registry_accessor.so

## 功能

用于二进制匹配时调用 CANN 注册的定制 simplifiedKey 生成函数。

TTK 在算子编译流程中需要为每个算子生成 simplified key（简化键），部分算子（MatMulV3、BatchMatMulV3 等）注册了自定义 `gen_simplifiedkey` 回调，需要构造 `TilingContext` 作为入参来调用。本模块封装了查找回调函数和调用的完整流程。

## 导出 API

```c
// 查找算子的 gen_simplifiedkey 函数句柄
int FindGenSimplifiedKeyFuncs(const char *op_type, void **handle);

// 调用已查找的句柄生成 simplified key
int InvokeGenSimplifiedKey(void *handle, const char *op_type,
                           const char *inputs, const char *outputs,
                           const char *attrs, const char *extra_params,
                           char *result_buf);
```

Python 端通过 `ttk.core_modules.operator.registries` 使用，按需自动编译加载。

## 构建

```bash
# Standalone
cmake -S csrc/op_registry_accessor -B csrc/op_registry_accessor/build
cmake --build csrc/op_registry_accessor/build

# Top-level (whl 发布)
cmake -S csrc -B csrc/build
cmake --build csrc/build
```

## 依赖

| 依赖 | 类型 | 来源 |
|------|------|------|
| `opp_registry` | 链接 | CANN (`$ASCEND_HOME/lib64`) |
| `exe_graph` | 链接 | CANN (`$ASCEND_HOME/lib64`) |
| `nlohmann/json.hpp` | 头文件 | `csrc/third_party/nlohmann/` |
