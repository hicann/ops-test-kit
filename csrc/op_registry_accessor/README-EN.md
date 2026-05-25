# libttk_op_registry_accessor.so

## Overview

Invokes CANN-registered custom simplifiedKey generator functions for binary matching.

During operator compilation, TTK generates a simplified key for each operator. Some operators (MatMulV3, BatchMatMulV3, etc.) register custom `gen_simplifiedkey` callbacks that require a `TilingContext` as input. This module encapsulates the full workflow of looking up and invoking these callbacks.

## Exported API

```c
// Look up the gen_simplifiedkey function handle for an operator
int FindGenSimplifiedKeyFuncs(const char *op_type, void **handle);

// Invoke a previously looked-up handle to generate a simplified key
int InvokeGenSimplifiedKey(void *handle, const char *op_type,
                           const char *inputs, const char *outputs,
                           const char *attrs, const char *extra_params,
                           char *result_buf);
```

Used by Python via `ttk.core_modules.operator.registries`, with on-demand compilation and loading.

## Build

```bash
# Standalone
cmake -S csrc/op_registry_accessor -B csrc/op_registry_accessor/build
cmake --build csrc/op_registry_accessor/build

# Top-level (whl packaging)
cmake -S csrc -B csrc/build
cmake --build csrc/build
```

## Dependencies

| Dependency | Type | Source |
|------------|------|--------|
| `opp_registry` | Link | CANN (`$ASCEND_HOME/lib64`) |
| `exe_graph` | Link | CANN (`$ASCEND_HOME/lib64`) |
| `nlohmann/json.hpp` | Header | `csrc/third_party/nlohmann/` |
