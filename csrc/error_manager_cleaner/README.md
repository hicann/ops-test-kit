# libttk_error_manager_cleaner.so

## 功能

清理 CANN ErrorManager 残留错误，防止跨用例错误污染。

TTK 使用 `forkserver` 多进程模型，同一 worker 进程串行执行多个用例。CANN ErrorManager 以 `PROCESS_MODE` 运行，所有用例共享错误容器。若前一个用例执行异常产生错误但未清理，后续用例的 tiling 阶段会读到残留错误导致误判失败。

本模块在每个用例执行前调用 `GetErrMgrRawErrorMessages()` 读取并清空残留错误。

## 导出 API

```c
int ClearErrorManager(void);
```

Python 端通过 `ttk.core_modules.npu.error_cleaner.clear_error_manager()` 使用，已接入 `compilation_process` 和 `profile_process` 入口。

## 构建

```bash
# Standalone
cmake -S csrc/error_manager_cleaner -B csrc/error_manager_cleaner/build
cmake --build csrc/error_manager_cleaner/build

# Top-level (whl 发布)
cmake -S csrc -B csrc/build
cmake --build csrc/build
```

## 依赖

| 依赖 | 类型 | 来源 |
|------|------|------|
| `error_manager` | 链接 | CANN (`$ASCEND_HOME/lib64`) |
| `base/err_mgr.h` | 头文件 | CANN (`$ASCEND_HOME/{arch}-linux/pkg_inc/`) |
