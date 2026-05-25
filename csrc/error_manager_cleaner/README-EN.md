# libttk_error_manager_cleaner.so

## Overview

Clears CANN ErrorManager residual errors to prevent cross-testcase error contamination.

TTK uses a `forkserver` multiprocessing model where a single worker process executes multiple testcases sequentially. The CANN ErrorManager runs in `PROCESS_MODE`, sharing an error container across all testcases. If a previous testcase fails and leaves errors uncleaned, the subsequent testcase's tiling phase may read stale errors and incorrectly report failure.

This module calls `GetErrMgrRawErrorMessages()` before each testcase to read and clear any residual errors.

## Exported API

```c
int ClearErrorManager(void);
```

Used by Python via `ttk.core_modules.npu.error_cleaner.clear_error_manager()`, wired into `compilation_process` and `profile_process` entry points.

## Build

```bash
# Standalone
cmake -S csrc/error_manager_cleaner -B csrc/error_manager_cleaner/build
cmake --build csrc/error_manager_cleaner/build

# Top-level (whl packaging)
cmake -S csrc -B csrc/build
cmake --build csrc/build
```

## Dependencies

| Dependency | Type | Source |
|------------|------|--------|
| `error_manager` | Link | CANN (`$ASCEND_HOME/lib64`) |
| `base/err_mgr.h` | Header | CANN (`$ASCEND_HOME/{arch}-linux/pkg_inc/`) |
