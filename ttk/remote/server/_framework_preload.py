#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""forkserver 预载用的框架加载 shim —— 让 torch 与 tensorflow 能共存于同一进程。

**为什么需要它**

`set_forkserver_preload(["tensorflow", ...])` 只是让 forkserver 进程 ``__import__`` 这些
模块名, 我们插不进任何 dlopen 标志; 而这两个框架恰恰需要控制 dlopen 才能共存。

**问题是什么**(本机实测, torch 2.10.0 + tensorflow 2.21.0)

同一进程里先 ``import torch`` 再 ``import tensorflow`` 必定 SIGSEGV; 反过来则正常。
gdb 原生栈显示崩在 tf 的 **.so 静态初始化阶段**(``call_init`` → ``_GLOBAL__sub_I_...`` →
``AssignDescriptors`` → ``ExtensionSet::RegisterMessageExtension``), 连一行 Python 都没跑到。

**根因不是 protobuf 符号冲突**(两边导出的 protobuf 符号名交集为 0 —— 版本差太远, 连
mangled name 都不同), 而是 **libstdc++ 通用模板实例化的弱符号被跨库合并**:

  - 两边都**静态**链入各自的 protobuf/absl(``ldd`` 里都没有 libprotobuf 依赖),
    却都把符号导到全局(未做 visibility=hidden);
  - 两个 .so 的导出符号有 88 个同名项, 全是 vague-linkage 弱符号 ——
    ``std::_Hashtable<std::string,...>`` 16 个、``std::vector<...>`` 19 个、typeinfo/vtable 17 个 …
  - 合并弱符号本是 C++ 的正常机制, 前提是"两边编出来的是同一份代码"; 而
    **torch 由 GCC 13.3.1/8.5.0 编、tensorflow 由 clang 18.1.8 编**, 模板实例化并不保证等价。
    先加载者胜出, 于是 tf 用 clang 生成的 descriptor 注册代码, 跑的却是 GCC 版本的
    ``_Hashtable`` 实现 → 布局对不上 → 段错误。

**解法**: 以 ``RTLD_DEEPBIND`` 加载 tensorflow, 让它优先使用**自己**那份符号, 不再被先
加载者覆盖。对照实验(torch 先加载): 普通 dlopen → core; 加 DEEPBIND → 正常, 且两个框架
随后都功能可用。这条与"谁先加载""protobuf 版本差多少"都无关, 不依赖顺序运气。

**DEEPBIND 的代价**(用到时要知道):
  - 依赖符号打桩的机制会对该库失效(LD_PRELOAD 的 malloc 拦截、tcmalloc/jemalloc 替换、
    部分 sanitizer)。故这里**只在 import tensorflow 这一次开启, 立即还原**, 不做全局设置。
  - typeinfo 不再跨库共享 ⇒ 跨框架传递 C++ 对象时 ``dynamic_cast`` / 异常捕获会失效。
    本服务端里两个框架各跑各的子进程、不交换 C++ 对象, 故安全; **若将来新增跨框架传对象
    的功能, 必须重新评估这一条**。
"""

import logging
import os
import sys
import tempfile

# glibc <dlfcn.h>: RTLD_DEEPBIND —— 优先用本库自己的符号, 不被已加载库覆盖。
# Python 未导出该常量, 按 glibc 定义硬编码(x86_64/aarch64 一致)。
_RTLD_DEEPBIND = 0x00008


def _import_isolated(module_name: str):
    """以 RTLD_DEEPBIND 导入 module_name; 该模块已导入则直接返回。"""
    if module_name in sys.modules:
        return sys.modules[module_name]
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | _RTLD_DEEPBIND)
    try:
        return __import__(module_name)
    finally:
        sys.setdlopenflags(old_flags)  # 只作用于这一次导入


class _CapturedFds:
    """临时把 fd 1/2 重定向到一个临时文件, 退出时丢弃(出错时可回吐)。

    必须在 **文件描述符** 层做, 不能只换 ``sys.stdout`` —— 这些日志绝大多数由 C 扩展
    (TF 的 absl、CANN 的 TBE 注册器)直接写 fd 1, Python 层的替换拦不住。

    为什么非拦不可: 导入 torch + tensorflow 一次会向 stdout 吐 **113,873 字节**。
    forkserver 子进程继承服务端的 stdout, 若服务端是被 ``subprocess.PIPE`` 拉起且
    调用方没有持续读管道(管道缓冲区仅 64 KB), 写操作会**永久阻塞** —— 服务端再也
    回不了包, 客户端只能等到 dispatch 死线。

    为什么不直接丢 /dev/null: 框架若在 ``.so`` 静态初始化阶段硬崩, 现场只会出现在
    C 层写 fd 2 的输出里; 丢掉就等于把最关键的诊断信息也丢了。故先落临时文件,
    正常结束才删除, 需要时用 ``tail()`` 取回。
    """

    MAX_TAIL = 4096  # 回吐时只取末尾, 避免把 100 KB 灌进日志

    def __enter__(self):
        self._tmp = tempfile.TemporaryFile()
        self._saved = (os.dup(1), os.dup(2))
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._tmp.fileno(), 1)
        os.dup2(self._tmp.fileno(), 2)
        return self

    def tail(self):
        """取回被拦下的输出末尾(供失败时写进日志)。"""
        try:
            self._tmp.seek(max(0, self._tmp.tell() - self.MAX_TAIL))
            return self._tmp.read().decode("utf-8", "replace")
        except Exception:  # noqa: BLE001  取诊断信息失败不该影响主流程
            return ""

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        for fd in self._saved:
            os.close(fd)
        return False

    def close(self):
        self._tmp.close()


def preload():
    """预载可用框架。任一框架不可用只记日志, 不影响其余框架与服务。"""
    loaded = []
    # tensorflow 走 DEEPBIND —— 它是"在 .so 静态初始化阶段就重度使用被合并符号"的一方,
    # 也是实测中会崩的一方。torch 按常规导入即可。
    import importlib.util

    cap = _CapturedFds()
    failures = []
    with cap:
        if importlib.util.find_spec("tensorflow") is not None:
            try:
                _import_isolated("tensorflow")
                loaded.append("tensorflow")
            except Exception as e:  # noqa: BLE001  预载失败只降级, 绝不让服务不可用
                failures.append(("tensorflow", e))
        if importlib.util.find_spec("torch") is not None:
            try:
                __import__("torch")
                loaded.append("torch")
            except Exception as e:  # noqa: BLE001
                failures.append(("torch", e))
    # 还原 fd 之后再记日志, 否则日志本身也会被拦进临时文件
    for name, err in failures:
        logging.warning("preload %s failed (%s), skipped; captured output tail:\n%s", name, err, cap.tail())
    cap.close()
    return loaded


preload()
