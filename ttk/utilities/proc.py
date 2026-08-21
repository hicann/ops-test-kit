#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Process Related Utilities
"""


__all__ = ["set_process_name", "get_process_name", "is_main_process",
           "set_thread_name", "process_exists", "waiting_for_memory",
           "append_ld_library_path", "signal_registered",
           "cpu_count", "insert_env_path", "add_exec_permission",
           "get_ubuntu_main_version", "insert_python_path",
           "msdebug_runtime_injection_enabled", "kernel_debug_compile_enabled",
           ]


# Standard Packages
import os
import pathlib
import re
import shutil
import stat
import subprocess
import time
from typing import Union

import psutil
import threading
import multiprocessing


def set_process_name(name: str = "MP"):
    """Set process name for logging"""
    multiprocessing.current_process().name = name


def get_process_name() -> str:
    """Set process name for logging"""
    return multiprocessing.current_process().name


def set_thread_name(name: str = "MT"):
    """Set current thread name for threading"""
    threading.current_thread().setName(name)


def is_main_process() -> bool:
    return os.getpid() == int(os.getenv('TTK_PARENT_PID', "0"))


def process_exists(pid: int):
    """
    Check whether @pid exists
    """
    try:
        os.kill(pid, 0)
        return True
    except:
        return False


def waiting_for_memory():
    import logging
    GB = 1024 * 1024 * 1024
    print_once = False
    while (psutil.virtual_memory().available <= psutil.virtual_memory().total * 0.5 and
            psutil.virtual_memory().available <= 128 * GB):
        if not print_once:
            logging.warning(f"Task paused because of insufficient memory, "
                            f"available: {psutil.virtual_memory().available / GB} GB, "
                            f"total: {psutil.virtual_memory().total / GB} GB")
            print_once = True
        time.sleep(1)
        if os.path.exists("/tmp/TTK_FORCE_MEMORY_OVERRIDE"):
            logging.warning("Detected OVERRIDE signal, continue")
            return


def insert_env_xpath(path: Union[pathlib.Path, str], env_name):
    env_xpath = os.getenv(env_name, '')
    lists = env_xpath.split(':')
    if isinstance(path, str):
        path = pathlib.Path(path)
    abs_path = path.resolve()
    if abs_path not in lists:
        os.environ[env_name] = f"{abs_path}:{env_xpath}"


def append_ld_library_path(path: Union[pathlib.Path, str]):
    insert_env_xpath(path, 'LD_LIBRARY_PATH')


def insert_env_path(path: Union[pathlib.Path, str]):
    insert_env_xpath(path, 'PATH')


def insert_python_path(path: Union[pathlib.Path, str]):
    insert_env_xpath(path, 'PYTHONPATH')


def add_exec_permission(file_path):
    cur_permissions = os.stat(file_path).st_mode
    new_permissions = cur_permissions | stat.S_IXUSR
    os.chmod(file_path, new_permissions)


def signal_registered(signum: int) -> bool:
    pid = os.getpid()
    status_file = f"/proc/{pid}/status"
    signal_fields = ["SigCgt", "SigIgn", "SigBlk"]

    if not os.path.exists(status_file):
        return False

    signal_data = {}
    with open(status_file, "r") as f:
        for line in f:
            for field in signal_fields:
                if line.startswith(field + ":"):
                    hex_str = line.split(":")[1].strip()
                    bitmask = int(hex_str, 16)
                    signal_data[field] = bitmask

    results = {}
    for field, bitmask in signal_data.items():
        # signum stars from 1. bit is `signum - 1`
        is_set = (bitmask >> (signum - 1)) & 1
        results[field] = bool(is_set)
    return any(results.values())


def cpu_count() -> int:
    return len(os.sched_getaffinity(0))


def get_ubuntu_main_version_via_lsb_release():
    if shutil.which('lsb_release') is None:
        return None

    try:
        result = subprocess.run(['lsb_release', '-r'],
                                capture_output=True, text=True,
                                shell=False)
        if result.returncode == 0:
            # Release:	18.04
            version_str = result.stdout.strip()
            match = re.search(r'(\d+)\.\d+', version_str)
            if match:
                return int(match.group(1))
    except Exception:
        pass
    return None


def get_ubuntu_main_version_via_etc_files():
    etc_files = ['/etc/os-release', '/etc/lsb-release', '/usr/lib/os-release']
    for file_path in etc_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                if 'ubuntu' in content.lower():
                    patterns = [
                        r'VERSION_ID\s*=\s*"(\d+)\.\d+"',
                        r'DISTRIB_RELEASE\s*=\s*(\d+)\.\d+'
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            return int(match.group(1))
        except Exception:
            continue
    return None


def get_ubuntu_main_version() -> int:
    v = get_ubuntu_main_version_via_lsb_release()
    if v is None:
        v = get_ubuntu_main_version_via_etc_files()
    return v


_REGBASE_V2_SOURCE_DEBUG_OPTION = "--cce-ignore-always-inline=false"


def msdebug_runtime_injection_enabled() -> bool:
    """
    Return whether the current process runs under msdebug runtime injection.

    msdebug launches the target process with:
      - MSOP_SOCKET_PATH=<socket> (stub<->lldb communication)
      - LD_PRELOAD=<...>/libruntime_stub.so (runtime hijack)
    Either signal indicates RTS calls should resolve through the process-global
    symbol table so that the preloaded stub intercepts them.
    """
    if os.getenv("MSOP_SOCKET_PATH"):
        return True
    return _msdebug_runtime_stub_preloaded()


def _msdebug_runtime_stub_preloaded() -> bool:
    """Return whether msdebug's runtime interposition library is preloaded."""
    preload = os.getenv("LD_PRELOAD", "")
    return any(
        os.path.basename(path) == "libruntime_stub.so"
        for path in preload.replace(":", " ").split()
    )


def kernel_debug_compile_enabled() -> bool:
    """
    Return whether TTK should compile kernels with source-level debug info (-g -O0).

    MSOP_SOCKET_PATH is shared by other runtime interposers (for example
    msopprof), so only msdebug's preloaded runtime stub is a sufficiently
    specific signal for changing kernel compilation to -O0 -g.
    """
    return _msdebug_runtime_stub_preloaded()
