"""T6 Step 6: 合规 grep 测试——5 文件零硬编码厂商术语字面量。

spec §4.5：5 文件代码/注释/docstring 零硬编码 nvidia/gpu/cuda/mlu/cambricon/musa/
npu/davinci 厂商术语（全配置驱动）。

用词边界 ``\\b`` + IGNORECASE 排除误报：
  - 局部变量 ``gpu_ids``/``gpu_count``（``_`` 是 \\w，无边界）→ 不命中
  - ``input_schema``/``_gpu_locks``/``--gpus``/``torch_cuda``/``*_VISIBLE_DEVICES`` → 不命中
  - spec 语法 ``/device:TYPE:N``/``{lib}:{id}`` 不含厂商词 → 不命中

只命中独立厂商词字面量。spec §1.1 / §4.5 / §7 合规 grep。
"""
import os
import re

import pytest

# 词边界 + IGNORECASE（spec §4.5 五词 + gpu/cuda）。``\\b`` 排除标识符内子串误报。
VENDOR = re.compile(r'\b(nvidia|gpu|cuda|mlu|cambricon|musa|npu|davinci)\b', re.IGNORECASE)

# spec §2 范围：server 5 文件。
_SERVER_DIR = os.path.join("ttk", "remote", "server")
FIVE_FILES = [
    os.path.join(_SERVER_DIR, "config.py"),
    os.path.join(_SERVER_DIR, "executor.py"),
    os.path.join(_SERVER_DIR, "execution_container.py"),
    os.path.join(_SERVER_DIR, "container.py"),
    os.path.join(_SERVER_DIR, "xpu_server.py"),
]


def test_no_vendor_terms():
    """5 文件零硬编码厂商术语（含注释/docstring，全配置驱动）。

    失败时打印命中清单（path:line: content），定位清洗。
    """
    hits = []
    for path in FIVE_FILES:
        with open(path) as f:
            for i, line in enumerate(f, 1):
                if VENDOR.search(line):
                    hits.append(f"{path}:{i}: {line.strip()}")
    assert not hits, "vendor terms found:\n" + "\n".join(hits)


def test_five_files_exist():
    """守卫：FIVE_FILES 路径正确（防重构后路径漂移致 test_no_vendor_terms 静默空过）。"""
    for path in FIVE_FILES:
        assert os.path.isfile(path), f"FIVE_FILES path missing: {path}"
