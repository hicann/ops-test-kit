#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


__all__ = [
    "MAX_COMPARE_FAILURE_PRINT",
    "compare_log_size",
    "read_compare_log_failures",
    "print_compare_log_failures",
]

import logging
import os
from typing import Optional

MAX_COMPARE_FAILURE_PRINT = 20
COMPARE_LOG_PATH = "ttk-compare.log"


def compare_log_size() -> int:
    try:
        return os.path.getsize(COMPARE_LOG_PATH)
    except OSError:
        return 0


def read_compare_log_failures(start_size: int) -> tuple:
    if not os.path.exists(COMPARE_LOG_PATH):
        return [], start_size
    try:
        with open(COMPARE_LOG_PATH, "rb") as f:
            f.seek(start_size)
            data = f.read()
    except OSError:
        return [], start_size
    diff_lines = []
    for line in data.splitlines():
        text = line.decode("utf-8", errors="replace")
        if "Index:" in text and "Diff:" in text:
            diff_lines.append(text.rstrip("\n"))
    return diff_lines, start_size + len(data)


def print_compare_log_failures(diff_lines, testcase_name: Optional[str] = None):
    diff_lines = diff_lines[:MAX_COMPARE_FAILURE_PRINT]
    if diff_lines:
        header = testcase_name
        logging.info(
            "\n=========== %s (first %d) ===========\n%s",
            header,
            len(diff_lines),
            "\n".join(diff_lines),
        )
