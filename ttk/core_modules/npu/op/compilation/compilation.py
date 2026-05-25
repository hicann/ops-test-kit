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
Operator compilation method for universal csv testcases
"""
# Standard Packages
import os

# Third-Party Packages
from ....testcase_manager import TestcaseOp
from ....tbe_multiprocessing import get_process_context
from .dynamic_compilation import dynamic_compilation
from .static_compilation import static_compilation


def compilation_process(testcase: TestcaseOp, mode: str, **_):
    """
    Universal Operator Compilation Sequence
    :param mode:
    :param testcase:
    :return:
    """
    from ...error_cleaner import clear_error_manager
    clear_error_manager()
    get_process_context().notify_status("InitCompilation")
    get_process_context().change_name(testcase.testcase_name)
    if not testcase.is_valid:
        return None

    if mode in ("Dyn", "Bin"):
        return dynamic_compilation(testcase, mode)
    elif mode == "Cst":
        return static_compilation(testcase, mode)
    else:
        raise RuntimeError("Unknown mode %s" % mode)
