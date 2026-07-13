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
ttk entry point
"""
# Import all modules
# noinspection PyBroadException
try:
    import os
    if os.getenv("TTK_LOAD_TF") == "1":
        print("Loading tensorflow...")
        __import__("tensorflow")
except:
    print("Tensorflow load failed")
from . import utilities
from . import core_modules
from .core_modules import runtime
from .core_modules.runtime import RTSInterface
from .core_modules.testcase_manager import TestcaseOp
from .core_modules.testcase_manager import TestcaseAclnn
