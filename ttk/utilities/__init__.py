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
Precious Utility Functions
"""
from .platform import *
from .classes import *
from .string_utils import *
from .container_utils import *
from .math import *
from .format_utils import *
from .file_utils import *
from .singleton import Singleton
from .dtypes import *
from .data import RandomData
from .plog_utils import extract_plog_errors
from .proc import *
from .func_dispatch import framework_of, bind_by_name, resolve_callable_str, UnknownParamError
VERSION = "3.0.0"
FAQ = ""
