#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Special input data generation rules"""
# Standard Packages
import logging
from typing import Sequence
from functools import wraps

special_input_func = {}


def register_input(operator_names: Sequence[str]):
    """Register input function"""

    def __inner_input_registry(func):
        @wraps(func)
        def __wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        for operator_name in operator_names:
            if operator_name in special_input_func:
                logging.warning("input function of %s has already been registered!" % operator_name)
            special_input_func[operator_name] = __wrapper
        return __wrapper

    return __inner_input_registry
