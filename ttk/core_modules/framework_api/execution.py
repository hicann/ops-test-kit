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
API execution utilities.

Parameter assembly (ParamPlan, match_overload, coerce_value, build_positional_args)
has been moved to ttk.core_modules.testcase_manager.param_plan.
"""
import logging


def call_api(api_name, matched_oidx, resolved, args, kwargs):
    """Call resolved API with error logging.

    Safe wrapper that logs TypeError details before re-raising.
    Used by profiling.py and golden_generation.py.
    """
    try:
        return resolved(*args, **kwargs)
    except TypeError as e:
        logging.warning(
            f"API call {api_name}(oidx={matched_oidx}) failed with TypeError: {e}. "
            f"args={len(args)}, kwargs={list(kwargs.keys())}"
        )
        raise
