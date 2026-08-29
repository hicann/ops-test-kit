#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Spec loader - TestSpec public API adapter for plugin_loader
"""

from typing import Callable, Optional


def load_spec_function(operator_name: str, plugin_type: str, plugin_path) -> Optional[Callable]:
    """从 TestSpec 加载 golden/input 函数（通过 get_spec_attr）。"""
    from ttk.test_spec import get_spec_attr

    attr_name = "customize_inputs" if plugin_type == "input" else plugin_type
    return get_spec_attr(operator_name, attr_name, plugin_path)
