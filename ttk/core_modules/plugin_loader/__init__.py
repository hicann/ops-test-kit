#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License License Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Plugin loader module - unified interface for loading TestSpec and custom plugins
"""

__all__ = ["get_plugin_function"]

from typing import Callable, Dict, Optional, Tuple

from .custom_plugin_manager import CustomPluginManager
from .spec_loader import load_spec_function

_func_cache: Dict[Tuple[str, str, str, Optional[str]], Optional[Callable]] = {}


def get_plugin_function(
    operator_name: str, plugin_type: str = "golden", level_type: str = "kernel", plugin_path: Optional[str] = None
) -> Optional[Callable]:
    """
    Get plugin function by priority

    Priority:
        1. TestSpec (TestSpecManager)
        2. Custom plugin

    Args:
        operator_name: Operator name
        plugin_type: "golden" or "input"
        level_type: "kernel" or "aclnn"
        plugin_path: Custom plugin path

    Returns:
        func, or None if not found
    """
    cache_key = (operator_name, plugin_type, level_type, str(plugin_path) if plugin_path else None)
    if cache_key in _func_cache:
        return _func_cache[cache_key]

    func = None

    # 1. TestSpec（最高优先级）
    func = load_spec_function(operator_name, plugin_type, plugin_path)

    # 2. Custom plugin
    if func is None and plugin_path:
        custom_manager = CustomPluginManager(plugin_path)
        custom_func = custom_manager.load_if_available(operator_name, plugin_type, level_type)
        if custom_func:
            func = custom_func

    _func_cache[cache_key] = func
    return func
