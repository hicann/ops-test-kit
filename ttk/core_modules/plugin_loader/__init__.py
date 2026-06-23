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
Plugin loader module - unified interface for loading custom and builtin plugins
"""

__all__ = ["get_plugin_function"]

from typing import Optional, Callable, Set, Tuple, Dict

from .custom_plugin_manager import CustomPluginManager
from .builtin_plugin_loader import BuiltinPluginLoader
from .spec_loader import load_spec_function

_func_cache: Dict[Tuple[str, str, str, Optional[str]], Tuple[Optional[Callable], Optional[str]]] = {}


def get_plugin_function(operator_name: str,
                     plugin_type: str = "golden",
                     level_type: str = "kernel",
                     plugin_path: Optional[str] = None) -> Tuple[Optional[Callable], Optional[str]]:
    """
    Get plugin function by priority

    Priority:
        1. TestSpec (TestSpecManager)
        2. Custom plugin
        3. Builtin plugin

    Args:
        operator_name: Operator name
        plugin_type: "golden" or "input"
        level_type: "kernel" or "aclnn"
        plugin_path: Custom plugin path

    Returns:
        (func, source) tuple, or (None, None) if not found
        func: Callable object
        source: "spec", "custom" or "builtin"
    """
    cache_key = (operator_name, plugin_type, level_type, str(plugin_path) if plugin_path else None)
    if cache_key in _func_cache:
        return _func_cache[cache_key]

    func = None
    source = None

    # 1. TestSpec（最高优先级）
    func, source = load_spec_function(operator_name, plugin_type, plugin_path)

    # 2. Custom plugin
    if func is None and plugin_path:
        custom_manager = CustomPluginManager(plugin_path)
        custom_func = custom_manager.load_if_available(operator_name, plugin_type, level_type)
        if custom_func:
            func = custom_func
            source = "custom"

    # 3. Builtin plugin
    if func is None:
        builtin_loader = BuiltinPluginLoader()
        builtin_func = builtin_loader.load_if_available(operator_name, plugin_type, level_type)
        if builtin_func:
            func = builtin_func
            source = "builtin"

    _func_cache[cache_key] = (func, source)
    return (func, source)


def get_available_operators(plugin_type: str = "golden",
                          level_type: str = "kernel",
                          plugin_path: Optional[str] = None) -> Set[str]:
    """
    Get available kernel functions
    
    Returns:
        Available kernel functions set (custom ∪ builtin)
    """
    available = set()
    
    # 1. Custom plugin
    if plugin_path:
        custom_manager = CustomPluginManager(plugin_path)
        available.update(custom_manager.get_available_operators(plugin_type, level_type))
    
    # 2. Builtin plugin
    builtin_loader = BuiltinPluginLoader()
    available.update(builtin_loader.get_available_operators(plugin_type, level_type))
    
    return available
