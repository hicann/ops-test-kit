#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.

"""
Spec loader - TestSpecManager adapter for plugin_loader
"""

import logging
from typing import Optional, Callable, Tuple


_spec_manager = None
_spec_search_paths = None


def load_spec_function(operator_name: str,
                          plugin_type: str,
                          plugin_path) -> Tuple[Optional[Callable], Optional[str]]:
    """从 TestSpecManager 加载 golden/input 函数。

    Args:
        operator_name: 算子名
        plugin_type: "golden" 或 "input"
        plugin_path: 插件搜索路径（str/path/tuple/list）

    Returns:
        (func, "spec") 或 (None, None)
    """
    mgr = _get_or_create_manager(plugin_path)
    if mgr is None:
        return (None, None)

    cls = mgr.load(operator_name)
    if cls is None:
        return (None, None)

    attr_name = "customize_inputs" if plugin_type == "input" else plugin_type
    if not mgr.has(cls, attr_name):
        return (None, None)

    func = mgr.get(cls, attr_name)
    if func is not None:
        logging.debug(f"Spec found for {operator_name}.{attr_name}")
        return (func, "spec")

    return (None, None)


def _get_or_create_manager(plugin_path):
    """懒创建 TestSpecManager。路径不变时复用已有实例。"""
    global _spec_manager, _spec_search_paths

    if plugin_path is None:
        return None

    if isinstance(plugin_path, (list, tuple)):
        paths = [str(p) for p in plugin_path]
    else:
        paths = [str(plugin_path)]

    if paths == _spec_search_paths and _spec_manager is not None:
        return _spec_manager

    from ttk.test_spec import TestSpecManager
    _spec_manager = TestSpecManager(search_paths=paths)
    _spec_search_paths = paths
    return _spec_manager
