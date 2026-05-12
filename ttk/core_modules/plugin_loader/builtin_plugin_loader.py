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
Builtin plugin loader - loads loads from builtin registries on demand
"""

import logging
from typing import Set, Optional

from ...utilities import Singleton
from .custom_plugin_manager import CustomPluginManager


class BuiltinPluginLoader(metaclass=Singleton):
    """Builtin plugin loader - loads from builtin registries on demand"""

    def __init__(self):
        self._registry = None
        
    def _ensure_loaded(self, plugin_type: str, level_type: str):
        """
        Ensure builtin registry loaded for specific type (lazy import)
        
        Args:
            plugin_type: "golden" or "input"
            level_type: "kernel" or "aclnn"
        """
        if self._registry is None:
            self._registry = {}
    
        if plugin_type not in self._registry:
            self._registry[plugin_type] = {}

        if level_type not in self._registry[plugin_type]:
            # Only import the needed registry
            if plugin_type == CustomPluginManager.PLUGIN_TYPE_GOLDEN:
                if level_type == CustomPluginManager.LEVEL_KERNEL:
                    from ttk.user_defined_modules.op.golden_funcs.registry import golden_funcs
                    self._registry[plugin_type][level_type] = golden_funcs
                    logging.info(f"Loaded builtin golden registry for type: {level_type}")
                elif level_type == CustomPluginManager.LEVEL_ACLNN:
                    from ttk.user_defined_modules.op_api.golden_funcs.registry import golden_funcs
                    self._registry[plugin_type][level_type] = golden_funcs
                    logging.info(f"Loaded builtin golden registry for type: {level_type}")
                elif level_type == CustomPluginManager.LEVEL_E2E:
                    self._registry[plugin_type][level_type] = {}
            elif plugin_type == CustomPluginManager.PLUGIN_TYPE_INPUTS:
                if level_type == CustomPluginManager.LEVEL_KERNEL:
                    from ttk.user_defined_modules.op.input_funcs.registry import special_input_func
                    self._registry[plugin_type][level_type] = special_input_func
                    logging.info(f"Loaded builtin input registry for type: {level_type}")
                elif level_type == CustomPluginManager.LEVEL_ACLNN:
                    from ttk.user_defined_modules.op_api.input_funcs.registry import special_input_func
                    self._registry[plugin_type][level_type] = special_input_func
                    logging.info(f"Loaded builtin input registry for type: {level_type}")
                elif level_type == CustomPluginManager.LEVEL_E2E:
                    self._registry[plugin_type][level_type] = {}
        
    def get_available_operators(self, plugin_type: str, level_type: str) -> Set[str]:
        """Get available kernel operators"""
        self._ensure_loaded(plugin_type, level_type)
        return set(self._registry[plugin_type][level_type].keys())
        
    def load_if_available(self, operator_name: str, 
                       plugin_type: str, level_type: str) -> Optional[callable]:
        """Load if available"""
        self._ensure_loaded(plugin_type, level_type)
        
        registry = self._registry[plugin_type][level_type]
        
        if operator_name in registry:
            func = registry[operator_name]
            logging.debug(f"Using builtin {plugin_type}: {level_type}.{operator_name}")
            return func
        else: 
            return None
