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
Custom plugin manager with configuration-driven support
"""

import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ...utilities import Singleton
from .scanner import PluginScanner


class CustomPluginManager(metaclass=Singleton):
    """Custom plugin manager with configuration-driven support"""

    PLUGIN_TYPE_GOLDEN = "golden"
    PLUGIN_TYPE_INPUTS = "input"
    LEVEL_KERNEL = "kernel"
    LEVEL_ACLNN = "aclnn"
    LEVEL_E2E = "e2e"
    FIELD_GOLDEN = "__golden__"
    FIELD_INPUTS = "__input__"

    PLUGIN_CONFIG = {
        PLUGIN_TYPE_GOLDEN: {
            "golden_types": [LEVEL_KERNEL, LEVEL_ACLNN, LEVEL_E2E],
            "field_name": FIELD_GOLDEN,
            "builtin_registries": {
                LEVEL_KERNEL: ("ttk.user_defined_modules.op.golden_funcs.registry", "golden_funcs"),
                LEVEL_ACLNN: ("ttk.user_defined_modules.op_api.golden_funcs.registry", "golden_registry")
            }
        },
        PLUGIN_TYPE_INPUTS: {
            "golden_types": [LEVEL_KERNEL, LEVEL_ACLNN, LEVEL_E2E],
            "field_name": FIELD_INPUTS,
            "builtin_registries": {
                LEVEL_KERNEL: ("ttk.user_defined_modules.op.input_funcs.registry", "input_funcs"),
                LEVEL_ACLNN: ("ttk.user_defined_modules.op_api.input_funcs.registry", "input_registry")
            }
        }
    }

    def __init__(self, plugin_path=None):
        self.plugin_path = plugin_path

        self._scan_cache = {
            self.PLUGIN_TYPE_GOLDEN: {
                self.LEVEL_KERNEL: None,
                self.LEVEL_ACLNN: None,
                self.LEVEL_E2E: None
            },
            self.PLUGIN_TYPE_INPUTS: {
                self.LEVEL_KERNEL: None,
                self.LEVEL_ACLNN: None,
                self.LEVEL_E2E: None
            }
        }

        self._loaded_funcs = {
            self.PLUGIN_TYPE_GOLDEN: {
                self.LEVEL_KERNEL: {},
                self.LEVEL_ACLNN: {},
                self.LEVEL_E2E: {}
            },
            self.PLUGIN_TYPE_INPUTS: {
                self.LEVEL_KERNEL: {},
                self.LEVEL_ACLNN: {},
                self.LEVEL_E2E: {}
            }
        }

        logging.info(f"CustomPluginManager initialized with plugin_path: {plugin_path}")
        
    def _get_config(self, plugin_type: str) -> dict:
        """Get plugin type configuration"""
        if plugin_type not in self.PLUGIN_CONFIG:
            raise ValueError(f"Unsupported plugin type: {plugin_type}")
        return self.PLUGIN_CONFIG[plugin_type]
        
    def _get_field_name(self, plugin_type: str) -> str:
        """Get field name"""
        config = self._get_config(plugin_type)
        return config["field_name"]
        
    def _get_supported_golden_types(self, plugin_type: str) -> List[str]:
        """Get supported golden types"""
        config = self._get_config(plugin_type)
        return config["golden_types"]
        
    def _is_valid_golden_type(self, plugin_type: str, level_type: str) -> bool:
        """Validate golden type"""
        return level_type in self._get_supported_golden_types(plugin_type)
        
    def _ensure_scanned(self, plugin_type: str, level_type: str):
        """Ensure scanned"""
        if not self._is_valid_golden_type(plugin_type, level_type):
            raise ValueError(f"Invalid golden type {level_type} for plugin type {plugin_type}")
            
        if self._scan_cache[plugin_type][level_type] is not None:
            return
            
        if not self.plugin_path:
            self._scan_cache[plugin_type][level_type] = {}
            logging.info(f"No plugin_path, custom {plugin_type} cache empty")
            return
            
        logging.info(f"Scanning custom {plugin_type} for type: {level_type}")
        
        field_name = self._get_field_name(plugin_type)
        scanner = PluginScanner(self.plugin_path, plugin_type, field_name)
        self._scan_cache[plugin_type][level_type] = scanner.scan()
        
        logging.info(f"Scanned {len(self._scan_cache[plugin_type][level_type])} custom {plugin_type}.{level_type} functions")
        
    def get_available_operators(self, plugin_type: str, level_type: str) -> Set[str]:
        """Get available operators"""
        self._ensure_scanned(plugin_type, level_type)
        return set(self._scan_cache[plugin_type][level_type].keys())
        
    def load_if_available(self, operator_name: str, 
                       plugin_type: str, level_type: str) -> Optional[callable]:
        """Load if available"""
        if not self._is_valid_golden_type(plugin_type, level_type):
            raise ValueError(f"Invalid golden type {level_type} for plugin type {plugin_type}")
            
        if operator_name in self._loaded_funcs[plugin_type][level_type]:
            return self._loaded_funcs[plugin_type][level_type][operator_name]
            
        self._ensure_scanned(plugin_type, level_type)
        
        scan_results = self._scan_cache[plugin_type][level_type]
        
        if operator_name in scan_results:
            module_name, func_name, file_path = scan_results[operator_name]
            
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, str(file_path)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                func = getattr(module, func_name)
                self._loaded_funcs[plugin_type][level_type][operator_name] = func
                logging.info(f"Loaded custom {plugin_type}: {level_type}.{operator_name} from {file_path}")
                return func
                
            except Exception as e:
                logging.error(f"Failed to load custom {plugin_type} {operator_name}: {e}")
                
        return None
